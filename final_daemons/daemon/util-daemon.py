import os
import time
import threading
from datetime import datetime
import docker
import sqlite3
import json


UTIL_TIMESTEP = 1 # Get utilization metrics per container every 10 milliseconds
NEW_CONTAINER_POLLING_TIMESTEP =  0.1 # Check for new container creation every 100 milliseconds

# Function to monitor memory and CPU utilization for a container using cgroups
def monitor_container_cgroups(container_id):
    db_conn = sqlite3.connect('invoker_data.db')
    cursor = db_conn.cursor()

    memory_cgroup_path = f"/sys/fs/cgroup/memory/docker/{container_id}"
    cpu_cgroup_path = f"/sys/fs/cgroup/cpu/docker/{container_id}"

    prev_cpu_usage = float('inf')
    prev_system_usage = float('inf')

    client = docker.from_env()
    container = client.containers.get(container_id)

    container_cpu = 0
    system_cpu = 0

    stats_stream = container.stats(stream=True)

    # Get memory limit - this won't change over course of docker container lifetime
    with open(os.path.join(memory_cgroup_path, 'memory.limit_in_bytes'), 'r') as memory_limit_file:
        memory_limit = int(memory_limit_file.read()) / (1024 * 1024)  # Convert to MB

    while True:
        try:
            for stats in stats_stream:
                stats = json.loads(stats.decode('utf-8'))
                # Calculate memory usage: usage - total_inactive_file, found per
                # cadvisor commit: https://github.com/google/cadvisor/commit/307d1b1cb320fef66fab02db749f07a459245451
                # containerd commit: https://github.com/containerd/cri/commit/6b8846cdf8b8c98c1d965313d66bc8489166059a
                with open(os.path.join(memory_cgroup_path, 'memory.usage_in_bytes'), 'r') as memory_usage_file:
                    memory_usage = int(memory_usage_file.read())
                
                with open(os.path.join(memory_cgroup_path, 'memory.stat'), 'r') as memory_stat_file:
                    for line in memory_stat_file:
                        if line.startswith('total_inactive_file '):
                            total_inactive_file = int(line.split(' ')[1])

                memory_util = 0
                if memory_usage > total_inactive_file:
                    memory_util = (memory_usage - total_inactive_file) / (1024 * 1024) # Convert to MB

                # Calculate system usage: logic is obtained from docker stats source code
                # https://github.com/rancher/docker/blob/3f1b16e236ad4626e02f3da4643023454d7dbb3f/daemon/stats_collector_unix.go#L145
                with open('/proc/stat', 'r') as stat_file:
                    for line in stat_file:
                        if line.startswith('cpu '): 
                            cpu_stats = line.split()[1:8]  # Get the next 7 fields (user, nice, system, idle, iowait, irq, softirq)
                            cpu_stats = [int(stat) for stat in cpu_stats] 
                            curr_system_usage = (sum(cpu_stats)) / 100 # convert from Jiffies (USER_HZ) hundredths of a seconds to seconds

                container_cpu = stats['cpu_stats']['cpu_usage']['total_usage']
                system_cpu = stats['cpu_stats']['system_cpu_usage']
                num_cpu = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])
                print(container_cpu)
                print(system_cpu)
                print(num_cpu)

                with open(os.path.join(cpu_cgroup_path, 'cpuacct.usage'), 'r') as cpu_file:
                    cpu_usage_ns = int(cpu_file.read()) / 1e9 # convert form ns to s
                
                with open(os.path.join(cpu_cgroup_path, 'cpuacct.usage_percpu'), 'r') as usage_percpu_file:
                    usage_data = usage_percpu_file.read().strip().split()
                    usage_values = [int(value) for value in usage_data]
                    positive_values = [value for value in usage_values if value > 0]
                    num_cores = len(usage_values)

                # Get timestamp of collected data
                timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

                # Calculate CPU utilization percentage: logic obtained from docker stats source code
                # https://github.com/moby/moby/blob/eb131c5383db8cac633919f82abad86c99bffbe5/cli/command/container/stats_helpers.go#L175
                cpu_delta = cpu_usage_ns - prev_cpu_usage
                system_delta = curr_system_usage - prev_system_usage
                cpu_percent = 0.0
                if system_delta > 0.0 and cpu_delta > 0.0:
                    cpu_percent = cpu_delta / system_delta * num_cores * 100.0 
                print(f'My container cpu: {cpu_usage_ns}')
                print(f'My system_cpu: {curr_system_usage}')
                print(f'My num cores: {num_cores}')
                print(cpu_percent)
                print()

                cursor.execute('INSERT INTO function_utilization VALUES (?, ?, ?, ?, ?)',
                                (container_id, timestamp, cpu_percent, memory_util, memory_limit))
                db_conn.commit()

                # print(f"{timestamp} | Container {container_id} | Mem {memory_usage:.2f} MB / {memory_limit:.2f} MB | CPU {cpu_percent:.2f}%")

                prev_cpu_usage = cpu_usage_ns
                prev_system_usage = curr_system_usage
        except FileNotFoundError:
            print(f"Container {container_id} not found files not found. Exiting monitoring thread.")
            break
        time.sleep(UTIL_TIMESTEP)

# Function to monitor memory and CPU utilization for a container using docker API
def monitor_container_docker_api(container_id):
    db_conn = sqlite3.connect('invoker_data.db')
    cursor = db_conn.cursor()
    
    client = docker.from_env()
    container = client.containers.get(container_id)

    container_cpu = 0
    system_cpu = 0

    stats_stream = container.stats(stream=True)
    try:
        for stats in stats_stream:
            stats = json.loads(stats.decode('utf-8'))
            
            # Compute CPU utilization
            last_container_stats = container_cpu
            last_system_cpu = system_cpu

            container_cpu = stats['cpu_stats']['cpu_usage']['total_usage']
            system_cpu = stats['cpu_stats']['system_cpu_usage']
            num_cpu = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])

            cpu_percent = 0 
            if last_container_stats and last_system_cpu:
                cpu_percent = (container_cpu - last_container_stats) / (system_cpu - last_system_cpu)
                cpu_percent = cpu_percent * num_cpu * 100
            
            # Compute memory utilization
            memory_used = stats['memory_stats']['usage'] - stats['memory_stats']['stats']['cache'] + stats['memory_stats']['stats']['active_file']
            memory_limit = stats['memory_stats']['limit']
            memory_util = round(memory_used / memory_limit * 100, 2)
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            cursor.execute('INSERT INTO function_utilization VALUES (?, ?, ?, ?, ?)',
                            (container_id, timestamp, cpu_percent, memory_util, memory_limit))
            db_conn.commit()
    except Exception as e:
        print(f'Container {container_id} stopped. Exiting monitoring thread.')
        return

# Function to periodically check for new container IDs and start monitoring threads
def monitor_new_containers(no_monitor_ids):
    existing_containers = set()

    while True:
        containers = os.listdir("/sys/fs/cgroup/memory/docker")
        new_containers = set()

        for container_id in containers:
            if os.path.exists(f"/sys/fs/cgroup/memory/docker/{container_id}/tasks") and (container_id not in no_monitor_ids):
                new_containers.add(container_id)

        added_containers = new_containers - existing_containers

        for container_id in added_containers:
            # thread = threading.Thread(target=monitor_container_docker_api, args=(container_id,))
            thread = threading.Thread(target=monitor_container_cgroups, args=(container_id,))
            thread.start()

        existing_containers = new_containers
        time.sleep(NEW_CONTAINER_POLLING_TIMESTEP)  # Adjust the polling interval as needed

if __name__=='__main__':

    # Little bit of preprocessing to get the id of the container with the name 'invoker' or 'prewarm' in it
    # We aren't going to monitor those containers - no need
    client = docker.from_env()
    containers = client.containers.list()
    no_monitor_ids = []
    for container in containers:
        if ('invoker' in container.name) or ('prewarm' in container.name):
            no_monitor_ids.append(container.id)

    # Start the thread for monitoring new containers
    new_container_thread = threading.Thread(target=monitor_new_containers, args=(no_monitor_ids,))
    new_container_thread.start()

    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")





