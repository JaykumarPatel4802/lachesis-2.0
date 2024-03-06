import sqlite3
import threading
import time
import json
import sys
import queue
import numpy as np
import grpc
import argparse
import glob
from datetime import datetime
import subprocess

from generated import lachesis_pb2_grpc, lachesis_pb2, cypress_pb2_grpc, cypress_pb2

# Function to parse and insert data into the database
def insert_execution_data(log_line, cold_start_tid, cold_start_latency, db_conn, cursor):
    # Function data
    split_line = log_line.split(' ')
    start_time = split_line[0][1:-1]
    function = split_line[7].split('/')[2]
    container_id = split_line[9][12:-1]
    tid = split_line[2][1:-1]

    final_cold_start_latency = int(cold_start_latency)
    if (tid != cold_start_tid) and (int(cold_start_latency) > 0):
        # print('NOTE: cold start tid does not match current tid of activation run start')
        final_cold_start_latency = 0

    # Get activation id 
    activation_id = None
    if len(split_line) > 12:
        inputs = split_line[17][2:-2].split(',')
        activation_id = inputs[0].split(':')[1]

    # Get primary key for fxn_exec_data table in Central controller
    pk = -1
    if len(split_line) > 12:
        inputs = split_line[13][2:-2].split(',')
        for input in inputs:
            if 'pk' in input:
                pk = int(input.split(':')[1])
            # if 'mem_limit' in input:
            #     mem_limit = float(input.split(':')[1])
            # elif 'cpu_limit' in input:
            #     cpu_limit = float(input.split(':')[1])
            # elif 'image' in input:
            #     image = input.split(':')[1][1:-1]
            # elif 'm1' in input:
            #     m1 = input.split(':')[1][1:-1]
            # elif 'm2' in input:
            #     m2 = input.split(':')[1][1:-1]
            # elif 'predicted_ml' in input:
            #     predicted_mem = float(input.split(':')[1])
            # elif 'predicted_cpu' in input:
            #     predicted_cpu = float(input.split(':')[1])
            # elif 'slo' in input:
            #     slo = float(input.split(':')[1])
    
    # input_json = ''
    # if 'floatmatmult' in function:
    #     input_list = [m1, m2]
    #     input_json = json.dumps(input_list)
    # elif 'imageprocess' in function:
    #     input_list = [image]
    #     input_json = json.dumps(input_list)

    
    # cursor.execute("INSERT INTO function_executions VALUES (?, ?, NULL, ?, ?, NULL, ?, ?, ?)", (tid, start_time, function, container_id, final_cold_start_latency, pk, activation_id))

    cursor.execute("INSERT OR REPLACE INTO function_executions VALUES (?, ?, NULL, ?, ?, NULL, ?, ?, ?)",
               (tid, start_time, function, container_id, final_cold_start_latency, pk, activation_id))


    db_conn.commit()

# Function to monitor and process start lines
def monitor_start_lines(db_conn, cursor, data_queue):

    # Get exact path of log file - log_file should be array of size 1
    log_path = '/home/cc/openwhisk-tmp-dir/wsklogs/'
    file_pattern = 'invoker*/invoker*_logs.log'
    log_file = glob.glob(log_path + file_pattern)

    with open(log_file[0], 'r') as input_log:
        input_log.seek(0, 2)
        cs_tid = ''
        cs_latency = 0
        while True:
            log_line = input_log.readline()
            if ('invoker_activationInit_finish' in log_line):
                split_line = log_line.split(' ')
                cs_tid = split_line[2][1:-1]
                cs_latency = split_line[-1].split(':')[-1][:-2]
            elif ('invoker_activationRun_start' in log_line) and ('/guest/' in log_line) and ('[DockerContainer]' in log_line): 
                insert_execution_data(log_line, cs_tid, cs_latency, db_conn, cursor)
                cs_tid = ''
                cs_latency = 0
            elif ('invoker_activationRun_finish' in log_line):
                split_line = log_line.split(' ')
                end_time = split_line[0][1:-1]
                tid = split_line[2][1:-1]
                duration = float(split_line[-1].split(':')[3][:-2])
                cursor.execute("UPDATE function_executions SET end_time = ?, duration = ? WHERE tid = ?",
                               (end_time, duration, tid))
                db_conn.commit()
                # Retrieve start_time and container_id from the database
                cursor.execute("SELECT start_time, container_id, function, cold_start_latency, pk, activation_id FROM function_executions WHERE tid = ?", (tid,))
                result = cursor.fetchone()
                if result:
                    start_time, container_id, function, cold_start_latency, pk, activation_id = result
                    data_queue.put((start_time, container_id, function, duration, cold_start_latency, pk, activation_id, end_time))
            elif not log_line:
                time.sleep(0.1)

def filter_outliers(time_power):
    power = time_power[: 1]
    mean_power = np.mean(power)
    std_power = np.std(power)
    upper_bound = mean_power + (3 * std_power)
    lower_bound = mean_power - (3 * std_power)
    filtered_time_power = time_power[(time_power[:, 1] <= upper_bound) & (time_power[:, 1] >= lower_bound)]
    
    return filtered_time_power

def timestamp_to_utc(timestamp_str):
    timestamp_obj = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    return (timestamp_obj - datetime(1970, 1, 1)).total_seconds()


def parse_energy(rows):
    timestamp_list = np.array([row[0] for row in rows])
    timestamp_list = np.vectorize(timestamp_to_utc)(timestamp_list) # convert timestamp to UTC for energy calculation
    socket_list = np.array([row[1] for row in rows])
    durations_sec_list = np.array([row[2] for row in rows])
    ascribed_pkg_joules_list = np.array([row[3] for row in rows])
    ascribed_dram_joules_list = np.array([row[4] for row in rows])

    # everytime stamp has 2 entries, for each socket, so combine and condense data here
    timestamps = timestamp_list[::2]
    timestamps = timestamps.astype(np.float64)
    # if len(timestamps) % 2 != 0:
    #     # If it's odd, remove the last element
    #     timestamps = timestamps[:-1]
    durations = durations_sec_list[::2]
    # if len(durations) % 2 != 0:
    #     # If it's odd, remove the last element
    #     durations = durations[:-1]
    # if len(ascribed_pkg_joules_list) % 2 != 0:
    #     # If it's odd, remove the last element
    #     ascribed_pkg_joules_list = ascribed_pkg_joules_list[:-1]
    pkg_pairs = ascribed_pkg_joules_list.reshape(-1, 2)
    ascribed_pkg_joules_list = np.sum(pkg_pairs, axis = 1)
    # if len(ascribed_dram_joules_list) % 2 != 0:
    #     # If it's odd, remove the last element
    #     ascribed_dram_joules_list = ascribed_dram_joules_list[:-1]
    dram_pairs = ascribed_dram_joules_list.reshape(-1, 2)
    ascribed_dram_joules_list = np.sum(dram_pairs, axis = 1)
    energies = ascribed_pkg_joules_list + ascribed_dram_joules_list

    # now timestamps, durations, and total_energies has everything I need to calcualte final energy
    centered_timestamps = timestamps - (durations / 2)
    powers = energies / durations

    time_power = np.array(list(zip(centered_timestamps, powers)))
    filtered_time_power = filter_outliers(time_power)
    final_timestamps = filtered_time_power[:, 0]
    final_powers = filtered_time_power[:, 1]

    if len(final_timestamps) == 0:
        return 0
    elif len(final_timestamps) == 1:
        return energies[0]

    final_energy = np.trapz(final_powers, x=final_timestamps)

    return final_energy

def kill_container(container_id):
    # execute `docker kill` command
    command = f'docker kill {container_id}'
    process = subprocess.Popen(command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f'Killed container {container_id}')
    else:
        print(f'Failed to kill container {container_id}')
        print(stdout)
        print(stderr)
    return

# Function to monitor the queue and process data
def monitor_queue(data_queue):
    try:
        db_conn = sqlite3.connect('invoker_data.db', isolation_level=None)
        cursor = db_conn.cursor()
        # Enable WAL mode
        cursor.execute("PRAGMA journal_mode=WAL;")
        while True:
            try:
                # Get data from the queue
                start_time, container_id, function, duration, cold_start_latency, pk, activation_id, end_time = data_queue.get(timeout=1)
                # Query the database for data between start_time and end_time with the same container_id
                # cursor.execute("SELECT cpu_util, mem_util, mem_limit FROM function_utilization WHERE timestamp BETWEEN ? AND ? AND container_id = ?",
                #             (start_time, end_time, container_id))
                cursor.execute("SELECT cpu_usage_ns, num_cores, curr_system_usage, mem_util, mem_limit FROM function_utilization_advanced WHERE timestamp BETWEEN ? AND ? AND container_id = ?",
                            (start_time, end_time, container_id))
                rows = cursor.fetchall()

                cursor.execute("SELECT timestamp, socket, duration_sec, ascribed_pkg_joules, ascribed_dram_joules from function_energy_utilization_advanced WHERE timestamp BETWEEN ? AND ? and container_id = ?",
                            (start_time, end_time, container_id))
                energy_rows = cursor.fetchall()

                if len(energy_rows) % 2 != 0:
                    energy_rows = energy_rows[:-1]  # Remove the last entry

                # Sometimes we may not capture utilization -- usually because new container was spun up 

                energy = -1

                if energy_rows and (len(energy_rows) >= 2):
                    energy = parse_energy(energy_rows)

                kill_container(container_id)

                # for an invocation and completed before thread for that container began collecting util
                if rows:

                    # Extract data from the rows
                    cpu_usages = np.array([row[0] for row in rows])
                    num_cores = np.array([row[1] for row in rows])
                    curr_system_usages = np.array([row[2] for row in rows])
                    mem_utils = [row[3] for row in rows]
                    memory_limit = rows[0][4]

                    # Compute CPU utilization in same form as docker stats
                    cpu_utilization_list = np.zeros(len(cpu_usages), dtype=float)
                    prev_cpu_usages = np.roll(cpu_usages, 1)
                    prev_cpu_usages[0] = 0
                    prev_system_usages = np.roll(curr_system_usages, 1)
                    prev_system_usages[0] = 0

                    cpu_delta = cpu_usages - prev_cpu_usages
                    system_delta = curr_system_usages - prev_system_usages
                    mask = (system_delta > 0.0) & (cpu_delta > 0.0)
                    cpu_utilization_list[mask] = (cpu_delta[mask] / system_delta[mask]) * num_cores[mask] * 100.0

            
                    # cpu_utilization_list = [row[0] for row in rows]
                    # memory_usage_list = [row[1] for row in rows]
                    # memory_limit = rows[0][2]

                    # Calculate various percentiles for CPU utilization
                    min_cpu_utilization = min(cpu_utilization_list)
                    p10_cpu_utilization = np.percentile(cpu_utilization_list, 10)
                    p25_cpu_utilization = np.percentile(cpu_utilization_list, 25)
                    p50_cpu_utilization = np.percentile(cpu_utilization_list, 50)
                    p75_cpu_utilization = np.percentile(cpu_utilization_list, 75)
                    p90_cpu_utilization = np.percentile(cpu_utilization_list, 90)
                    p95_cpu_utilization = np.percentile(cpu_utilization_list, 95)
                    p99_cpu_utilization = np.percentile(cpu_utilization_list, 99)
                    max_cpu_utilization = max(cpu_utilization_list)

                    # Calculate the maximum memory usage
                    max_memory_usage = max(mem_utils)
                    
                    print(f"Energy for container: {container_id} is {energy}")

                    # print(f'Obtained data for {function} with activation_id {activation_id}')
                    with grpc.insecure_channel(CENTRAL_CONTROLLER_IP + ':' + CENTRAL_CONTROLLER_PORT) as channel:
                        stub = lachesis_pb2_grpc.LachesisStub(channel)
                        response = stub.InsertFunctionData(lachesis_pb2.InsertFunctionDataRequest(pk = int(pk),
                                                                                                function=function,
                                                                                                start_time=start_time,
                                                                                                end_time=end_time,
                                                                                                duration=float(duration),
                                                                                                cold_start_latency=float(cold_start_latency),
                                                                                                p90_cpu=float(p90_cpu_utilization),
                                                                                                p95_cpu=float(p95_cpu_utilization),
                                                                                                p99_cpu=float(p99_cpu_utilization),
                                                                                                max_cpu=float(max_cpu_utilization),
                                                                                                max_mem=float(max_memory_usage),
                                                                                                invoker_ip=INVOKER_IP,
                                                                                                invoker_name=INVOKER_NAME,
                                                                                                activation_id=activation_id,
                                                                                                energy=float(energy)
                                                                                                ))
                # Didn't capture any utilization data unfortunately
                else:
                    # print(f'Could not obtain data for {function} with activation_id {activation_id}')
                    # Make stub call to controller node -- store data in SQLITE database
                    with grpc.insecure_channel(CENTRAL_CONTROLLER_IP + ':' + CENTRAL_CONTROLLER_PORT) as channel:
                        stub = lachesis_pb2_grpc.LachesisStub(channel)
                        response = stub.InsertFunctionData(lachesis_pb2.InsertFunctionDataRequest(pk = int(pk),
                                                                                                function=function,
                                                                                                start_time=start_time,
                                                                                                end_time=end_time,
                                                                                                duration=float(duration),
                                                                                                cold_start_latency=float(cold_start_latency),
                                                                                                p90_cpu=-2.0,
                                                                                                p95_cpu=-2.0,
                                                                                                p99_cpu=-2.0,
                                                                                                max_cpu=-2.0,
                                                                                                max_mem=-2.0,
                                                                                                invoker_ip=INVOKER_IP,
                                                                                                invoker_name=INVOKER_NAME,
                                                                                                activation_id=activation_id,
                                                                                                energy=float(energy)
                                                                                                ))
                data_queue.task_done()
            except queue.Empty:
                pass
    finally:
        db_conn.close()

if __name__=='__main__':
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Daemon to monitor and process log data.')
    parser.add_argument('--controller-ip', dest='controller_ip', default='129.114.108.12', help='central controller IP')
    parser.add_argument('--controller-port', dest='controller_port', default='50051', help='central controller port')
    parser.add_argument('--invoker-ip', dest='invoker_ip', default='129.114.109.158', help='Invoker IP')
    parser.add_argument('--invoker-name', dest='invoker_name', default='w1', help='Invoker name')
    args = parser.parse_args()

    CENTRAL_CONTROLLER_IP = args.controller_ip
    CENTRAL_CONTROLLER_PORT = args.controller_port
    INVOKER_IP = args.invoker_ip
    INVOKER_NAME = args.invoker_name

    try:
        db_conn_main = sqlite3.connect('invoker_data.db', isolation_level=None)
        cursor_main = db_conn_main.cursor()

        # Enable WAL mode
        cursor_main.execute("PRAGMA journal_mode=WAL;")

        # Create a queue for communication between threads
        data_queue = queue.Queue()

        # Create a separate thread to monitor the queue
        queue_thread = threading.Thread(target=monitor_queue, args=(data_queue,))
        # queue_thread.daemon = True
        queue_thread.start()

        # Start the main thread to monitor start lines
        monitor_start_lines(db_conn_main, cursor_main, data_queue)
    except KeyboardInterrupt:
        db_conn_main.close()
        sys.exit()
