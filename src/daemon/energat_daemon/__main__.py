import sys

from common import FLAGS, logger
from tracer import EnergyTracer
import os
import sqlite3
import time
import docker
import traceback


init_containers = set()

energy_tracer_objects = dict()

def getDockerPid(container_id):
    client = docker.from_env()
    container = client.containers.get(container_id)
    return container.attrs['State']['Pid']

def obtainInitContainer():
    container_dir = "/sys/fs/cgroup/memory/docker"
    try:
        with os.scandir(container_dir) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name not in ('.', '..'):
                    init_containers.add(entry.name)
    except FileNotFoundError:
        print(f"Directory '{container_dir}' not found.")

def process_container(container_id, db):

    """
    check if EnergyTracer object exists for container_id
    if it does:
        use that object to obtain metrics
    if not:
        create a new object to obtain metrics
        save the object
    """

    try:
        docker_pid = getDockerPid(container_id)
        energy_tracer = None
    except:
        return

    # if container_id in energy_tracer_objects:
    #     energy_tracer = energy_tracer_objects[container_id]
    #     # energy_tracer.recreate_tracer_process()
    # else:
    #     name = f"target-{docker_pid}"
    #     energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
    #     energy_tracer_objects[container_id] = energy_tracer
    

    # Without New Process Code
    name = f"target-{docker_pid}"
    # print(f"Name is {name}")
    # print(f"Docker container PID: {docker_pid}")
    energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
    # print("Running tracer")
    try:
        energy_tracer.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        # traceback.print_exc()

    # print("DONE")

    # ## With New Process Code
    # print(f"NEW DOCKER: {docker_pid} at: {time.time()}")
    # name = f"target-{docker_pid}"
    # energy_tracer = EnergyTracer(docker_pid, attach=True, project=name)
    # try:
    #     energy_tracer.launch()
    #     energy_tracer.tracer_process.join()
    # except:
    #     energy_tracer.stop()

    # try:
    #     # energy_tracer.run()
    #     print(f"STARTING {docker_pid}")
    #     energy_tracer.launch()
    #     energy_tracer.tracer_process.join()
    # except KeyboardInterrupt:
    #     energy_tracer.stop()
    #     # print("Error running tracer")


    pass

def main():

    if FLAGS.basepower:
        EnergyTracer(1).estimate_baseline_power(save=True)
        return 0

    db_file = "../invoker_data.db"
    try:
        # Open the database
        print(f"Opening {db_file}")
        db = sqlite3.connect(db_file)
        
        # Set WAL mode
        print("Setting WAL mode")
        db.execute("PRAGMA journal_mode=WAL;")

        # To delete the table and reset all data
        # db.execute("DROP TABLE function_energy_utilization_advanced")
        
        # Create the table if it doesn't exist
        print("Creating table if it doesn't exist")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS function_energy_utilization_advanced (
            container_id TEXT,
            timestamp TEXT,
            socket REAL,
            duration_sec REAL,
            num_proc REAL,
            num_threads REAL,
            pkg_credit_frac REAL,
            dram_credit_frac REAL,
            total_pkg_joules REAL,
            total_dram_joules REAL,
            base_pkg_joules REAL,
            base_dram_joules REAL,
            ascribed_pkg_joules REAL,
            ascribed_dram_joules REAL,
            tracer_pkg_joules REAL,
            tracer_dram_joules REAL,
            pkg_percent REAL,
            dram_percent REAL
        );
        """
        db.execute(create_table_sql)
        
        db.commit()  # Commit the changes

        EnergyTracer.sqlite_db = db
        EnergyTracer.csv_file = "energat_traces.csv"
        
    except sqlite3.Error as e:
        print("SQLite error:", e)


    obtainInitContainer()

    daemon_period = 0.1

    while (1):
        container_dir = "/sys/fs/cgroup/memory/docker"
        try:
            with os.scandir(container_dir) as entries:
                for entry in entries:
                    if entry.is_dir() and entry.name not in ('.', '..'):
                        container_name = entry.name
                        if container_name not in init_containers:
                            # print(f"container name: {container_name} time: {time.time()}")
                            process_container(container_name, db)
                        # process_container(container_name, db)
            # time.sleep(daemon_period)  # Sleep for 500 milliseconds
        except FileNotFoundError:
            print(f"Directory '{container_dir}' not found.")

    if db:
        db.close()

        
    print("Exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
