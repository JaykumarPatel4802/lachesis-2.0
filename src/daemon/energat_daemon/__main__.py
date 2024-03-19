import sys

from common import FLAGS, logger
from tracer import EnergyTracer
import os
import sqlite3
import time
import docker
import traceback

energy_tracer_objects = dict()

def getDockerPid(container_id):
    client = docker.from_env()
    container = client.containers.get(container_id)
    return container.attrs['State']['Pid']

def obtainInitContainer():
    init_containers= set()
    container_dir = "/sys/fs/cgroup/memory/docker"
    try:
        with os.scandir(container_dir) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name not in ('.', '..'):
                    init_containers.add(entry.name)
    except FileNotFoundError:
        print(f"Directory '{container_dir}' not found.")
    return init_containers

def process_container(container_id, db):

    """
    check if EnergyTracer object exists for container_id
    if it does:
        use that object to obtain metrics
    if not:
        create a new object to obtain metrics
        save the object
    """

    # try:
    #     docker_pid = getDockerPid(container_id)
    #     energy_tracer = None
    # except:
    #     return

    # if container_id in energy_tracer_objects:
    #     energy_tracer = energy_tracer_objects[container_id]
    #     # energy_tracer.recreate_tracer_process()
    # else:
    #     name = f"target-{docker_pid}"
    #     energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
    #     energy_tracer_objects[container_id] = energy_tracer
    
    # Without New Process Code
    # name = f"target-{docker_pid}"
    # print(f"Name is {name}")
    # print(f"Docker container PID: {docker_pid}")


    try:
        docker_pid = getDockerPid(container_id)
        name = f"target-{docker_pid}"
        energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
        try:
            energy_tracer.run()
            print("Run GOOD")
            return
        except Exception as e1:
            # energy_tracer.stop_tracer_daemon_thread = True
            # if energy_tracer.tracer_daemon_thread is not None:
                # if energy_tracer.tracer_daemon_thread.is_alive():
                    # energy_tracer.tracer_daemon_thread.join()
            print("Run BAD")
            energy_tracer.stop_flag.set()
            if energy_tracer.tracer_daemon_thread is not None:
                energy_tracer.tracer_daemon_thread.join()
            print(f"An error occurred: {e1}")
            # traceback.print_exc()
            return
    except Exception as e2:
        print(f"Error creating tracer object: {e2}")
        return

    # try:
    #     docker_pid = getDockerPid(container_id)
    #     name = f"target-{docker_pid}"
    #     energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
    #     try:
    #         energy_tracer.run()
    #         return
    #     except Exception as e1:
    #         print(f"An error occurred: {e1}")
    #         traceback.print_exc()
    #         return
    # except Exception as e2:
    #     print(f"Error creating tracer object: {e2}")
    #     return


    return
    # try:
    #     docker_pid = getDockerPid(container_id)
    #     name = f"target-{docker_pid}"
    #     with EnergyTracer(docker_pid, attach=True, project=name, container_id=container_id) as energy_tracer:
    #         try:
    #             energy_tracer.run()
    #         except Exception as e1:
    #             print(f"An error occurred: {e1}")
    # except Exception as e2:
    #     print(f"Error creating tracer object: {e2}")

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


    # # try:
    # #     docker_pid = getDockerPid(container_id)
    # #     energy_tracer = None
    # # except:
    # #     return

    # # if container_id in energy_tracer_objects:
    # #     energy_tracer = energy_tracer_objects[container_id]
    # #     # energy_tracer.recreate_tracer_process()
    # # else:
    # #     name = f"target-{docker_pid}"
    # #     energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
    # #     energy_tracer_objects[container_id] = energy_tracer
    
    # # Without New Process Code
    # # name = f"target-{docker_pid}"
    # # print(f"Name is {name}")
    # # print(f"Docker container PID: {docker_pid}")

    # try:
    #     docker_pid = getDockerPid(container_id)
    #     name = f"target-{docker_pid}"
    #     energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = container_id)
    #     try:
    #         energy_tracer.run()
    #         return
    #     except Exception as e1:
    #         energy_tracer.stop_tracer_daemon_thread = True
    #         if energy_tracer.tracer_daemon_thread is not None:
    #             if energy_tracer.tracer_daemon_thread.is_alive():
    #                 energy_tracer.tracer_daemon_thread.join()
    #         print(f"An error occurred: {e1}")
    #         # traceback.print_exc()
    #         return
    # except Exception as e2:
    #     print(f"Error creating tracer object: {e2}")
    #     return

    # return

    # # try:
    # #     docker_pid = getDockerPid(container_id)
    # #     name = f"target-{docker_pid}"
    # #     with EnergyTracer(docker_pid, attach=True, project=name, container_id=container_id) as energy_tracer:
    # #         try:
    # #             energy_tracer.run()
    # #         except Exception as e1:
    # #             print(f"An error occurred: {e1}")
    # # except Exception as e2:
    # #     print(f"Error creating tracer object: {e2}")

    # # print("DONE")

    # # ## With New Process Code
    # # print(f"NEW DOCKER: {docker_pid} at: {time.time()}")
    # # name = f"target-{docker_pid}"
    # # energy_tracer = EnergyTracer(docker_pid, attach=True, project=name)
    # # try:
    # #     energy_tracer.launch()
    # #     energy_tracer.tracer_process.join()
    # # except:
    # #     energy_tracer.stop()

    # # try:
    # #     # energy_tracer.run()
    # #     print(f"STARTING {docker_pid}")
    # #     energy_tracer.launch()
    # #     energy_tracer.tracer_process.join()
    # # except KeyboardInterrupt:
    # #     energy_tracer.stop()
    # #     # print("Error running tracer")


    # # pass


def get_container_name(container_id):
    # Create a Docker client
    client = docker.from_env()

    try:
        # Get the container object using its ID
        container = client.containers.get(container_id)
        
        # Get the container name
        container_name = container.name
    except Exception as e:
        return ""

    return container_name


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
        db.execute("DROP TABLE function_energy_utilization_advanced")
        
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

    init_containers = set()
    init_containers = obtainInitContainer()

    daemon_period = 0.1

    count = 0

    while (1):
        count += 1
        if count == 100000:
            print(f"Count is 10000")
            count = 0
        container_dir = "/sys/fs/cgroup/memory/docker"
        try:
            with os.scandir(container_dir) as entries:
                for entry in entries:
                    if entry.is_dir() and entry.name not in ('.', '..'):
                        container_name = entry.name
                        if container_name not in init_containers:
                            name = get_container_name(container_name)
                            if name != "" and "nodejs" not in name:
                                print(f"ID: {container_name}")
                                process_container(container_name, db)
                                count = 0
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
