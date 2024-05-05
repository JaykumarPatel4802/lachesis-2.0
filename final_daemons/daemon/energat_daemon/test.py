# this is to test that we are writing to the sqlite database properly and the aggregator is correctly able to retrieve that information.

import sys

from common import FLAGS, logger
from tracer import EnergyTracer
import os
import sqlite3
import time
import docker
import traceback
import numpy as np

def getDockerPid(container_id):
    client = docker.from_env()
    container = client.containers.get(container_id)
    return container.attrs['State']['Pid']

def filter_outliers(time_power):
    power = time_power[: 1]
    mean_power = np.mean(power)
    std_power = np.std(power)
    upper_bound = mean_power + (3 * std_power)
    lower_bound = mean_power - (3 * std_power)
    print(f"Time Power: {time_power}")
    filtered_time_power = time_power[(time_power[:, 1] <= upper_bound) & (time_power[:, 1] >= lower_bound)]
    
    return filtered_time_power

def parse_energy(rows):
    timestamp_list = np.array([row[0] for row in rows])
    socket_list = np.array([row[1] for row in rows])
    durations_sec_list = np.array([row[2] for row in rows])
    ascribed_pkg_joules_list = np.array([row[3] for row in rows])
    ascribed_dram_joules_list = np.array([row[4] for row in rows])

    # everytime stamp has 2 entries, for each socket, so combine and condense data here
    timestamps = timestamp_list[::2]
    timestamps = timestamps.astype(np.float64)
    durations = durations_sec_list[::2]
    pkg_pairs = ascribed_pkg_joules_list.reshape(-1, 2)
    ascribed_pkg_joules_list = np.sum(pkg_pairs, axis = 1)
    dram_pairs = ascribed_dram_joules_list.reshape(-1, 2)
    ascribed_dram_joules_list = np.sum(dram_pairs, axis = 1)
    energies = ascribed_pkg_joules_list + ascribed_dram_joules_list

    print(f"Energies: {energies}")

    # now timestamps, durations, and total_energies has everything I need to calcualte final energy
    centered_timestamps = timestamps - (durations / 2)
    powers = energies / durations

    time_power = np.array(list(zip(centered_timestamps, powers)))
    print(f"OG time_power: {time_power}")
    filtered_time_power = filter_outliers(time_power)
    print(f"Filtered time_power: {filtered_time_power}")
    final_timestamps = filtered_time_power[:, 0]
    final_powers = filtered_time_power[:, 1]

    print(f"final_timestamps: {final_timestamps}")
    print(f"final_powers: {final_powers}")

    if len(final_timestamps) == 0:
        return 0
    elif len(final_timestamps) == 1:
        return energies[0]

    final_energy = np.trapz(final_powers, x=final_timestamps)
    print(f"Final Energy: {final_energy}")

    return final_energy


def handle_aggregator_logic(cursor, docker_id):
    cursor.execute("SELECT timestamp, socket, duration_sec, ascribed_pkg_joules, ascribed_dram_joules from function_energy_utilization_advanced_test WHERE timestamp < ? and container_id = ?", (time.time(), docker_id))
    energy_rows = cursor.fetchall()

    print("Printing data rows")
    for row in energy_rows:
        print(row)

    print()

    print("Printing total energy")
    energy = parse_energy(energy_rows)
    print(energy)

def main():

    db_file = "invoker_data_test.db"
    db = None
    cursor = None
    try:
        # Open the database
        print(f"Opening {db_file}")
        db = sqlite3.connect(db_file)
        cursor = db.cursor()
        
        # Set WAL mode
        print("Setting WAL mode")
        db.execute("PRAGMA journal_mode=WAL;")
        
        db.execute("DROP TABLE function_energy_utilization_advanced_test")

        # Create the table if it doesn't exist
        print("Creating table if it doesn't exist")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS function_energy_utilization_advanced_test (
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
            pkg_percent REAL,
            dram_percent REAL
        );
        """
        db.execute(create_table_sql)
        
        db.commit()  # Commit the changes

        EnergyTracer.sqlite_db = db
        
    except sqlite3.Error as e:
        print("SQLite error:", e)


    docker_id = "163fe7a0ae109a6346ed7d5460c7c6d70d2556a903a3475d8cf60c8ef3adb753"
    docker_pid = getDockerPid(docker_id)

    for i in range(2):

        name = f"target-{docker_pid}"
        energy_tracer = EnergyTracer(docker_pid, attach=True, project=name, container_id = docker_id)
        try:
            energy_tracer.run()
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
    
    handle_aggregator_logic(cursor, docker_id)

    return 0

if __name__ == "__main__":
    sys.exit(main())