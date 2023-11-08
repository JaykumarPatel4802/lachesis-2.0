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

from generated import lachesis_pb2_grpc, lachesis_pb2

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

    # Get primary key for fxn_exec_data table in Lachesis controller
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

    cursor.execute("INSERT INTO function_executions VALUES (?, ?, NULL, ?, ?, NULL, ?, ?, ?)",
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

                # Sometimes we may not capture utilization -- usually because new container was spun up 
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
                    # print(f'Obtained data for {function} with activation_id {activation_id}')
                    with grpc.insecure_channel(LACHESIS_CONTROLLER_IP + ':' + LACHESIS_CONTROLLER_PORT) as channel:
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
                                                                                                activation_id=activation_id
                                                                                                ))
                # Didn't capture any utilization data unfortunately
                else:
                    # print(f'Could not obtain data for {function} with activation_id {activation_id}')
                    # Make stub call to controller node -- store data in SQLITE database
                    with grpc.insecure_channel(LACHESIS_CONTROLLER_IP + ':' + LACHESIS_CONTROLLER_PORT) as channel:
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
                                                                                                activation_id=activation_id
                                                                                                ))
                data_queue.task_done()
            except queue.Empty:
                pass
    finally:
        db_conn.close()

if __name__=='__main__':
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Script to monitor and process log data.')
    parser.add_argument('--controller-ip', dest='controller_ip', default='10.52.3.142', help='Lachesis controller IP')
    parser.add_argument('--controller-port', dest='controller_port', default='50051', help='Lachesis controller port')
    parser.add_argument('--invoker-ip', dest='invoker_ip', default='129.114.108.158', help='Invoker IP')
    parser.add_argument('--invoker-name', dest='invoker_name', default='w1', help='Invoker name')
    args = parser.parse_args()

    LACHESIS_CONTROLLER_IP = args.controller_ip
    LACHESIS_CONTROLLER_PORT = args.controller_port
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
