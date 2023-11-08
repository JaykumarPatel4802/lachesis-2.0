import re
import sqlite3
import threading
import time
import json
import sys

# Function to parse and insert data into the database
def insert_execution_data(log_line, cold_start_tid, cold_start_latency, db_conn, cursor):

    # Function data
    split_line = log_line.split(' ')
    start_time = split_line[0][1:-1]
    function = split_line[7].split('/')[2]
    container_id = split_line[9][12:-1]
    tid = split_line[2][1:-1]
    if (tid != cold_start_tid) and (cold_start_latency > 0):
        print('ERROR: cold start tid does not match current tid of activation run start')

    # Function inputs
    mem_limit = -1
    cpu_limit = -1
    predicted_cpu = -1
    predicted_mem = -1
    image = ''
    m1 = ''
    m2 = ''
    slo = -1
    if len(split_line) > 12:
        inputs = split_line[13][2:-2].split(',')
        for input in inputs:
            if 'mem_limit' in input:
                mem_limit = float(input.split(':')[1])
            elif 'cpu_limit' in input:
                cpu_limit = float(input.split(':')[1])
            elif 'image' in input:
                image = input.split(':')[1][1:-1]
            elif 'm1' in input:
                m1 = input.split(':')[1][1:-1]
            elif 'm2' in input:
                m2 = input.split(':')[1][1:-1]
            elif 'predicted_ml' in input:
                predicted_mem = float(input.split(':')[1])
            elif 'predicted_cpu' in input:
                predicted_cpu = float(input.split(':')[1])
            elif 'slo' in input:
                slo = float(input.split(':')[1])
    
    # print(f'Start Time: {start_time}')
    # print(f'Function: {function}')
    # print(f'Container ID: {container_id}')
    # print(f'TID: {tid}')
    # print(f'Mem Limit: {mem_limit}')
    # print(f'CPU Limit: {cpu_limit}')
    # print(f'Predicted CPU: {predicted_cpu}')
    # print(f'Predicted Mem: {predicted_mem}')
    # print(f'Image: {image}')
    # print(f'M1: {m1}')
    # print(f'M2: {m2}')
    # print(f'SLO: {slo}')
    # print(f'Cold Start Latency: {cold_start_latency}')
    # print()

    input_json = ''
    if 'floatmatmult' in function:
        input_list = [m1, m2]
        input_json = json.dumps(input_list)
    elif 'imageprocess' in function:
        input_list = [image]
        input_json = json.dumps(input_list)

    cursor.execute("INSERT INTO function_executions VALUES (?, ?, NULL, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
                   (tid, start_time, function, container_id, cpu_limit, mem_limit, predicted_cpu, predicted_mem, cold_start_latency, slo, input_json))
    db_conn.commit()

# Function to monitor and process start lines
def monitor_start_lines(db_conn, cursor):
    with open('/home/cc/openwhisk-tmp-dir/wsklogs/invoker0/invoker0_logs.log', 'r') as input_log:
        input_log.seek(0, 2)
        cold_start_tid = ''
        cold_start_latency = 0
        while True:
            log_line = input_log.readline()
            if ('invoker_activationInit_finish' in log_line):
                split_line = log_line.split(' ')
                cold_start_tid = split_line[2][1:-1]
                cold_start_latency = split_line[-1].split(':')[-1][:-2]
            elif ('invoker_activationRun_start' in log_line) and ('/guest/' in log_line) and ('[DockerContainer]' in log_line): 
                insert_execution_data(log_line, cold_start_tid, cold_start_latency, db_conn, cursor)
                cold_start_tid = ''
                cold_start_latency = 0
            elif ('invoker_activationRun_finish' in log_line):
                split_line = log_line.split(' ')
                end_time = split_line[0][1:-1]
                tid = split_line[2][1:-1]
                duration = float(split_line[-1].split(':')[3][:-2])
                cursor.execute("UPDATE function_executions SET end_time = ?, duration = ? WHERE tid = ?",
                               (end_time, duration, tid))
                db_conn.commit()
            elif not log_line:
                time.sleep(0.1)

if __name__=='__main__':
    try:
        db_conn = sqlite3.connect('invoker_data.db')
        cursor = db_conn.cursor()
        monitor_start_lines(db_conn, cursor)
    except KeyboardInterrupt:
        db_conn.close()
        sys.exit()

