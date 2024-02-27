from concurrent import futures
import grpc
import subprocess
import pandas as pd
from sklearn.utils import shuffle
import socket
import sqlite3
import json
import math
import time
from datetime import datetime

from generated import lachesis_pb2_grpc, lachesis_pb2

MAX_CPU_INCREASE = 6 # Changed this from 10 -- realized this may be too aggressive (LRTrain hogs up)
MAX_CPU_ALLOWED = 32 # only allow at most 32 cores per invocation, we have 96 cores total per server
MAX_CPU_DECREASE = 6

START_CPU = 16
START_MEM = 4096

READY_CPU_INVOCATIONS = 10 # increased this from 5
READY_MEM_INVOCATIONS = 20 # originally 20
UNDER_PREDICTION_SEVERITY = 35
MAX_CPU_DECREASE = 6
MAX_CPU_INCREASE = 6 # Changed this from 10 -- realized this may be too aggressive (LRTrain hogs up)

CONTROLLER_DB = './datastore/lachesis-controller.db'

class Lachesis(lachesis_pb2_grpc.LachesisServicer):
    def InsertFunctionData(self, request, context):
        print("GREEN: Received function data from Invoker")
        # Create a connection to the database
        lachesis_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        db_conn = sqlite3.connect(CONTROLLER_DB)
        cursor = db_conn.cursor()

        # Insert the function data into the database
        cursor.execute('SELECT slo, cpu_limit, mem_limit, inputs FROM fxn_exec_data WHERE rowid = ?', (request.activation_id,))
        row = cursor.fetchone() 
        slo = -1
        cpu_limit = -1
        mem_limit = -1
        inputs = None
        if row:
            slo, cpu_limit, mem_limit, inputs = row[0], row[1], row[2], json.loads(row[3])
        
        p90_cpu_used = max(cpu_limit, request.p90_cpu / 100) # convert 90th percentile from percentage to number of cores
        p95_cpu_used = max(cpu_limit, request.p95_cpu / 100) # convert 95th percentile from percentage to number of cores
        p99_cpu_used = max(cpu_limit, request.p99_cpu / 100) # convert 99th percentile from percentage to number of cores
        max_cpu_used = max(cpu_limit, request.max_cpu / 100) # convert max cpu from percentage to number of cores
        max_mem_used = int(request.max_mem) # convert max mem from percentage to MB 

        function_name_breakdown = request.function.split('_')
        function_name = function_name_breakdown[0]
        scheduled_cores = cpu_limit
        scheduled_mem = mem_limit
        if len(function_name_breakdown) == 3:
            scheduled_cores = int(function_name_breakdown[1])
            scheduled_mem = int(function_name_breakdown[2])
        # import pdb; pdb.set_trace()
        # Insert invocation data into database
        print(f"GREEN CONTROLLER: updating database")
        cursor.execute('''UPDATE fxn_exec_data
                          SET lachesis_end = ?, start_time = ?, end_time = ?, duration = ?, cold_start_latency = ?,
                              p90_cpu = ?, p95_cpu = ?, p99_cpu = ?, max_cpu = ?, max_mem = ?, invoker_ip = ?, 
                              invoker_name = ?, scheduled_cpu = ?, scheduled_mem = ?, energy = ?
                          WHERE activation_id = ?''', (lachesis_end, request.start_time, request.end_time, request.duration, request.cold_start_latency,
                                                p90_cpu_used, p95_cpu_used, p99_cpu_used, max_cpu_used, max_mem_used, request.invoker_ip, request.invoker_name, 
                                                scheduled_cores, scheduled_mem, request.activation_id, request.energy))
        print(f"GREEN: Row count is: {cursor.rowcount}")
        if cursor.rowcount == 0:
            # If not, insert a new row
            cursor.execute('''INSERT INTO fxn_exec_data VALUES (?, ?, ?, ?, ?,  ?,  ?, ?, ?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (function_name, 'NA', lachesis_end, request.start_time, request.end_time, request.duration, request.cold_start_latency, -1.0, 'NA', -1.0, -1.0, -1.0, -1.0, p90_cpu_used, p95_cpu_used, p99_cpu_used, max_cpu_used, max_mem_used, request.invoker_ip, request.invoker_name, request.activation_id, 'NA', scheduled_cores, scheduled_mem, request.energy))
        db_conn.commit()
        db_conn.close()

        print("GREEN: Returning")

        return lachesis_pb2.Reply(status='SUCCESS', message=f'Received daemon metrics for function {request.function} from Invoker {request.invoker_ip}')
    
    
def run_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=40))
    lachesis_pb2_grpc.add_LachesisServicer_to_server(Lachesis(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    # print('Lachesis server is up and running, ready for your request!')
    server.wait_for_termination()


if __name__=='__main__':    
    run_server()
