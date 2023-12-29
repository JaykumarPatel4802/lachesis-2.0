import grpc
import queue
import threading
import time
import math
import sqlite3
import json
import subprocess
from concurrent import futures

from generated import cypress_pb2_grpc, cypress_pb2

STATIC_MEM_REGISTRATIONS = {'floatmatmult': 3584, 'imageprocess': 512, 'videoprocess': 512, 'sentiment': 512, 'lrtrain': 2560, 'mobilenet': 512, 'encrypt': 512, 'linpack': 1792}
CONTROLLER_DB = './datastore/lachesis-controller.db'
EXP_VERSION = f'cypress_exp_3_slo_20_max_test'

class Cypress(cypress_pb2_grpc.CypressServicer):
    def __init__(self):
        self.function_queues = {}
    
    def __monitor_queue(self, function):
        func_queue = self.function_queues[function]
        item = func_queue.get(block=True)
        time, parameters, slo, batch_size, exp_version = item
        batch = {}
        batch[batch_size] = [(function, parameters, slo, exp_version)]

        while True:
            try:
                updated_queue = self.function_queues[function]
                next_item = updated_queue.get(block=True, timeout=4)
                next_time, next_parameters, next_slo, next_batch_size, next_exp_version = next_item
                if time != -1:
                    if next_time > time + 2:
                        # invocation_queue.put(batch)
                        self.__process_invocation_queue(batch)
                        time, parameters, slo, batch_size, exp_version = next_time, next_parameters, next_slo, next_batch_size, next_exp_version
                        batch = {}
                        batch[batch_size] = [(function, parameters, slo, exp_version)]
                    else:
                        if next_batch_size in batch:
                            batch[next_batch_size].append((function, next_parameters, next_slo, next_exp_version))
                        else:
                            batch[next_batch_size] = [(function, next_parameters, next_slo, next_exp_version)]
                else:
                    time = next_time
                    batch[next_batch_size] = [(function, next_parameters, next_slo, next_exp_version)]
                    
            except queue.Empty:
                if batch:
                    self.__process_invocation_queue(batch)
                    batch = {}
                    time = -1
                pass
                
    def __process_invocation_queue(self, batch):
        print(batch)
        db_conn = sqlite3.connect(CONTROLLER_DB)
        cursor = db_conn.cursor()
        for batch_size in batch:
            for i, request in enumerate(batch[batch_size]):
                num_requests = len(batch[batch_size])
                cores = 0

                if i >= batch_size:
                    cores = batch_size
                else:
                    cores = math.ceil(batch_size/num_requests)
                if cores > 32:
                    cores = 32
                mem = cores * STATIC_MEM_REGISTRATIONS[request[0]]
                if mem > 5120:
                    mem = 5120
                # print(f'{request[1][0]}: {cores} cores and {mem}MB')
                cursor.execute('INSERT INTO fxn_exec_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                (request[0], 'NA', 'NA', 'NA', 'NA', -1.0, -1.0, request[2], json.dumps(list(request[1])), cores, mem, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 'NA', 'NA', 'NA', request[3], -1.0, -1.0))
                pk = cursor.lastrowid
                activation_id = self.__launch_ow(request[0], cores, mem, pk, request[1])
                cursor.execute('UPDATE fxn_exec_data SET activation_id = ? WHERE rowid = ?', (activation_id, pk))
                db_conn.commit()
        # print('-----------------------')
        db_conn.close()

    def __launch_ow(self, fxn, cpu_limit, mem_limit, primary_key, inputs):
        fxn_invocation_command = None

        if len(inputs) == 2:
            fxn_invocation_command = f'wsk -i action invoke {fxn}_{cpu_limit}_{mem_limit} \
                                        --param input1 {inputs[0]} \
                                        --param input2 {inputs[1]} \
                                        --param pk {primary_key}\n'
        elif len(inputs) == 1:
            fxn_invocation_command = f'wsk -i action invoke {fxn}_{cpu_limit}_{mem_limit} \
                                        --param input1 {inputs[0]} \
                                        --param pk {primary_key}\n'
        
        # Invoke function
        invocation_output = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = invocation_output.communicate()
        if invocation_output.returncode == 0:
            print(f'Launched function {fxn} with {cpu_limit} CPUs, {mem_limit} Memory, and {inputs} inputs.')
            activation_id = stdout.decode('utf-8').strip().split(' ')[-1]
            return activation_id
    
    def Register(self, request, context):
        # Setup function metadata part of registration command
        function_metadata_string = ''
        for metadata in request.function_metadata:
            split_metadata = metadata.split(':')
            key = split_metadata[0]
            value = ''
            for i in range(1, len(split_metadata)):
                value += split_metadata[i] + ":"
            value = value[0:-1]
            function_metadata_string += '--{} {} '.format(key, value)

        # Setup function parameters of registration command
        parameter_string = ''
        for parameter in request.parameters:
            split_parameter = parameter.split(':')
            key = split_parameter[0]
            value = ''
            for i in range(1, len(split_parameter)):
                value += split_parameter[i] + ":"
            value = value[0:-1]
            parameter_string += '--param {} {} '.format(key, value)
        
        # Create final registration command, one per cpu core
        for cpu in range(1, MAX_CPU_ALLOWED + 1):
            for mem_class in range(1, MAX_MEM_CLASSES + 1):
                memory = mem_class * 128
                fxn_registration_command = 'cd {}; wsk -i action update {}_{}_{} {}.py --cpu {} --memory {} {} {}\n'.format(request.function_path, request.function, cpu, memory, request.function, cpu, memory, function_metadata_string, parameter_string)
                tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                fxn_reg_out, fxn_reg_err = tmp.communicate()
                fxn_reg_out = fxn_reg_out.decode()
                fxn_reg_err = fxn_reg_err.decode()
                if 'ok: updated action' not in fxn_reg_out:
                    return cypress_pb2.Reply(status='FAILURE', message='failed to register function {} with cpu {} and memory {}. Output was: {}. Eror was: {}.'.format(request.function, cpu, memory, fxn_reg_out, fxn_reg_err))
            
        return cypress_pb2.Reply(status='SUCCESS', message='successfully registered function {} with all {} cpu levels and {} memory levels'.format(request.function, MAX_CPU_ALLOWED, MAX_MEM_CLASSES))

    def Invoke(self, request, context):
        function = request.function

        # Check if the queue exists for the function, create one if not
        if function not in self.function_queues:
            self.function_queues[function] = queue.Queue()
            # Create a thread to monitor the new queue
            thread = threading.Thread(target=self.__monitor_queue, args=(function,))
            thread.daemon = True
            thread.start()
        
        current_time = time.time()
        item = (current_time, request.parameters, request.slo, math.floor(request.batch_size), request.exp_version)
        self.function_queues[function].put(item)

        return cypress_pb2.Reply(status='SUCCESS', message=f'Queued {request.function} with {request.slo} SLO and {request.parameters}')
    
    def InsertFunctionData(self, request, context):
        db_conn = sqlite3.connect(CONTROLLER_DB)
        cursor = db_conn.cursor()

        # Get SLO, CPU limit, and inputs for online updates
        cursor.execute('SELECT cpu_limit FROM fxn_exec_data WHERE rowid = ?', (request.pk,))
        row = cursor.fetchone()
        cpu_limit = -1
        if row:
            cpu_limit = row[0]

        p90_cpu_used = min(cpu_limit, request.p90_cpu / 100)
        p95_cpu_used = min(cpu_limit, request.p95_cpu / 100)
        p99_cpu_used = min(cpu_limit, request.p99_cpu / 100)
        max_cpu_used = min(cpu_limit, request.max_cpu / 100)
        max_mem_used = int(request.max_mem)
        
        # Insert invocation data into database
        cursor.execute('''UPDATE fxn_exec_data
                          SET start_time = ?, end_time = ?, duration = ?, cold_start_latency = ?,
                              p90_cpu = ?, p95_cpu = ?, p99_cpu = ?, max_cpu = ?, max_mem = ?, invoker_ip = ?, 
                              invoker_name = ?
                          WHERE rowid = ?''', (request.start_time, request.end_time, request.duration, request.cold_start_latency,
                                                p90_cpu_used, p95_cpu_used, p99_cpu_used, max_cpu_used, max_mem_used, request.invoker_ip, 
                                                request.invoker_name, request.pk))
        db_conn.commit()
        db_conn.close()

        return cypress_pb2.Reply(status='SUCCESS', message=f'Received daemon metrics for function {request.function} from Invoker {request.invoker_ip}')

def run_server():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cypress_pb2_grpc.add_CypressServicer_to_server(Cypress(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Cypress server is up and running, ready for your request!')
    server.wait_for_termination()

if __name__=='__main__':
    run_server()