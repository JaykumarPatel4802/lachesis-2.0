from concurrent import futures
import grpc
import subprocess
from sklearn.utils import shuffle
import pandas as pd
import socket
import sqlite3
import json
import math
import time
from datetime import datetime

from generated import lachesis_pb2_grpc, lachesis_pb2

# TODO: Create table for features

feature_dict = {'floatmatmult': shuffle(pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), random_state=0), 
                'imageprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'), random_state=0),
                'videoprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/videoprocess-inputs.csv'), random_state=0),
                'transcribe': shuffle(pd.read_csv('../data/vw-prediction-inputs/audio-inputs.csv'), random_state=0),
                'sentiment': shuffle(pd.read_csv('../data/vw-prediction-inputs/sentiment-inputs.csv'), random_state=0),
                'lrtrain': shuffle(pd.read_csv('../data/vw-prediction-inputs/lrtrain-inputs.csv'), random_state=0), 
                'mobilenet': shuffle(pd.read_csv('../data/vw-prediction-inputs/mobilenet-inputs.csv'), random_state=0)}

MIN_CPU_ALLOWED = 1
MAX_CPU_ALLOWED = 32 # only allow at most 32 cores per invocation, we have 96 cores total per server
MAX_MEM_CLASSES = 40 # 40 classes, multiply class number by 128MB to get memory limits

START_CPU = 16
START_MEM = 4096

READY_CPU_INVOCATIONS = 10 # increased this from 5
READY_MEM_INVOCATIONS = 20 # originally 20
UNDER_PREDICTION_SEVERITY = 35
MAX_CPU_DECREASE = 6
MAX_CPU_INCREASE = 6 # Changed this from 10 -- realized this may be too aggressive (LRTrain hogs up)

SYSTEM = 'lachesis'
SMALL_CPU = 4
MEDIUM_CPU = 12
LARGE_CPU = 20

# STATIC_MEM_REGISTRATIONS = {'floatmatmult': 3584, 'imageprocess': 512, 'videoprocess': 512, 'sentiment': 512, 'lrtrain': 2560, 'mobilenet': 512, 'encrypt': 512, 'linpack': 1792}
PARROTFISH_MEM_REGISTRATIONS = {'linpack': 3840, 'floatmatmult': 3584, 'lrtrain': 5760, 'videoprocess': 1024, 'imageprocess': 512, 'sentiment': 512, 'encrypt': 512, 'mobilenet': 512}
AQUATOPE_MEM_REGISTRATIONS = {'linpack': 1849, 'floatmatmult': 3947, 'lrtrain': 2724, 'videoprocess': 834, 'imageprocess': 617, 'sentiment': 633, 'encrypt': 681, 'mobilenet': 562}
AQUATOPE_CPU_REGISTRATIONS = {'linpack': 25, 'floatmatmult': 23, 'lrtrain': 28, 'videoprocess': 8, 'imageprocess': 4, 'sentiment': 5, 'encrypt': 11, 'mobilenet': 10}

EXP_VERSION = f'full_lachesis_rps_1_azure'

CONTROLLER_DB = './datastore/lachesis-controller.db'

class Lachesis(lachesis_pb2_grpc.LachesisServicer):
    __curr_port = 26542
    __launched_one_daemon = False
    __image_cpu_port = None
    
    def __launch_ow(self, cpu_limit, mem_limit, primary_key, fxn, inputs):
        # Construct invocation command for OpenWhisk
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
        else:
            print(f'ERROR WITH INVOCATIONS')
            print(f'-----------------------')
            print(stdout)
            print()
            print(stderr)

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
        
        # # Lachesis/ParrotFish Registrations
        # for cpu in range(1, MAX_CPU_ALLOWED + 1):
        #     for mem_class in range(1, MAX_MEM_CLASSES + 1):
        # memory = mem_class * 128
        memory = request.memory
        cpu = request.cpu
        fxn_registration_command = 'cd {}; wsk -i action update {}_{}_{} {}.py --cpu {} --memory {} {} {}\n'.format(request.function_path, request.function, cpu, memory, request.function, cpu, memory, function_metadata_string, parameter_string)
        tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_reg_out, fxn_reg_err = tmp.communicate()
        fxn_reg_out = fxn_reg_out.decode()
        fxn_reg_err = fxn_reg_err.decode()
        if 'ok: updated action' not in fxn_reg_out:
            return lachesis_pb2.Reply(status='FAILURE', message='failed to register function {} with cpu {} and memory {}. Output was: {}. Eror was: {}.'.format(request.function, cpu, memory, fxn_reg_out, fxn_reg_err))

        return lachesis_pb2.Reply(status='SUCCESS', message='successfully registered function {} with cpu {} and memory {}'.format(request.function, request.cpu, request.memory))
    
    def Invoke(self, request, context):
        
        lachesis_start = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        db_conn = sqlite3.connect(CONTROLLER_DB)
        cursor = db_conn.cursor()

        cpu_assigned = request.cpu
        mem_assigned = request.memory
        
        # Insert invocation data into database, begin executing, and update activation id in database!
        cursor.execute('INSERT INTO fxn_exec_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (request.function, lachesis_start, 'NA', 'NA', 'NA', -1.0, -1.0, request.slo, json.dumps(list(request.parameters)), cpu_assigned, mem_assigned, cpu_data, memory_mb, -1.0, -1.0, -1.0, -1.0, -1.0, 'NA', 'NA', 'NA', request.exp_version, -1, -1, -1))
        pk = cursor.lastrowid
        activation_id = self.__launch_ow(cpu_assigned, mem_assigned, pk, request.function, request.parameters)
        cursor.execute('UPDATE fxn_exec_data SET activation_id = ? WHERE rowid = ?', (activation_id, pk))
        db_conn.commit()
        db_conn.close()
        # return lachesis_pb2.Reply(message=f'Test invocation submitted')
        return lachesis_pb2.Reply(message=f'Submitted invocation for {request.function} with CPU limit {cpu_assigned}, memory limit {mem_assigned}, activation_id {activation_id}'), pk

    def Delete(self, request, context):
        fxn_deletion_cmd = 'wsk action delete {}_{}'.format(request.function, request.cpu)
        print(fxn_deletion_cmd)
        tmp = subprocess.Popen(fxn_deletion_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_del_out, fxn_del_err = tmp.communicate()
        fxn_del_out = fxn_del_out.decode()
        fxn_del_err = fxn_del_err.decode()
        print(fxn_del_out)
        print(fxn_del_err)
        if 'ok: deleted action' not in fxn_del_out:
            return lachesis_pb2.Reply(status='FAILURE', message='failed to delete function {} with cpu {}'.format(request.function, request.cpu))

        return lachesis_pb2.Reply(status='SUCCESS', message='successfully deleted function {} with cpu {}'.format(request.function, request.cpu))

    def InsertFunctionData(self, request, context):

        lachesis_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        db_conn = sqlite3.connect(CONTROLLER_DB)
        cursor = db_conn.cursor()

        # Get SLO, CPU limit, and inputs for online updates
        cursor.execute('SELECT slo, cpu_limit, mem_limit, inputs FROM fxn_exec_data WHERE rowid = ?', (request.pk,))
        row = cursor.fetchone()
        slo = -1
        cpu_limit = -1
        mem_limit = -1
        inputs = None
        if row:
            slo, cpu_limit, mem_limit, inputs = row[0], row[1], row[2], json.loads(row[3])

        p90_cpu_used = min(cpu_limit, request.p90_cpu / 100)
        p95_cpu_used = min(cpu_limit, request.p95_cpu / 100)
        p99_cpu_used = min(cpu_limit, request.p99_cpu / 100)
        max_cpu_used = min(cpu_limit, request.max_cpu / 100)
        max_mem_used = int(request.max_mem)
        
        function_name_breakdown = request.function.split('_')
        function_name = function_name_breakdown[0]
        scheduled_cores = cpu_limit
        scheduled_mem = mem_limit
        if len(function_name_breakdown) == 3:
            scheduled_cores = int(function_name_breakdown[1])
            scheduled_mem = int(function_name_breakdown[2])

        # Insert invocation data into database
        cursor.execute('''UPDATE fxn_exec_data
                          SET lachesis_end = ?, start_time = ?, end_time = ?, duration = ?, cold_start_latency = ?,
                              p90_cpu = ?, p95_cpu = ?, p99_cpu = ?, max_cpu = ?, max_mem = ?, invoker_ip = ?, 
                              invoker_name = ?, scheduled_cpu = ?, scheduled_mem = ?, energy = ?
                          WHERE activation_id = ?''', (lachesis_end, request.start_time, request.end_time, request.duration, request.cold_start_latency,
                                                p90_cpu_used, p95_cpu_used, p99_cpu_used, max_cpu_used, max_mem_used, request.invoker_ip, request.invoker_name, 
                                                scheduled_cores, scheduled_mem, request.activation_id, request.energy))
        db_conn.commit()
        db_conn.close()

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