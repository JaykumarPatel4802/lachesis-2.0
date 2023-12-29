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

# EXP_VERSION = 'exp_7_slo_0.2_invoker_mem_75gb_mem_load_balancer'
# EXP_VERSION = 'exp_16_slo_0.4_quantile_50_invoker_mem_75gb_full_load_balancer_rps_2'
# EXP_VERSION = 'exp_18_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_rps_2' # Don't use these results or exp_17, bug in scheduler
# EXP_VERSION = 'exp_20_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_custom_mem_rps_2'
# EXP_VERSION = 'exp_21_slo_0.4_quantile_50_invoker_mem_125gb_cold_start_scheduler_custom_mem_rps_2_admission_control'
# EXP_VERSION = 'exp_22_slo_0.4_quantile_50_invoker_mem_125gb_cpu_120_cold_start_scheduler_custom_mem_rps_2_admission_control'
# EXP_VERSION = 'exp_23_slo_0.4_quantile_50_invoker_mem_125gb_cpu_140_cold_start_scheduler_custom_mem_rps_2_admission_control'
# EXP_VERSION = 'exp_24_slo_0.4_quantile_50_invoker_mem_125gb_cpu_120_cold_start_scheduler_custom_mem_rps_2_admission_control'
# EXP_VERSION = 'exp_25_slo_0.4_quantile_50_invoker_mem_125gb_cpu_120_cold_start_scheduler_custom_mem_rps_2_admission_control'
# EXP_VERSION = 'exp_26_slo_0.4_quantile_50_invoker_mem_125gb_cpu_120_cold_start_scheduler_custom_mem_rps_2'
# EXP_VERSION = 'exp_27_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_custom_mem_rps_2'
# EXP_VERSION = f'exp_28_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_custom_mem_rps_2_{SYSTEM}'
# EXP_VERSION = f'exp_29_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_custom_mem_rps_2_{SYSTEM}'
# EXP_VERSION = f'exp_30_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_custom_mem_rps_2_{SYSTEM}'
# EXP_VERSION = f'exp_31_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_background_custom_mem_rps_2' # Bug in the releasing of slots when scheduling
# EXP_VERSION = f'exp_32_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_background_custom_mem_rps_2' # Bug in the releasing of slots when scheduling still
# EXP_VERSION = f'exp_33_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_background_custom_mem_rps_2'
# EXP_VERSION = f'exp_36_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_lachesis_scheduler_custom_mem_rps_2_{SYSTEM}'
# EXP_VERSION = f'exp_37_slo_0.4_quantile_50_invoker_mem_125gb_cpu_90_cold_start_scheduler_background_custom_mem_rps_2'
# EXP_VERSION = f'exp_40_slo_0.4_quantile_50_invoker_mem_125gb_default_scheduler_custom_mem_rps_2_{SYSTEM}'
# EXP_VERSION = f'exp_43_slo_0.4_quantile_50_invoker_mem_125gb_lachesis_scheduler_lachesis_ra_custom_mem_rps_2' # severity 50, 15 invocations, +1 class
# EXP_VERSION = f'exp_44_slo_0.4_quantile_50_invoker_mem_125gb_lachesis_scheduler_lachesis_ra_custom_mem_rps_2'# severity 50, 15 invocations, +0 class linpack
# EXP_VERSION = f'exp_45_slo_0.4_quantile_50_invoker_mem_125gb_lachesis_scheduler_lachesis_ra_custom_mem_rps_2'# severity 50, 15 invocations, +0 class floatmatmult
# EXP_VERSION = f'exp_46_linpack_severity_35_15_invocation_memory_prediction_test'
# EXP_VERSION = f'exp_47_linpack_severity_40_15_invocation_memory_prediction_test'
# EXP_VERSION = f'exp_48_slo_0.4_quantile_50_rps_2_full_lachesis'
EXP_VERSION = f'full_lachesis_rps_1_azure'

CONTROLLER_DB = './datastore/lachesis-controller.db'

class Lachesis(lachesis_pb2_grpc.LachesisServicer):
    __curr_port = 26542
    __launched_one_daemon = False
    __image_cpu_port = None
    
    def __compute_costs(self, best_class, max_classes):
        '''
            This function assumes that max_cpu is in
            the form of # of cores, not Linux percentage
        '''
        costs = [0] * (max_classes + 1)
        for i in range(1, max_classes + 1):
            if i < best_class:
                costs[i] = UNDER_PREDICTION_SEVERITY + (best_class - i)
            else:
                costs[i] = i - best_class + 1
        return costs

    def __format_costs(self, costs, max_classes):
        sample = ''
        for i in range(1, max_classes + 1):
            sample += f'{i}:{costs[i]} '
        return sample

    def __format_features(self, features):
        sample = '| '
        for feature in features:
            sample += '{0} '.format(feature)
        return sample

    def __format_one_hot_cpu_features(self, function, features):
        '''
        Formatting is as following, with numbers including feature and SLO
        | <encrypt, 3> <matmult, 5> <imageprocess, 7> <linpack, 2> <lrtrain, 4> <mobilenet, 7> <sentiment, 3> <videoprocess, 10>
        '''
        indices = {'encrypt': 0, 'floatmatmult': 3, 'imageprocess': 8, 'linpack': 15, 'lrtrain': 17, 'mobilenet': 21, 'sentiment': 28, 'videoprocess': 31}
        
        start_index = indices[function]
        final_feature_list = [0] * 41
        for i, feature in enumerate(features):
            final_feature_list[start_index + i] = feature
        
        sample = '| '
        for feature in final_feature_list:
            sample += '{0} '.format(feature)
        return sample
    
    def __format_one_hot_mem_features(self, function, features):
        '''
        Formatting is as following, with numbers including only features, not SLO
        | <encrypt, 2> <matmult, 4> <imageprocess, 6> <linpack, 1> <lrtrain, 3> <mobilenet, 6> <sentiment, 2> <videoprocess, 9>
        '''
        indices = {'encrypt': 0, 'floatmatmult': 2, 'imageprocess': 6, 'linpack': 12, 'lrtrain': 13, 'mobilenet': 16, 'sentiment': 22, 'videoprocess': 24}
        
        start_index = indices[function]
        final_feature_list = [0] * 33
        for i, feature in enumerate(features):
            final_feature_list[start_index + i] = feature
        
        sample = '| '
        for feature in final_feature_list:
            sample += '{0} '.format(feature)
        return sample

    def __vw_format_creator(self, costs, max_classes, features):
        sample_cost = self.__format_costs(costs, max_classes)
        sample_features = self.__format_features(features)
        sample = sample_cost + sample_features
        return sample
    
    def __vw_format_one_hot_creator(self, costs, max_classes, function, resource_type, features):
        sample_cost = self.__format_costs(costs, max_classes)
        sample_features = None
        if resource_type == 'mem':
            sample_features = self.__format_one_hot_mem_features(function, features[0:-1])
        elif resource_type == 'cpu':
            sample_features = self.__format_one_hot_cpu_features(function, features)
        sample = sample_cost + sample_features
        return sample

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
        
        # Aquatope Registrations
        # fxn_registration_command = f'cd {request.function_path}; wsk -i action update {request.function}_{AQUATOPE_CPU_REGISTRATIONS[request.function]}_{AQUATOPE_MEM_REGISTRATIONS[request.function]} {request.function}.py --cpu {AQUATOPE_CPU_REGISTRATIONS[request.function]} --memory {AQUATOPE_MEM_REGISTRATIONS[request.function]} {function_metadata_string} {parameter_string}\n'
        # tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # fxn_reg_out, fxn_reg_err = tmp.communicate()
        # fxn_reg_out = fxn_reg_out.decode()
        # fxn_reg_err = fxn_reg_err.decode()
        # if 'ok: updated action' not in fxn_reg_out:
        #     return lachesis_pb2.Reply(status='FAILURE', message='failed to register function {} with cpu {} and memory {}. Output was: {}. Eror was: {}.'.format(request.function, cpu, memory, fxn_reg_out, fxn_reg_err))

        # Lachesis/ParrotFish Registrations
        for cpu in range(1, MAX_CPU_ALLOWED + 1):
            for mem_class in range(1, MAX_MEM_CLASSES + 1):
                memory = mem_class * 128
                fxn_registration_command = 'cd {}; wsk -i action update {}_{}_{} {}.py --cpu {} --memory {} {} {}\n'.format(request.function_path, request.function, cpu, memory, request.function, cpu, memory, function_metadata_string, parameter_string)
                tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                fxn_reg_out, fxn_reg_err = tmp.communicate()
                fxn_reg_out = fxn_reg_out.decode()
                fxn_reg_err = fxn_reg_err.decode()
                if 'ok: updated action' not in fxn_reg_out:
                    return lachesis_pb2.Reply(status='FAILURE', message='failed to register function {} with cpu {} and memory {}. Output was: {}. Eror was: {}.'.format(request.function, cpu, memory, fxn_reg_out, fxn_reg_err))

        return lachesis_pb2.Reply(status='SUCCESS', message='successfully registered function {} with all {} cpu levels and {} memory levels'.format(request.function, MAX_CPU_ALLOWED, MAX_MEM_CLASSES))
    
    def Invoke(self, request, context):
        
        lachesis_start = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        db_conn = sqlite3.connect(CONTROLLER_DB)
        cursor = db_conn.cursor()

        # Check if Lachesis has seen this function before
        cursor.execute("SELECT cpu_port, mem_port, no_invocations FROM pred_models WHERE function = ?", (request.function,))
        row = cursor.fetchone()
        cpu_port = None
        mem_port = None
        no_invocations = -1
        if row:
            cpu_port, mem_port, no_invocations = row
        else:
            # if (request.function == 'imageprocess') or (request.function == 'mobilenet'):
            #     if (self.__image_cpu_port != None):
            #         cpu_port = self.__image_cpu_port
            #         mem_port = self.__image_cpu_port + 1
            #     else:
            #         cpu_port = self.__curr_port
            #         mem_port = cpu_port + 1
            #         self.__curr_port += 2
            #         self.__image_cpu_port = cpu_port
            #         vw_cpu_command = f'vw --csoaa {MAX_CPU_ALLOWED} --daemon --quiet --port {cpu_port}'
            #         vw_mem_command = f'vw --csoaa {MAX_MEM_CLASSES} --daemon --quiet --port {mem_port}'
            #         tmp = subprocess.Popen(vw_cpu_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #         tmp = subprocess.Popen(vw_mem_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #         time.sleep(0.5)
            #     cursor.execute("INSERT INTO pred_models VALUES (?, ?, ?, ?, ?)",
            #                 (request.function, 'localhost', cpu_port, mem_port, 0))
            #     db_conn.commit()
            # else:
            
            # Set up csoaa models for this function
            cpu_port = self.__curr_port
            mem_port = cpu_port + 1
            self.__curr_port += 2
            # if not self.__launched_one_daemon:
            #     self.__launched_one_daemon = True
            vw_cpu_command = f'vw --csoaa {MAX_CPU_ALLOWED} --daemon --quiet --port {cpu_port}'
            vw_mem_command = f'vw --csoaa {MAX_MEM_CLASSES} --daemon --quiet --port {mem_port}'
            tmp = subprocess.Popen(vw_cpu_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tmp = subprocess.Popen(vw_mem_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.5)
            cursor.execute("INSERT INTO pred_models VALUES (?, ?, ?, ?, ?)",
                            (request.function, 'localhost', cpu_port, mem_port, 0))
            db_conn.commit()

        # Establish connection to this functions CPU and memory model
        cpu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mem_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cpu_server_address = ('localhost', int(cpu_port))
        mem_server_address = ('localhost', int(mem_port))
        # print(f'{request.function}')
        # print(f'CPU Server Address: {cpu_server_address}')
        # print(f'Mem Server Address: {mem_server_address}')
        cpu_socket.connect(cpu_server_address)
        mem_socket.connect(mem_server_address)

        # Extract features of parameter(s)
        features = []
        if request.function in feature_dict:
            df = feature_dict[request.function]
            features = df[df['file_name'] == request.parameters[0]].drop(columns=['file_name', 'duration']).values.tolist()[0]
        else:
            for parameter in request.parameters:
                features.append(float(parameter))
        mem_features = features.copy()
        cpu_features = features
        cpu_features.append(request.slo)

        # Get CPU Assignment
        cpu_assigned = START_CPU
        cpu_vw_sample = self.__format_features(cpu_features)
        # cpu_vw_sample = self.__format_one_hot_cpu_features(request.function, cpu_features)
        cpu_vw_sample += '\n'
        cpu_socket.sendall(cpu_vw_sample.encode())
        cpu_data = cpu_socket.recv(1024).decode().strip()

        # Get Memory Assignment
        mem_assigned = START_MEM
        mem_vw_sample = self.__format_features(mem_features)
        # mem_vw_sample = self.__format_one_hot_mem_features(request.function, mem_features)
        mem_vw_sample += '\n'
        mem_socket.sendall(mem_vw_sample.encode())
        mem_data = mem_socket.recv(1024).decode().strip()
        memory_mb = (int(mem_data)) * 128 # each class corresponds to an additional 128 MB of memory

        if (request.system == 'lachesis'):
            if (no_invocations >= READY_CPU_INVOCATIONS):
                cpu_assigned = cpu_data           
            if (no_invocations >= READY_MEM_INVOCATIONS):
                mem_assigned = memory_mb
        elif (request.system == 'small'):
            cpu_assigned = SMALL_CPU
            # mem_assigned = STATIC_MEM_REGISTRATIONS[request.function]
            mem_assigned = int(SMALL_CPU / 2 * 512)
        elif (request.system == 'medium'):
            cpu_assigned = MEDIUM_CPU
            # mem_assigned = STATIC_MEM_REGISTRATIONS[request.function]
            mem_assigned = int(MEDIUM_CPU / 2 * 512)
        elif (request.system == 'large'):
            cpu_assigned = LARGE_CPU
            # mem_assigned = STATIC_MEM_REGISTRATIONS[request.function]
            mem_assigned = int(LARGE_CPU / 2 * 512)
        elif (request.system == 'parrotfish'):
            mem_assigned = PARROTFISH_MEM_REGISTRATIONS[request.function]
            cpu_assigned = int(mem_assigned / 512 * 2)
        elif (request.system == 'aquatope'):
            mem_assigned = AQUATOPE_MEM_REGISTRATIONS[request.function]
            cpu_assigned = AQUATOPE_CPU_REGISTRATIONS[request.function]
        
        # Insert invocation data into database, begin executing, and update activation id in database!
        cursor.execute('INSERT INTO fxn_exec_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (request.function, lachesis_start, 'NA', 'NA', 'NA', -1.0, -1.0, request.slo, json.dumps(list(request.parameters)), cpu_assigned, mem_assigned, cpu_data, memory_mb, -1.0, -1.0, -1.0, -1.0, -1.0, 'NA', 'NA', 'NA', request.exp_version, -1, -1))
        pk = cursor.lastrowid
        activation_id = self.__launch_ow(cpu_assigned, mem_assigned, pk, request.function, request.parameters)
        cursor.execute('UPDATE fxn_exec_data SET activation_id = ? WHERE rowid = ?', (activation_id, pk))
        db_conn.commit()
        db_conn.close()
        # return lachesis_pb2.Reply(message=f'Test invocation submitted')
        return lachesis_pb2.Reply(message=f'Submitted invocation for {request.function} with CPU limit {cpu_assigned}, memory limit {mem_assigned}, activation_id {activation_id}')

    def Delete(self, request, context):
        for cpu in range(2, MAX_CPU_ALLOWED+1):
            fxn_deletion_cmd = 'wsk action delete {}_{}'.format(request.function, cpu)
            print(fxn_deletion_cmd)
            tmp = subprocess.Popen(fxn_deletion_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            fxn_del_out, fxn_del_err = tmp.communicate()
            fxn_del_out = fxn_del_out.decode()
            fxn_del_err = fxn_del_err.decode()
            print(fxn_del_out)
            print(fxn_del_err)
            if 'ok: deleted action' not in fxn_del_out:
                return lachesis_pb2.Reply(status='FAILURE', message='failed to delete function {} with cpu {}'.format(request.function, cpu))
        return lachesis_pb2.Reply(status='SUCCESS', message='successfully deleted function {} with all {} cpu levels'.format(request.function, MAX_CPU_ALLOWED))

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

        # Only retrain if daemon captured utilization data for the invocation
        if request.p99_cpu > 0:
            # Compute proper cpu limit (max_cpu) for this invocation
            max_cpu = math.ceil(max_cpu_used)
            slack = slo - (request.duration + request.cold_start_latency)
            # print()
            # print(f'Inputs: {inputs}')
            # print(f'Observed Latency: {request.duration + request.cold_start_latency}')
            # print(f'SLO: {slo}')
            # print(f'Slack: {slack}')
            # print(f'Max cores used: {max_cpu_used}')
            # print(f'CPU Limit was: {cpu_limit}')
            
            # TODO: rethink this logic -- should be be increasing CPU even if less than limit was used
            if slack < 0:
                '''
                    Increase the cpu limit that should've been used 
                    by 1 for every 0.5 seconds over the SLO if utilization was high
                '''
                if (cpu_limit - max_cpu) <= 1:
                    # New cost function algo
                    cpu_increase_factor = (request.duration + request.cold_start_latency + abs(slack)) / (request.duration + request.cold_start_latency)
                    max_cpu = min(math.ceil(max_cpu * cpu_increase_factor), max_cpu + MAX_CPU_INCREASE)
                    max_cpu = min(MAX_CPU_ALLOWED, max_cpu)

                    
                    
                    # print('Increasing CPU limit')
                    # cpu_increase = min(math.ceil(abs(slack) / 500), MAX_CPU_INCREASE)
                    # max_cpu += cpu_increase
                    # max_cpu = min(MAX_CPU_ALLOWED, max_cpu)
                # print('Oh shit, did not meet SLO!')
            # TODO: rethink this logic -- should be decreasing CPU regardless of whether max_cpu was used or less was used
            # elif max_cpu == cpu_limit:
            else:
                '''
                    Decrease the CPU limit by 1 for every 1.5 
                    seconds we are under the SLO, regardless of the 
                    amount of cores used
                '''
                # New cost function algo
                cpu_decrease_factor = (request.duration + request.cold_start_latency - slack) / (request.duration + request.cold_start_latency)
                max_cpu = max(math.ceil(max_cpu * cpu_decrease_factor), max_cpu - MAX_CPU_DECREASE)
                max_cpu = max(MIN_CPU_ALLOWED, max_cpu)

                # print('Decreasing CPU limit')
                # cpu_decrease = min(MAX_CPU_DECREASE, math.floor(abs(slack) / 1500))
                # max_cpu = max_cpu - cpu_decrease
                # max_cpu = max(MIN_CPU_ALLOWED, max_cpu)
            # print('Max CPU cores assigned: {}'.format(max_cpu))

            # Compute max memory class - we want to give between 5-10% more mem than the max mem used
            # max_mem_class = math.ceil((mem_limit - ((mem_limit - max_mem_used) / 2)) / 128)
            # print(f'Mem limit: {mem_limit}, Max mem used: {max_mem_used}, Max mem class: {max_mem_class}')
            if max_mem_used > scheduled_mem:
                max_mem_used = scheduled_mem - 128
            max_mem_class = math.ceil(max_mem_used / 128)

            # Extract features
            features = []
            if function_name in feature_dict:
                df = feature_dict[function_name]
                features = df[df['file_name'] == inputs[0]].drop(columns=['file_name', 'duration']).values.tolist()[0]
            elif function_name == 'encrypt':
                features.append(float(inputs[0]))
                features.append(float(inputs[1]))
            elif function_name == 'linpack':
                features.append(float(inputs[0]))
            mem_features = features.copy()
            cpu_features = features
            cpu_features.append(slo)

            # Retrain VW CSOAA models
            cpu_costs = self.__compute_costs(max_cpu, MAX_CPU_ALLOWED)
            vw_cpu_sample = self.__vw_format_creator(cpu_costs, MAX_CPU_ALLOWED, cpu_features)
            # vw_cpu_sample = self.__vw_format_one_hot_creator(cpu_costs, MAX_CPU_ALLOWED, request.function.split('_')[0], 'cpu', cpu_features)
            vw_cpu_sample += '\n'

            mem_costs = self.__compute_costs(max_mem_class, MAX_MEM_CLASSES)
            vw_mem_sample = self.__vw_format_creator(mem_costs, MAX_MEM_CLASSES, mem_features)
            # vw_mem_sample = self.__vw_format_one_hot_creator(mem_costs, MAX_MEM_CLASSES, request.function.split('_')[0], 'mem', cpu_features)
            vw_mem_sample += '\n'

            cursor.execute("SELECT cpu_port, mem_port FROM pred_models WHERE function = ?", (function_name,))
            row = cursor.fetchone()
            cpu_port = None
            mem_port = None
            if row:
                cpu_port = row[0]
                mem_port = row[1]

            cpu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            mem_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cpu_server_address = ('localhost', int(cpu_port))
            mem_server_address = ('localhost', int(mem_port))
            
            cpu_socket.connect(cpu_server_address)
            cpu_socket.sendall(vw_cpu_sample.encode())
            cpu_data = cpu_socket.recv(1024).decode().strip()

            mem_socket.connect(mem_server_address)
            mem_socket.sendall(vw_mem_sample.encode())
            mem_data = mem_socket.recv(1024).decode().strip()

            # Update number of invocations made to this function after training function
            cursor.execute(''' 
                UPDATE pred_models
                SET no_invocations = no_invocations + 1
                WHERE function = ?
            ''', (function_name,))
            db_conn.commit()

        # Insert invocation data into database
        cursor.execute('''UPDATE fxn_exec_data
                          SET lachesis_end = ?, start_time = ?, end_time = ?, duration = ?, cold_start_latency = ?,
                              p90_cpu = ?, p95_cpu = ?, p99_cpu = ?, max_cpu = ?, max_mem = ?, invoker_ip = ?, 
                              invoker_name = ?, activation_id = ?, scheduled_cpu = ?, scheduled_mem = ?
                          WHERE rowid = ?''', (lachesis_end, request.start_time, request.end_time, request.duration, request.cold_start_latency,
                                                p90_cpu_used, p95_cpu_used, p99_cpu_used, max_cpu_used, max_mem_used, request.invoker_ip, request.invoker_name, 
                                                request.activation_id, scheduled_cores, scheduled_mem, request.pk))
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