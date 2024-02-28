import grpc
import pandas as pd
from sklearn.utils import shuffle
import subprocess
import json
import time
import random
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import requests
import urllib3
import os

from generated import lachesis_pb2_grpc, lachesis_pb2, cypress_pb2_grpc, cypress_pb2

# Used for interacting with OpenWhisk API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
OPENWHISK_API_URL = 'https://127.0.1.1:443'
AUTHORIZATION_KEY = 'MjNiYzQ2YjEtNzFmNi00ZWQ1LThjNTQtODE2YWE0ZjhjNTAyOjEyM3pPM3haQ0xyTU42djJCS0sxZFhZRnBYbFBrY2NPRnFtMTJDZEFzTWdSVTRWck5aOWx5R1ZDR3VNREdJd1A='
RESULTS_PER_PAGE = 100  # Adjust as needed

# feature_dict = {
#         'floatmatmult': shuffle(pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), random_state=0), 
#         'imageprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'), random_state=0),
#         'videoprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/videoprocess-inputs.csv'), random_state=0),
#         # 'transcribe': shuffle(pd.read_csv('../data/vw-prediction-inputs/audio-inputs.csv'), random_state=0),
#         'sentiment': shuffle(pd.read_csv('../data/vw-prediction-inputs/sentiment-inputs.csv'), random_state=0),
#         'lrtrain': shuffle(pd.read_csv('../data/vw-prediction-inputs/lrtrain-inputs.csv'), random_state=0), 
#         'mobilenet': shuffle(pd.read_csv('../data/vw-prediction-inputs/mobilenet-inputs.csv'), random_state=0),
#         'encrypt': shuffle(pd.read_csv('../data/vw-prediction-inputs/encrypt-inputs.csv'), random_state=0),
#         'linpack': shuffle(pd.read_csv('../data/vw-prediction-inputs/linpack-inputs.csv'), random_state=0)
#     }

# functions = ['floatmatmult', 'imageprocess', 'videoprocess', 'sentiment', 'lrtrain', 'mobilenet', 'encrypt', 'linpack']

feature_dict = {
        'floatmatmult': shuffle(pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), random_state=0), 
        'imageprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'), random_state=0),
        'videoprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/videoprocess-inputs.csv'), random_state=0),
        'encrypt': shuffle(pd.read_csv('../data/vw-prediction-inputs/encrypt-inputs.csv'), random_state=0),
        'linpack': shuffle(pd.read_csv('../data/vw-prediction-inputs/linpack-inputs.csv'), random_state=0)
    }

functions = ['floatmatmult', 'imageprocess', 'videoprocess', 'encrypt', 'linpack']


CPU_MAX = 32
FREQUENCY_MAX = 2400000
FREQUENCY_MIN = 1000000
FREQUENCY_INT = 100000
CONST_MEMORY = 5120

SLO_MULTIPLIER = 0.4 # originally 0.4

CONTROLLER_DB = './datastore/lachesis-controller.db'

INVOKER_USERNAME = "cc"
INVOKER_IP = "129.114.108.87"

'''
Plotting Functions
# '''
# def get_activations(limit):
#     headers = {
#         'Authorization': f'Basic {AUTHORIZATION_KEY}',
#         'Content-Type': 'application/json',
#     }

#     activations = []
#     total_fetched = 0

#     while total_fetched < limit:
#         # Calculate the number of activations to fetch in this iteration
#         remaining_to_fetch = limit - total_fetched
#         fetch_count = min(remaining_to_fetch, RESULTS_PER_PAGE)

#         # Calculate the offset for pagination
#         offset = total_fetched

#         # Make a GET request to fetch activations with SSL certificate verification disabled
#         response = requests.get(
#             f'{OPENWHISK_API_URL}/api/v1/namespaces/_/activations',
#             headers=headers,
#             params={'limit': fetch_count, 'skip': offset},
#             verify=False  # Disable SSL certificate verification
#         )

#         if response.status_code == 200:
#             activations.extend(response.json())
#             total_fetched += fetch_count
#         else:
#             print(f'Failed to retrieve activations. Status code: {response.status_code}')
#             break

#     return activations

# def create_activation_df(limit=2000):
#     activations = get_activations(limit)
    
#     if activations:
#         # Initialize lists to store data
#         activation_ids = []
#         cpu_limits = []
#         memory_limits = []
#         wait_times = []
#         init_times = []
#         durations = []
#         names = []
#         start_times = []
#         end_times = []
#         status_codes = []

#         for activation in activations:
#             # Extract data from the activation JSON
#             activation_id = activation['activationId']
#             annotation = next((ann for ann in activation['annotations'] if ann['key'] == 'limits'), None)
#             cpu_limit = annotation['value']['cpu'] if annotation else None
#             memory_limit = annotation['value']['memory'] if annotation else None
#             wait_time = next((ann['value'] for ann in activation['annotations'] if ann['key'] == 'waitTime'), 0)
#             init_time = next((ann['value'] for ann in activation['annotations'] if ann['key'] == 'initTime'), 0)
#             duration = activation['duration']
#             name = activation['name'].split('_')[0]
#             start_time = activation['start']
#             end_time = activation['end']
#             status_code = activation.get('statusCode', None)

#             # Append extracted data to lists
#             activation_ids.append(activation_id)
#             cpu_limits.append(cpu_limit)
#             memory_limits.append(memory_limit)
#             wait_times.append(wait_time)
#             init_times.append(init_time)
#             durations.append(duration)
#             names.append(name)
#             start_times.append(start_time)
#             end_times.append(end_time)
#             status_codes.append(status_code)

#         # Create a DataFrame from the lists
#         data = {
#             'activation_id': activation_ids,
#             'cpu': cpu_limits,
#             'memory': memory_limits,
#             'wait_time': wait_times,
#             'init_time': init_times,
#             'duration': durations,
#             'name': names,
#             'start_time': start_times,
#             'end_time': end_times,
#             'status_code': status_codes,
#         }

#         df = pd.DataFrame(data)
#         return df


'''
Lachesis Experiment Functions 
'''
def register_functions():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for cpu in range(1, CPU_MAX + 1):

            # Image Process
            function_metadata = ['docker:psinha25/main-python']
            parameters = ['endpoint:\"10.52.3.142:9002\"', 'access_key:\"testkey\"', 'secret_key:\"testsecret\"', 'bucket:\"openwhisk\"']
            response = stub.Register(lachesis_pb2.RegisterRequest(function='imageprocess', 
                                                                function_path='~/lachesis/benchmarks/functions/image-processing', 
                                                                function_metadata=function_metadata, 
                                                                parameters=parameters,
                                                                memory=CONST_MEMORY,
                                                                cpu=cpu))
            print(response)

            # Floatmatmult
            response = stub.Register(lachesis_pb2.RegisterRequest(function='floatmatmult',
                                                                function_path='~/lachesis/benchmarks/functions/matmult',
                                                                function_metadata=function_metadata,
                                                                parameters=parameters,
                                                                memory=CONST_MEMORY,
                                                                cpu=cpu))
            print(response)

            # Video Process
            function_metadata = ['docker:psinha25/video-ow']
            response = stub.Register(lachesis_pb2.RegisterRequest(function='videoprocess',
                                                                function_path='~/lachesis/benchmarks/functions/video-processing',
                                                                function_metadata=function_metadata,
                                                                parameters=parameters,
                                                                memory=CONST_MEMORY,
                                                                cpu=cpu))
            print(response)

            # Linpack
            function_metadata = ['docker:psinha25/main-python']
            response = stub.Register(lachesis_pb2.RegisterRequest(function='linpack',
                                                                function_path='~/lachesis/benchmarks/functions/linpack',
                                                                function_metadata=function_metadata,
                                                                parameters=parameters,
                                                                memory=CONST_MEMORY,
                                                                cpu=cpu))
            print(response)

            # Encryption
            function_metadata = ['docker:psinha25/main-python']
            response = stub.Register(lachesis_pb2.RegisterRequest(function='encrypt',
                                                                function_path='~/lachesis/benchmarks/functions/encryption',
                                                                function_metadata=function_metadata,
                                                                parameters=parameters,
                                                                memory=CONST_MEMORY,
                                                                cpu=cpu))
            print(response)

def test_invocations():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        # matmult_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=15000, parameters=['matrix1_4000_0.7.txt', 'matrix2_4000_0.7.txt']))
        # image_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=4000, parameters=[feature_dict['imageprocess'].iloc[4]['file_name']]))
        # video_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=10000, parameters=[feature_dict['videoprocess'].iloc[4]['file_name']]))
        # sentiment_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=10000, parameters=[feature_dict['sentiment'].iloc[4]['file_name']]))
        # linpack_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=10000, parameters=['5000']))
        # lrtrain_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='lrtrain', slo=10000, parameters=[feature_dict['lrtrain'].iloc[2]['file_name']]))
        mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        # encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['10000', '30']))

        # matmult_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=15000, parameters=['matrix1_4000_0.7.txt', 'matrix2_4000_0.7.txt']))
        # image_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=4000, parameters=[feature_dict['imageprocess'].iloc[4]['file_name']]))
        # video_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=10000, parameters=[feature_dict['videoprocess'].iloc[4]['file_name']]))
        # sentiment_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=10000, parameters=[feature_dict['sentiment'].iloc[4]['file_name']]))
        # linpack_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=10000, parameters=['5000']))
        # lrtrain_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='lrtrain', slo=10000, parameters=[feature_dict['lrtrain'].iloc[2]['file_name']]))
        # mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        # encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['50', '10']))

def run_experiment():
    
    function_counters = {func: 0 for func in functions}

    # Calculate total number of requests (RPS = 2 for 10 minutes)
    request_duration = 10 * 60
    total_requests = 2 * request_duration

    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        start_time = time.time()
        for _ in range(total_requests):
            
            # Randomly select a function and get a row from the dataframe
            selected_function = random.choice(functions)
            current_counter = function_counters[selected_function]
            df = feature_dict[selected_function]
            selected_row = df.iloc[current_counter]

            # Increment function row counter
            function_counters[selected_function] = (function_counters[selected_function] + 1) % len(df)

            # Construct parameter list
            parameter_list = []
            if selected_function == 'linpack':
                parameter_list.append(str(selected_row['matrix_size']))
            elif selected_function == 'floatmatmult':
                parameter_list.append(selected_row['file_name'])
                parameter_list.append(selected_row['file_name'])
            elif selected_function == 'encrypt':
                parameter_list.append(str(selected_row['length']))
                parameter_list.append(str(selected_row['iterations']))
            else:
                parameter_list.append(selected_row['file_name'])
            slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
            
            # Make gRPC invocation request
            response = stub.Invoke(lachesis_pb2.InvokeRequest(function=selected_function, slo=slo, parameters=parameter_list, exp_version='old-experiment-rps-2'))
            print(f'Response for function {selected_function}: {response}')

            # Control the request rate to achieve 2 requests per second
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.5 - elapsed_time % 0.5))

def changeInvokerFrequency(frequency):
    command = f'sudo /path/to/script.sh {frequency}'

    ssh_command = f"ssh {INVOKER_USERNAME}@{INVOKER_IP} '{command}'"
    process = subprocess.Popen(ssh_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f'Invoker frequency changed to {frequency}')
        return True
    else:
        print(f'Error changing invoker frequency to {frequency}')
        print(stdout)
        print(stderr)
        return False

def waitForInvocationCompletion(pk, cursor):
    while(True):
        cursor.execute("SELECT end_time FROM fxn_exec_data WHERE row_id = ?", (pk,))
        result = cursor.fetchone()

        # Check if the end_time value is "NA"
        if result and result[0] == "NA":
            # The end_time for row id {pk] is 'NA'
            sleep(1)
        else:
            # The end_time for row id {pk] is not 'NA'
            break

def test_linpack():
    df = feature_dict['linpack']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for i in range(0, len(df)):
            for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                for cpu in range(1, CPU_MAX + 1):
                    print(f"Running linpack with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['matrix_size'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                        response, pk = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function linpack: {response}')
                        # check if function invocation is completed
                        waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed linpack invocations')

        # wsk -i action invoke linpack_13_128 --param input1 8000

def test_floatmatmult():
    df = feature_dict['floatmatmult']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        
        for i in range(0, len(df)):
            for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                for cpu in range(1, CPU_MAX + 1):
                    print(f"Running floatmatmult with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = []
                        parameter_list.append(selected_row['file_name'])
                        parameter_list.append(selected_row['file_name'])
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                        response, pk = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Resposne for function floatmatmult: {response}')
                        # check if function invocation is completed
                        waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed floatmatmul invocations')

def test_image_process():
    df = feature_dict['imageprocess']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for i in range(0, len(df)):
            for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                for cpu in range(1, CPU_MAX + 1):
                    print(f"Running imageprocess with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['file_name'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                        response, pk = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function imageprocess: {response}')
                        # check if function invocation is completed
                        waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed imageprocess invocations')

def test_encrypt():
    df = feature_dict['encrypt']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for i in range(0, len(df)):
            for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                for cpu in range(1, CPU_MAX + 1):
                    print(f"Running encrypt with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['length']), str(selected_row['iterations'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
                        
                        response, pk = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function encrypt: {response}')
                        # check if function invocation is completed
                        waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed encrypt invocations')

def test_video_process():
    df = feature_dict['videoprocess']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for i in range(0, len(df)):
            for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                for cpu in range(1, CPU_MAX + 1):
                    print(f"Running videoprocess with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['file_name'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)    # this might be duration of video, ask Prasoon

                        response, pk = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function imageprocess: {response}')
                        # check if function invocation is completed
                        waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed imageprocess invocations')

if __name__=='__main__':
    # register_functions()
    # test_invocations()
    # launch_slo_invocations()
    # obtain_input_duration(quantile=0.5)
    # run_experiment()
    # test_linpack()
    # test_floatmatmult()
    # test_sentiment()
    # test_image_process()
    # test_encrypt()