import grpc
import pandas as pd
from sklearn.utils import shuffle
import subprocess
import json
import time
import random
# import seaborn as sns
import sqlite3
# import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import requests
import urllib3
import os
from enum import Enum
import traceback

from generated import lachesis_pb2_grpc, lachesis_pb2, cypress_pb2_grpc, cypress_pb2
from energy_model.generated import bayesian_regressor_pb2, bayesian_regressor_pb2_grpc


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

# feature_dict = {
#         'floatmatmult': shuffle(pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), random_state=0), 
#         'imageprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'), random_state=0),
#         'videoprocess': shuffle(pd.read_csv('../data/vw-prediction-inputs/videoprocess-inputs.csv'), random_state=0),
#         'encrypt': shuffle(pd.read_csv('../data/vw-prediction-inputs/encrypt-inputs.csv'), random_state=0),
#         'linpack': shuffle(pd.read_csv('../data/vw-prediction-inputs/linpack-inputs.csv'), random_state=0)
#     }


feature_dict = {
        'floatmatmult': pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), 
        'imageprocess': pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'),
        'videoprocess': pd.read_csv('../data/vw-prediction-inputs/videoprocess-inputs.csv'),
        'encrypt': pd.read_csv('../data/vw-prediction-inputs/encrypt-inputs.csv'),
        'linpack': pd.read_csv('../data/vw-prediction-inputs/linpack-inputs.csv'),
        'lrtrain': pd.read_csv('../data/vw-prediction-inputs/lrtrain-inputs.csv'),
        'sentiment': pd.read_csv('../data/vw-prediction-inputs/sentiment-inputs.csv')
    }

functions = ['floatmatmult', 'imageprocess', 'videoprocess', 'encrypt', 'linpack']

# use 512 increments
mem_test_dict_floatmatmult = {
    "matrix1_1000_0.7.txt": [256, 512, 1024, 2048, 4096],
    "matrix1_2000_0.3.txt": [512, 1024, 2048, 4096],
    "matrix1_4000_0.7.txt": [1024, 2048, 4096],
}


CPU_MAX = 32
FREQUENCY_MAX = 2400000
FREQUENCY_MIN = 1000000
FREQUENCY_INT = 100000
CONST_MEMORY = 5120
FREQUENCIES = [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2400000]
CPUS = [1, 2, 3, 5, 7, 10, 13, 16, 19, 22, 25, 28, 31]

SLO_MULTIPLIER = 0.4 # originally 0.4

CONTROLLER_DB = './datastore/lachesis-controller.db'

INVOKER_USERNAME = "cc"
INVOKER_IP = "129.114.108.87"

class FunctionType(Enum):
    ALL = 0
    FLOATMATMULT = 1
    IMAGEPROCESS = 2
    VIDEOPROCESS = 3
    LINPACK = 4
    ENCRYPT = 5
    LRTRAIN = 6
    SENTIMENT = 7

def register_floatmatmult_mem_test_functions():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        parameters = ['endpoint:\"10.52.2.197:9002\"', 'access_key:\"testkey\"', 'secret_key:\"testsecret\"', 'bucket:\"openwhisk\"']

        for memory in [1024]:
            for frequency in [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2400000]:
                # Floatmatmult
                function_metadata = ['docker:psinha25/main-python']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='floatmatmult',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/matmult',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=memory,
                                                                    cpu=4,
                                                                    frequency=frequency))
                print(response)

def test_floatmatmult_scheduler_test():

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        
        floatmatmul_input = "matrix1_4000_0.7.txt"
        # mem_freq = [(1024, 1000000), (1024, 1200000), (1024, 1400000), (1024, 1600000), (1024, 1800000), (1024, 2000000), (1024, 2200000), (1024, 2400000)]
        # mem_freq = [(1024, 1200000)]
        mem_freq = [(1024, 1000000)]
        cpu = 2
        for (mem, frequency) in mem_freq:
            print(f"Running floatmatmult with frequency: {frequency}, cpu: {cpu}, memory: {mem}")
            parameter_list = []
            parameter_list.append(floatmatmul_input)
            parameter_list.append(floatmatmul_input)
            slo = float(76132.2 * (1 + SLO_MULTIPLIER))

            response = stub.Invoke(bayesian_regressor_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list, cpu=cpu, memory=mem, frequency=frequency))
            print(f'Resposne for function floatmatmult: {response}')
            # check if function invocation is completed
            pk = response.primary_key
            time.sleep(0.5)
    db_conn.close()
    print('Completed floatmatmul invocations')


def test_floatmatmult_mem_test():

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    success = rerunEnergyTracer()
    assert(success)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        
        for floatmatmul_input in mem_test_dict_floatmatmult:
            for frequency in [2400000]:
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                for mem in mem_test_dict_floatmatmult[floatmatmul_input]:
                    for cpu in [1]:
                        print(f"Running floatmatmult with frequency: {frequency}, cpu: {cpu}, memory: {mem}")
                        for _ in range(3):
                            parameter_list = []
                            parameter_list.append(floatmatmul_input)
                            parameter_list.append(floatmatmul_input)
                            slo = float(76132.2 * (1 + SLO_MULTIPLIER))

                            response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list, cpu=cpu, memory=mem, frequency=frequency))
                            print(f'Resposne for function floatmatmult: {response}')
                            # check if function invocation is completed
                            pk = response.primary_key
                            waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed floatmatmul invocations')


def run_final_experiment():

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    measurement_study_df = dict()
    floatmatmult_file_path = '../data_processing/filtered_data/filtered_floatmatmult.csv'
    imageprocess_file_path = '../data_processing/filtered_data/filtered_imageprocess.csv'
    videoprocess_file_path = '../data_processing/filtered_data/filtered_videoprocess.csv'
    encrypt_file_path = '../data_processing/filtered_data/filtered_encrypt.csv'
    linpack_file_path = '../data_processing/filtered_data/filtered_linpack.csv'

    measurement_study_df["floatmatmult"] = pd.read_csv(floatmatmult_file_path)
    measurement_study_df["imageprocess"] = pd.read_csv(imageprocess_file_path)
    measurement_study_df["videoprocess"] = pd.read_csv(videoprocess_file_path)
    measurement_study_df["encrypt"] = pd.read_csv(encrypt_file_path)
    measurement_study_df["linpack"] = pd.read_csv(linpack_file_path)

    inputs = {
        "floatmatmult": ([["matrix1_1000_0.3.txt", "matrix1_1000_0.3.txt"], ["matrix1_1000_0.7.txt", "matrix1_1000_0.7.txt"], ["matrix1_2000_0.3.txt", "matrix1_2000_0.3.txt"], ["matrix1_2000_0.7.txt", "matrix1_2000_0.7.txt"], ["matrix1_4000_0.3.txt", "matrix1_4000_0.3.txt"], ["matrix1_4000_0.7.txt", "matrix1_4000_0.7.txt"], ["matrix1_6000_0.3.txt", "matrix1_6000_0.3.txt"], ["matrix1_6000_0.7.txt", "matrix1_6000_0.7.txt"], ["matrix1_8000_0.3.txt", "matrix1_8000_0.3.txt"], ["matrix1_8000_0.7.txt", "matrix1_8000_0.7.txt"]], 76132.2),

        "imageprocess": ([["13M-st_basils_cathedral_2_516323.jpg"], ["952K-iris_and_daisies_2_194935.jpg"], ["1.3M-daisy_514381.jpg"], ["4.5M-more_toys_514506.jpg"], ["2.5M-red_velvet_514717.jpg"], ["1.8M-deserted_bench_197838.jpg"], ["2.9M-water_splash_515404.jpg"], ["1.6M-lisbon_sunset_514529.jpg"], ["3.7M-south_boston_va_516934.jpg"], ["18M-sand_dunes_nightlife_516040.jpg"], ["1.4M-sunset_river_514676.jpg"], ["5.7M-yantra_river_514678.jpg"], ["1.3M-pigeon_2_197296.jpg"], ["5.6M-statue_of_liberty_original_model_version_2_517010.jpg"], ["6.4M-peacock_butterfly_2_514342.jpg"], ["4.6M-that_was_the_end_this_is_the_begining_513290.jpg"], ["2.4M-venice_in_winter_513381.jpg"], ["2.5M-feliz_quinta_flower_513506.jpg"], ["4.9M-seeing_red_515961.jpg"], ["1.9M-walk_away_517169.jpg"], ["3.4M-glass_chess_2_515415.jpg"], ["1.8M-opera_2_199207.jpg"], ["3.5M-standing_restrictions_explored_14th_january_2013_46_513166.jpg"], ["17M-week_2_majestic_rock_514665.jpg"], ["3.0M-french_fries_516538.jpg"], ["916K-pollution_2_514607.jpg"], ["6.1M-fire_514822.jpg"], ["7.9M-craggy_rock_517201.jpg"], ["4.6M-german_air_force_a340_3001602_517125.jpg"], ["7.8M-26365_kitty_515094.jpg"], ["7.4M-ford_gt_35_15_516157.jpg"], ["13M-forest_515915.jpg"], ["5.1M-itsthemensworld_514784.jpg"], ["5.6M-pink_sorrel_515688.jpg"], ["6.1M-simple_landscape_2_515009.jpg"], ["28M-the_river_514681.jpg"], ["5.9M-camping_516859.jpg"], ["5.7M-forest_floor_515925.jpg"], ["4.0M-quotfrench_lacequot_rose_516786.jpg"], ["2.4M-autumn_trees_2_194901.jpg"], ["4.5M-forest_513174.jpg"], ["1.4M-leaves_of_grass_515433.jpg"], ["6.8M-eiffel_ice_516309.jpg"], ["5.7M-winter_in_leeds_513733.jpg"], ["4.5M-hunting_a_toy_514521.jpg"], ["4.3M-sunset_and_evening_1_of_2_515486.jpg"], ["12M-disabled_fashion_513292.jpg"], ["3.9M-mouettes_seagulls_514235.jpg"], ["1.6M-white_tulips_514483.jpg"], ["9.2M-rose_2_514080.jpg"], ["3.2M-heathcote_515292.jpg"], ["2.5M-fashion_cupcakes_513638.jpg"], ["4.0M-hi_there_513531.jpg"], ["4.7M-cape_daisy_2_514383.jpg"], ["4.1M-backlit_flower_513144.jpg"], ["6.3M-cafe_coffe_515939.jpg"], ["8.2M-my_soul_517211.jpg"], ["7.8M-heavy_snow_2_516009.jpg"], ["1.7M-joe_joe_taking_flight_513760.jpg"], ["3.8M-sunset_and_evening_2_of_2_515480.jpg"], ["4.2M-praa_sands_2_516054.jpg"], ["30M-river_landscape_515440.jpg"], ["6.2M-dark_city_513227.jpg"], ["1.3M-one_tacky_cup_515072.jpg"]], 5491.8),

        "videoprocess": ([["1.5M-bird.avi"], ["820K-cbw3.avi"], ["660K-drop.avi"], ["6.1M-720.avi"], ["3.8M-lion-sample.avi"], ["5.6M-DLP_PART_2_768k.avi"], ["284K-flame.avi"], ["1008K-grb_2.avi"], ["2.2M.mp4"], ["404K-small.avi"], ["4.9M-640.avi"], ["5.6M.mp4"], ["3.8M.mp4"], ["4.9M.mp4"], ["6.1M.mp4"], ["2.2M-star_trails.avi"], ["11M-SampleVideo_1280x720_10mb.mp4"]], 20592.6),

        "encrypt": ([["500", "25"], ["500", "30"], ["500", "50"], ["500", "75"], ["500", "100"], ["1000", "10"], ["1000", "25"], ["1000", "30"], ["1000", "50"], ["1000", "75"], ["1000", "100"], ["5000", "10"], ["5000", "25"], ["5000", "30"], ["5000", "50"], ["5000", "75"], ["5000", "100"], ["10000", "10"], ["10000", "25"], ["10000", "30"], ["10000", "50"], ["10000", "75"], ["10000", "100"], ["35000", "10"], ["35000", "25"], ["35000", "30"], ["35000", "50"], ["35000", "75"], ["500", "10"], ["35000", "100"], ["50000", "10"], ["50000", "25"], ["50000", "30"], ["50000", "50"], ["50000", "75"], ["50000", "100"]], 41125.799999999996),

        "linpack": ([["5000"], ["500"], ["1000"], ["2000"], ["3500"], ["250"], ["750"], ["1500"], ["2500"], ["3000"], ["4000"], ["4500"]], 45008.67293666026)
    }

    port = '8080'

    output_df = pd.DataFrame(columns=['function_type', 'function_input', 'predicted_cpu', 'predicted_frequency', 'energy', 'duration', 'cpu_util', 'prediction_time'])

    with grpc.insecure_channel(f'localhost:{port}') as bayesian_channel:
        bayesian_stub = bayesian_regressor_pb2_grpc.BayesianRegressorStub(bayesian_channel)

        # with grpc.insecure_channel('localhost:50051') as lachesis_channel:
        #     lachesis_stub = lachesis_pb2_grpc.LachesisStub(lachesis_channel)

        for _ in range(5000):
            
            function_type = random.choice(list(inputs.keys()))
            input_value = inputs[function_type]
            function_input = random.choice(input_value[0])
            function_slo = input_value[1]

            print(f"Running {function_type} with input: {function_input} and SLO: {function_slo}")

            start_pred_time = time.perf_counter()

            response = None
            try:
                response = bayesian_stub.predictCPUFrequency(bayesian_regressor_pb2.predictCPUFrequencyRequest(function=function_type, inputs=function_input, slo=function_slo))
            except Exception as e:
                # Log the traceback to your server's log system
                print(traceback.format_exc())
                # Optionally, send a detailed error message back to the client
                print('Server error: ' + str(e))
                print(grpc.StatusCode.INTERNAL)
                break

            end_pred_time = time.perf_counter()

            # parse response for inferred CPU and frequency
            predicted_frequency = response.frequency
            predicted_cpu = response.cpu
            const_memory = 4096

            print(f"Predicted freq: {predicted_frequency}, predicted cpu: {predicted_cpu}")

            success, energy, latency, cpu_utilization = get_measurement_study_data(measurement_study_df[function_type], function_type, function_input, predicted_cpu, predicted_frequency)



            if success:
                # update database
                cursor.execute('INSERT INTO fxn_exec_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (function_type, 'NA', 'NA', 'NA', 'NA', latency, -1.0, function_slo, json.dumps(function_input), predicted_cpu, const_memory, -1.0, -1.0, -1.0, -1.0, -1.0, cpu_utilization, -1.0, 'NA', 'NA', 'NA', 1, predicted_cpu, const_memory, energy, predicted_frequency))
                db_conn.commit()
                data_to_add = pd.DataFrame({
                    'function_type': [function_type],
                    'function_input': [function_input],
                    'predicted_cpu': [predicted_cpu],
                    'predicted_frequency': [predicted_frequency],
                    'energy': [energy],
                    'duration': [latency],
                    'cpu_util': [cpu_utilization],
                    'prediction_time': [end_pred_time - start_pred_time]
                })
                output_df = pd.concat([output_df, data_to_add], ignore_index=True)
                output_df.to_csv("final_results_on_measurement_data.csv")
            else:
                print("ERROR: NOT SUCCESS")
            
            time.sleep(2)
        
    output_df.to_csv("final_results_on_measurement_data.csv")


            # response = lachesis_stub.Invoke(lachesis_pb2.InvokeRequest(function=function_type, slo=function_slo, parameters=function_input, cpu=predicted_cpu, memory=const_memory, frequency=predicted_frequency))
            # pk = response.primary_key
            # waitForInvocationCompletion(pk, cursor)


    db_conn.close()
    print('Completed floatmatmul invocations')

def get_measurement_study_data(df, function_type, function_input, predicted_cpu, predicted_frequency):
    df_filtered = df[(df['cpu_limit'] == predicted_cpu) & (df['frequency'] == predicted_frequency)]
    func_input_str = f'['
    for i, param in enumerate(function_input):
        final_param = None
        try:
            param_int = int(param)
            final_param = str(float(param_int))
        except:
            final_param = param
        func_input_str += f'"{final_param}"'
        if (i < len(function_input) - 1):
            func_input_str += ', '
        else:
            func_input_str += ']'
    df_filtered = df_filtered[df_filtered['inputs'] == func_input_str]
    if df_filtered.shape[0] == 0:
        print("ERROR CASE 1")
        return False, None, None, None
    energy = df_filtered['energy'].median()
    duration = df_filtered['duration'].median()
    cpu_util = df_filtered['max_cpu'].median()
    if energy <= 0 or duration <= 0 or cpu_util <= 0:
        print("ERROR CASE 2")
        return False, None, None, None
    return True, energy, duration, cpu_util
    

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
def register_functions(case = FunctionType.ALL):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        parameters = ['endpoint:\"10.52.0.193:9002\"', 'access_key:\"testkey\"', 'secret_key:\"testsecret\"', 'bucket:\"openwhisk\"']

        for cpu in range(1, CPU_MAX + 1):

            if (case == FunctionType.ALL) or (case == FunctionType.IMAGEPROCESS):
                # Image Process
                function_metadata = ['docker:psinha25/main-python']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='imageprocess', 
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/image-processing', 
                                                                    function_metadata=function_metadata, 
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,        
                                                                    cpu=cpu))
                print(response)

            if (case == FunctionType.ALL) or (case == FunctionType.FLOATMATMULT):
                # Floatmatmult
                function_metadata = ['docker:psinha25/main-python']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='floatmatmult',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/matmult',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,
                                                                    cpu=cpu))
                print(response)

            if (case == FunctionType.ALL) or (case == FunctionType.VIDEOPROCESS):
                # Video Process
                function_metadata = ['docker:psinha25/video-ow']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='videoprocess',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/video-processing',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,
                                                                    cpu=cpu))
                print(response)

            if (case == FunctionType.ALL) or (case == FunctionType.LINPACK):
                # Linpack
                function_metadata = ['docker:psinha25/main-python']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='linpack',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/linpack',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,
                                                                    cpu=cpu))
                print(response)

            if (case == FunctionType.ALL) or (case == FunctionType.ENCRYPT):
                # Encryption
                function_metadata = ['docker:psinha25/main-python']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='encrypt',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/encryption',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,
                                                                    cpu=cpu))
                print(response)

            if (case == FunctionType.ALL) or (case == FunctionType.LRTRAIN):
                # Logistic Regression Training
                function_metadata = ['docker:psinha25/lr-train-ow']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='lrtrain',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/logistic-regression-training',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,
                                                                    cpu=cpu))
                print(response)
            
            if (case == FunctionType.ALL) or (case == FunctionType.SENTIMENT):
                print("Registering Sentimenr")
                # Sentiment
                function_metadata = ['docker:psinha25/sentiment-ow']
                response = stub.Register(lachesis_pb2.RegisterRequest(function='sentiment',
                                                                    function_path='~/lachesis-2.0/benchmarks/functions/sentiment',
                                                                    function_metadata=function_metadata,
                                                                    parameters=parameters,
                                                                    memory=CONST_MEMORY,
                                                                    cpu=cpu))
                print(response)

def test_invocations():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        matmult_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=15000, parameters=['matrix1_4000_0.7.txt', 'matrix2_4000_0.7.txt']))
        # image_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=4000, parameters=[feature_dict['imageprocess'].iloc[4]['file_name']]))
        # video_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=10000, parameters=[feature_dict['videoprocess'].iloc[4]['file_name']]))
        # sentiment_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=10000, parameters=[feature_dict['sentiment'].iloc[4]['file_name']]))
        # linpack_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=10000, parameters=['5000']))
        # lrtrain_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='lrtrain', slo=10000, parameters=[feature_dict['lrtrain'].iloc[2]['file_name']]))
        # mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        # encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['10000', '30']))

        # matmult_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=15000, parameters=['matrix1_4000_0.7.txt', 'matrix2_4000_0.7.txt']))
        # image_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=4000, parameters=[feature_dict['imageprocess'].iloc[4]['file_name']]))
        # video_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=10000, parameters=[feature_dict['videoprocess'].iloc[4]['file_name']]))
        # sentiment_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=10000, parameters=[feature_dict['sentiment'].iloc[4]['file_name']]))
        # linpack_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=10000, parameters=['5000']))
        # lrtrain_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='lrtrain', slo=10000, parameters=[feature_dict['lrtrain'].iloc[2]['file_name']]))
        # mobilenet_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='mobilenet', slo=10000, parameters=[feature_dict['mobilenet'].iloc[4]['file_name']]))
        # encrypt_response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=150.75, parameters=['50', '10']))

def test_floatmatmult_invocation():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=(float(13297.5) * (1 + SLO_MULTIPLIER)), parameters=['matrix1_8000_0.5.txt', 'matrix1_8000_0.5.txt'], cpu=32, memory=CONST_MEMORY, frequency=2400000))
        pk = response.primary_key
        print(f'PK: {pk},  Resposne for function floatmatmult: {response}')

def test_videoprocess_invocation():
    changeInvokerFrequency(2400000)
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=17160.5 * (1 + SLO_MULTIPLIER), parameters=['11M-SampleVideo_1280x720_10mb.mp4'], cpu=10, memory=CONST_MEMORY, frequency=2400000))
        pk = response.primary_key
        print(f'PK: {pk},  Resposne for function videoprocess: {response}')

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
    command = f'sudo bash ~/daemon/energat_daemon/change_frequency.sh {frequency}'

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

def rerunEnergyTracer():
    command = f'sudo tmux kill-session -t energy_daemon'

    ssh_command = f"ssh {INVOKER_USERNAME}@{INVOKER_IP} '{command}'"
    process = subprocess.Popen(ssh_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f'Successfully killed energy tracer')
    else:
        print(f'Error killing energy tracer')
        print(stdout)
        print(stderr)

    command = f'cd ~/daemon/energat_daemon && sudo rm -r __pycache__ && sudo tmux new-session -d -s energy_daemon \'python3.10 __main__.py\''

    ssh_command = f"ssh {INVOKER_USERNAME}@{INVOKER_IP} '{command}'"
    process = subprocess.Popen(ssh_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    time.sleep(1)
    if process.returncode == 0:
        print(f'Successfully reran energy tracer')
        return True
    else:
        print(f'Error rerunning energy tracer')
        print(stdout)
        print(stderr)
        return False

def waitForInvocationCompletion(pk, cursor):
    while(True):
        # cursor.execute("SELECT end_time FROM fxn_exec_data WHERE row_id = ?", (pk,))
        cursor.execute("SELECT end_time FROM fxn_exec_data LIMIT 1 OFFSET ?", (pk - 1,))
        result = cursor.fetchone()

        # Check if the end_time value is "NA"
        if result and result[0] == "NA":
            # The end_time for row id {pk} is 'NA'
            time.sleep(1)
        else:
            # The end_time for row id {pk} is not 'NA'
            break

def test_linpack():
    df = feature_dict['linpack']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for i in range(0, len(df)):
            # for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
            for frequency in FREQUENCIES:
                success = rerunEnergyTracer()
                assert(success)
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                # for cpu in range(1, CPU_MAX + 1):
                for cpu in CPUS:
                    print(f"Running linpack with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['matrix_size'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='linpack', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function linpack: {response}')
                        # check if function invocation is completed
                        pk = response.primary_key
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
            # for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
            for frequency in FREQUENCIES:
                # rerun energy_tracer
                success = rerunEnergyTracer()
                assert(success)
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                # for cpu in range(1, CPU_MAX + 1):
                for cpu in CPUS:
                    print(f"Running floatmatmult with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = []
                        parameter_list.append(selected_row['file_name'])
                        parameter_list.append(selected_row['file_name'])
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)

                        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Resposne for function floatmatmult: {response}')
                        # check if function invocation is completed
                        pk = response.primary_key
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

                        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='imageprocess', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function imageprocess: {response}')
                        # check if function invocation is completed
                        pk = response.primary_key
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
            # for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
            for frequency in FREQUENCIES:
                success = rerunEnergyTracer()
                assert(success)
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                # for cpu in range(1, CPU_MAX + 1):
                for cpu in CPUS:
                    print(f"Running encrypt with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['length']), str(selected_row['iterations'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)
                        
                        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='encrypt', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function encrypt: {response}')
                        # check if function invocation is completed
                        pk = response.primary_key
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
            # for frequency in range(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_INT, FREQUENCY_INT):
            # selected_row = df.iloc[i]
            # if str(selected_row['file_name']) in ["1.5M-bird.avi", "820K-cbw3.avi", "660K-drop.avi", "6.1M-720.avi", "3.8M-lion-sample.avi"]:
                # continue
            for frequency in FREQUENCIES:
                success = rerunEnergyTracer()
                assert(success)
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                # for cpu in range(1, CPU_MAX + 1):
                for cpu in CPUS:
                    print(f"Running videoprocess with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['file_name'])]
                        slo = float(selected_row['duration']) * (1 + SLO_MULTIPLIER)    # this might be duration of video, ask Prasoon

                        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='videoprocess', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function imageprocess: {response}')
                        # check if function invocation is completed
                        pk = response.primary_key
                        waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed imageprocess invocations')

def test_sentiment():
    df = feature_dict['sentiment']
    print(df)

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)

        for i in range(0, len(df)):
            for frequency in FREQUENCIES:
                success = rerunEnergyTracer()
                assert(success)
                # handle frequency change
                success = changeInvokerFrequency(frequency)
                assert(success)
                # for cpu in range(1, CPU_MAX + 1):
                for cpu in CPUS:
                    print(f"Running sentiment with frequency: {frequency}, cpu: {cpu}, df_row: {i}, memory: {CONST_MEMORY}")
                    for _ in range(3):
                        selected_row = df.iloc[i]
                        parameter_list = [str(selected_row['file_name'])]
                        slo = 21313.8 * (1 + SLO_MULTIPLIER) # slo from cypress input

                        response = stub.Invoke(lachesis_pb2.InvokeRequest(function='sentiment', slo=slo, parameters=parameter_list, cpu=cpu, memory=CONST_MEMORY, frequency=frequency))
                        print(f'Response for function sentiment: {response}')
                        pk = response.primary_key
                        waitForInvocationCompletion(pk, cursor)
        print('Completed sentiment invocations')

def test_sequential_floatmatmult(cpu_count, frequency):

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        
        success = rerunEnergyTracer()
        assert(success)
        success = changeInvokerFrequency(frequency)
        assert(success)
        for i in range(5):
            print(f"Running floatmatmult with frequency: {frequency}, cpu: {cpu_count}, df_row: {i}, memory: {CONST_MEMORY}")
            parameter_list = []
            parameter_list.append("matrix1_8000_0.7.txt")
            parameter_list.append("matrix1_8000_0.7.txt")
            slo = 76132.2 * (1 + SLO_MULTIPLIER)

            response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list, cpu=cpu_count, memory=CONST_MEMORY, frequency=frequency))
            print(f'Resposne for function floatmatmult: {response}')
            # check if function invocation is completed
            pk = response.primary_key
            waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed floatmatmul invocations')

def test_parallel_floatmatmult(cpu_count, frequency, delay=0.5):

    db_conn = sqlite3.connect(CONTROLLER_DB)
    cursor = db_conn.cursor()

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = lachesis_pb2_grpc.LachesisStub(channel)
        pks = []

        success = rerunEnergyTracer()
        assert(success)
        success = changeInvokerFrequency(frequency)
        assert(success)
        for i in range(20):
            print(f"Running floatmatmult with frequency: {frequency}, cpu: {cpu_count}, df_row: {i}, memory: {CONST_MEMORY}")
            parameter_list = []
            parameter_list.append("matrix1_8000_0.7.txt")
            parameter_list.append("matrix1_8000_0.7.txt")
            slo = 76132.2 * (1 + SLO_MULTIPLIER)

            response = stub.Invoke(lachesis_pb2.InvokeRequest(function='floatmatmult', slo=slo, parameters=parameter_list, cpu=cpu_count, memory=CONST_MEMORY, frequency=frequency))
            print(f'Response for function floatmatmult: {response}')
            # check if function invocation is completed
            pk = response.primary_key
            pks.append(pk)
            time.sleep(delay)
        for pk in pks:
            waitForInvocationCompletion(pk, cursor)
    db_conn.close()
    print('Completed floatmatmul invocations')

if __name__=='__main__':
    # register_functions(case=FunctionType.SENTIMENT)
    # test_floatmatmult_invocation()
    # test_videoprocess_invocation()
    # test_invocations()
    # launch_slo_invocations()
    # obtain_input_duration(quantile=0.5)
    # run_experiment()

    # register_floatmatmult_mem_test_functions()
    # test_floatmatmult_scheduler_test()

    run_final_experiment()

    # test_floatmatmult_mem_test()
    # test_linpack()
    # test_floatmatmult()
    # test_sentiment()
    # test_image_process()
    # test_video_process()
    # test_encrypt()
    # test_video_process()
    # success = changeInvokerFrequency(1600000)
    # print(success)
    # rerunEnergyTracer()
    # cpu_values = [31, 25, 19, 13, 7, 3, 1]
    # for cpu_value in cpu_values:
    #     test_sequential_floatmatmult(cpu_value, 2400000)
    #     time.sleep(5)
    #     test_parallel_floatmatmult(cpu_value, 2400000)
    #     time.sleep(10)

"""
Simple Test:
Run floatmatmult with 2000 input on a container with 3 cores, 1024 mb, and 2400000 frequency

Then wait until that is done, but we still have a warm container, say container C1

Then run floatmatmult with 2000 input with 2 cores and 512 mb an 1800000 frequency.
- this should run on container C1, but create a warm container, say C2
- can check if this warm container was created using "docker stats"

Now wait for all invocations to end.

Then run floatmatmult with 2000 input with 2 cores and 512 mb and 1800000 frequency.
- this should now run on container C2 and not C1


According to the scheduler, initially, if no warm containers and all invokers have space, then it will keep iterating until it cannot.
That last invoker is the chosenInvoker and so the invocation would be routed to that.
That means that if we invoker with target 1800000 frequency, it will be run on 2400000 frequency one.
BUT we still want to warm start a container on the 1800000 frequency machine. STILL NEED TO MAKE SURE THE WARM START STUFF WORKS
"""