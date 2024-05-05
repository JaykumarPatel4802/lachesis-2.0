import grpc
import threading
import time
import sqlite3
from concurrent import futures
from generated import bayesian_regressor_pb2_grpc, bayesian_regressor_pb2
import ast
import random

import warnings
warnings.filterwarnings("ignore")



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CONTROLLER_DB = '../datastore/lachesis-controller.db'
FLOATMATMULT = 'floatmatmult'
IMAGEPROCESS = 'imageprocess'
VIDEOPROCESS = 'videoprocess'
LINPACK = 'linpack'
ENCRYPT = 'encrypt'

RANDOM = 'random'
MODEL_MIN = 'model_min'

INPUT_DIR_PATH = '../../data/vw-prediction-inputs/'

THRESHOLD = 0.2

normalization_min_max = {
    "floatmatmult": {
        "row_size": (1000, 8000),
        "col_size": (1000, 8000),
        "density": (0.1, 0.9)
    },
    "imageprocess": {
        "width": (2128, 7360),
        "height": (1601, 5616),
        "no_channels": (1, 3),
        "dpi_x": (72, 4136),
        "dpi_y": (72, 4136),
        "filesize": (934549, 31263363)
    },
    "videoprocess": {
        "file_size": (289280, 10498677),
        "width": (256, 1280),
        "height": (240, 720),
        "bitrate": (208702, 1128710),
        "avg_frame_rate": (23.976023976023978, 40)
    },
    "encrypt": {
        "length": (500, 50000),
        "iterations": (10, 100)
    },
    "linpack": {
        "matrix_size": (250, 10000)
    }
}

energy_min_max = {
    "floatmatmult": (0.0, 0.0),
    "imageprocess": (0.0, 0.0),
    "videoprocess": (0.0, 0.0),
    "encrypt": (0.0, 0.0),
    "linpack": (0.0, 0.0)
}

# FREQUENCIES = [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000]
# CPUS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

FREQUENCIES = [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2400000]
CPUS = [1, 2, 3, 5, 7, 10, 13, 16, 19, 22, 25, 28, 31]

class BayesianRegressor(bayesian_regressor_pb2_grpc.BayesianRegressorServicer):
    def __init__(self):
        kernel = 1.0 * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.floatmatmult_lock = threading.Lock()
        self.imageprocess_lock = threading.Lock()
        self.videoprocess_lock = threading.Lock()
        self.linpack_lock = threading.Lock()
        self.encrypt_lock = threading.Lock()
        # Initialize Gaussian Process models
        self.floatmatmult_energy_model = GaussianProcessRegressor(kernel=kernel)
        self.floatmatmult_time_model = GaussianProcessRegressor(kernel=kernel)
        self.imageprocess_energy_model = GaussianProcessRegressor(kernel=kernel)
        self.imageprocess_time_model = GaussianProcessRegressor(kernel=kernel)
        self.videoprocess_energy_model = GaussianProcessRegressor(kernel=kernel)
        self.videoprocess_time_model = GaussianProcessRegressor(kernel=kernel)
        self.linpack_energy_model = GaussianProcessRegressor(kernel=kernel)
        self.linpack_time_model = GaussianProcessRegressor(kernel=kernel)
        self.encrypt_energy_model = GaussianProcessRegressor(kernel=kernel)
        self.encrypt_time_model = GaussianProcessRegressor(kernel=kernel)

    def normalize_inputs(self,input_df, function_type):
        # print("1")
        new_input_df = input_df.copy(deep=True)
        # print(type(input_df))
        # print(input_df.columns)
        for column in new_input_df.columns:
            if column == "max_cpu":
                new_input_df[column] = (new_input_df[column]-1)/(31-1)
            elif column == "frequency":
                new_input_df[column] = (new_input_df[column]-1000000) / (2400000 - 1000000)
            else:
                if column not in normalization_min_max[function_type].keys():
                    new_input_df[column] = input_df[column]
                    continue
                min = normalization_min_max[function_type][column][0]
                max =  normalization_min_max[function_type][column][1]
                new_input_df[column] = (new_input_df[column] - min )/(max - min)
        # print("1 Finish")
        return new_input_df
        # return input_df

    def predictCPUFrequency(self, request, context):
        # print("2")
        print(f"Received request for: {request.function}, {request.inputs}, {request.slo}")
        #request.
        function_type = request.function
        inference = tuple()
        energy = 0
        if function_type == FLOATMATMULT:
            inference, energy, prediction_type, confidence = self.predict_floatmatmult(request)
        elif function_type == IMAGEPROCESS:
            inference, energy, prediction_type, confidence= self.predict_imageprocess(request)
        elif function_type == VIDEOPROCESS:
            inference, energy, prediction_type, confidence =  self.predict_videoprocess(request)
        elif function_type == LINPACK:
            inference, energy, prediction_type, confidence = self.predict_linpack(request)
        elif function_type == ENCRYPT:
            inference, energy, prediction_type, confidence = self.predict_encrypt(request)
        else:
            return bayesian_regressor_pb2.Reply(status='Function type not found', cpu = 0, frequency = 0, energy = 0, prediction_type = "", confidence = 0)

        print(f"Inference: {inference}")
        # print(f"2 Finish")
        
        return bayesian_regressor_pb2.Reply(status='SUCCESS', cpu = inference[0], frequency = inference[1], prediction_type = prediction_type, energy = energy, confidence = confidence)

    def predict_floatmatmult(self, data):
        # print("Predicting floatmatmult")
        # print("3")
        # Prediction logic for FLOATMATMULT
        #get all data rows that correspond to this function type
        slo = data.slo
        function_input = data.inputs[0]
        input_csv = pd.read_csv(INPUT_DIR_PATH + 'floatmatmult-inputs.csv')
        #get the row that corresponds to the input
        # print(input_csv)
        # print(f"FUNCTION INPUT: {function_input}")
        input_row = input_csv[input_csv['file_name'] == function_input]
        input_features = input_row[['row_size', 'col_size', 'density']]
        # print(f"INPUT OG FEATURES {input_features}")
        #convert input_features to a list (no column names)
        # input_features = input_features.to_numpy().tolist()[0]
        # print("3 Finish")
        inference, energy, prediction_type, confidence =  self.get_configurations(self.floatmatmult_energy_model, self.floatmatmult_time_model, input_features, slo, FLOATMATMULT)
        
        # print(f"CONFIDENCE: {confidence}")
        # print(f"ENERGY: {energy}")
        return inference, energy, prediction_type, confidence

        
       
       
    def get_configurations(self, energy_model, time_model, input_features, slo, function_type):
        # print("Getting config")
        # print("4", function_type)
        #acquire lock based on function type
        if function_type == FLOATMATMULT:
            lock = self.floatmatmult_lock
        elif function_type == IMAGEPROCESS:
            lock = self.imageprocess_lock
        elif function_type == VIDEOPROCESS:
            lock = self.videoprocess_lock
        elif function_type == LINPACK:
            lock = self.linpack_lock
        elif function_type == ENCRYPT:
            lock = self.encrypt_lock

        # print(f"TEST: energy_model id: {id(self.floatmatmult_energy_model)}")
            
        #acquire lock
        config_energies = {}
        # config_energies_violating_slo = {}
        with lock:
            for cpu in CPUS:
                for frequency in FREQUENCIES:
                    # inputs = input_features + [cpu, frequency]
                    input_features['max_cpu'] = [cpu]
                    input_features["frequency"] = [frequency]
                    # print(f"INPUT FEATURES: {input_features}")
                    inputs = self.normalize_inputs(input_features, function_type)
                    # print(f"Inputs: {inputs}")
                    inputs = np.array(inputs).reshape(1, -1)
                    energy, confidence = energy_model.predict(inputs, return_std=True)
                    # print(f"Model PRE predicted energy: {energy}, with type: {type(energy)}, with shape: {energy.shape}")
                    # if energy != 0.0:
                    if energy.shape == ():
                        energy = energy.item()
                    else:
                        energy = energy[0]
                    confidence = confidence[0]
                    # print(f"E: {energy}, C: {confidence}")                              
                    time = time_model.predict(inputs)
                    if time < slo:
                        config_energies[(cpu, frequency)] = (energy, confidence)
                    # else:
                        # config_energies_violating_slo[(cpu, frequency)] = (energy, confidence)
                    
 
                    
        #sort the config_energies by energy
        sorted_config_energies = sorted(config_energies.items(), key=lambda x: x[1][0])
        
        
        if len(sorted_config_energies) == 0 or len(sorted_config_energies[0]) == 0:
            # print("4.1 Finish", function_type)
            return ((random.choice(CPUS), random.choice(FREQUENCIES)), -1.0, RANDOM, -1.0)

        minimum_energy = sorted_config_energies[0][1][0]

        #if confidence is less than threshold (ex 0.3), return the minimum energy
        # print(f"Confidence is: {sorted_config_energies[0][1][1]}")
        # for i in range(25):
            # print(f"Confidence at {i} is: {sorted_config_energies[i][1][1]}")


        min_energy = None
        min_energy_confidence = None
        inference = None
        for config in sorted_config_energies:
            if sorted_config_energies[0][1][1] < 0.2:
                temp_energy = config[1][0]
                if min_energy == None or temp_energy < min_energy:
                    min_energy = temp_energy
                    min_energy_confidence = config[1][1]
                    inference = config[0]
        if min_energy != None:
            energy_norm = min_energy
            # denormalize the energy value
            min_energy = energy_min_max[function_type][0]
            max_energy = energy_min_max[function_type][1]
            if min_energy == max_energy:
                energy = energy_norm
            else:
                energy = (energy_norm * (max_energy - min_energy)) + min_energy
            return inference, energy, MODEL_MIN, min_energy_confidence
        else:
            random_config = random.choice(sorted_config_energies)
            # print(f"RANDOM CONFIG: {random_config}")
            energy_norm = random_config[1][0]   
            #denormalize the energy value
            min_energy = energy_min_max[function_type][0]
            max_energy = energy_min_max[function_type][1]
            # print(f"MIN: {min_energy}, MAX: {max_energy}")
            if min_energy == max_energy:
                energy = energy_norm
            else:
                energy = (energy_norm * (max_energy - min_energy)) + min_energy
            # print(f"RANDOM ENERGY: {energy}")
            confidence = random_config[1][1]
            # print("4.3 Finish", function_type)
            # print(f"random config: {random_config[0]}")
            # print(f"RANDOM E: {energy}, C: {confidence}")
            return random_config[0], energy, RANDOM, confidence


        # if sorted_config_energies[0][1][1] < 0.2:
        #     #return configuration with minimum energy
        #     # print(f"Proving confident results")
        #     # print("4.2 Finish")
        #     # print(f"RANDOM CONFIG: {sorted_config_energies[0]}")
        #     energy_norm = sorted_config_energies[0][1][0]
        #     #denormalize the energy value
        #     min_energy = energy_min_max[function_type][0]
        #     max_energy = energy_min_max[function_type][1]
        #     # print(f"MIN: {min_energy}, MAX: {max_energy}")
        #     if min_energy == max_energy:
        #         energy = energy_norm
        #     else:
        #         energy = (energy_norm * (max_energy - min_energy)) + min_energy
        #     # print(f"PREDICTED ENERGY: {energy}")
            
        #     confidence = sorted_config_energies[0][1][1]
        #     # print(f"MIN: E: {energy}, C: {confidence}")
        #     # print("4.2 Finish", function_type)
        #     # print(f"random config: {sorted_config_energies[0][0]}")
        #     return sorted_config_energies[0][0], energy, MODEL_MIN, confidence
        # else:
        #     #randomly sample a configuration
        #     random_config = random.choice(sorted_config_energies)
        #     # print(f"RANDOM CONFIG: {random_config}")
        #     energy_norm = random_config[1][0]   
        #     #denormalize the energy value
        #     min_energy = energy_min_max[function_type][0]
        #     max_energy = energy_min_max[function_type][1]
        #     # print(f"MIN: {min_energy}, MAX: {max_energy}")
        #     if min_energy == max_energy:
        #         energy = energy_norm
        #     else:
        #         energy = (energy_norm * (max_energy - min_energy)) + min_energy
        #     # print(f"RANDOM ENERGY: {energy}")
        #     confidence = random_config[1][1]
        #     # print("4.3 Finish", function_type)
        #     # print(f"random config: {random_config[0]}")
        #     # print(f"RANDOM E: {energy}, C: {confidence}")
        #     return random_config[0], energy, RANDOM, confidence
        
        
        
    def predict_imageprocess(self, data):
        # print("5")
        # Prediction logic for IMAGEPROCESS
        
        slo = data.slo
        function_input = data.inputs[0]
        input_csv = pd.read_csv(INPUT_DIR_PATH + 'imageprocess-inputs.csv')
        #get the row that corresponds to the input
        input_row = input_csv[input_csv['file_name'] == function_input]
        #drop file_name and duration columns
        input_features = input_row.drop(['file_name', 'duration'], axis=1)
        #convert to numpy array
        # input_features = input_features.to_numpy().tolist()[0]
        # print("5 Finish")
        return self.get_configurations(self.imageprocess_energy_model, self.imageprocess_time_model, input_features, slo, IMAGEPROCESS)


    def predict_videoprocess(self, data):
        # print("6")
        # Prediction logic for VIDEOPROCESS
        slo = data.slo
        function_input = data.inputs[0]
        input_csv = pd.read_csv(INPUT_DIR_PATH + 'videoprocess-inputs.csv')
        #get the row that corresponds to the input
        input_row = input_csv[input_csv['file_name'] == function_input]
        
        input_features = input_row.drop(['file_name', 'duration'], axis=1)
        # input_features = input_features.to_numpy().tolist()[0]
        # print("6 Finish")
        return self.get_configurations(self.videoprocess_energy_model, self.videoprocess_time_model, input_features, slo, VIDEOPROCESS)

    def predict_linpack(self, data):
        # print("7")
        # Prediction logic for LINPACK
        slo = data.slo
        function_input = float(data.inputs[0])
        #function input is the only input feature
        # input_features = [function_input]
        input_features =  pd.DataFrame({
            "matrix_size" : [function_input]
        })
        # print("7 Finish")
        return self.get_configurations(self.linpack_energy_model, self.linpack_time_model, input_features, slo, LINPACK)

    def predict_encrypt(self, data):
        # print("8")
        # Prediction logic for ENCRYPT
        slo = data.slo
        function_input = data.inputs
        #this input is a list of two values
        # input_features = [float(function_input[0]), float(function_input[1])]
        input_features = pd.DataFrame({
            "length" : [float(function_input[0])],
            "iterations" : [float(function_input[1])]
        })
        # print("8 Finish")
        return self.get_configurations(self.encrypt_energy_model, self.encrypt_time_model, input_features, slo,ENCRYPT)

    def train_models(self):
        while True:
            # print("6")
            # Train all the models
            print("Training FLOATMATMULT model")
            self.train_floatmatmult_model()
            print("Training IMAGEPROCESS model")
            self.train_imageprocess_model()
            print("Training VIDEOPROCESS model")
            self.train_videoprocess_model()
            print("Training LINPACK model")
            self.train_linpack_model()
            print("Training ENCRYPT model")
            self.train_encrypt_model()
            # Sleep for x minutes (e.g., 30 minutes) before the next training iteration
            #call one of the predict functions to get the configurations
            # print("TESTING PREDICTION FUNCTIONs")
            # data = {'slo': 0.5, 'function': ENCRYPT, 'inputs': ['500', '25']}
            # data = bayesian_regressor_pb2.predictCPUFrequencyRequest(slo=data['slo'], function=data['function'], inputs=data['inputs'])
            # print(self.predict_encrypt(data))
            training_period = 60
            time.sleep(training_period)
            # print("6 Finish")

                
    
    def train_floatmatmult_model(self):
        # print("Training FLOATMATMULT model")
        # Training logic for FLOATMATMULT models
        # floatmatmultdb = "../datastore/lachesis-controller-floatmatmult-completed.db"
        # db_conn = sqlite3.connect(floatmatmultdb)
        db_conn = sqlite3.connect(CONTROLLER_DB)
        db_conn.row_factory = sqlite3.Row
        cursor = db_conn.cursor()
        #select all rows that correspond to the function type, but where energy is not equal to -1
        cursor.execute("SELECT * FROM fxn_exec_data WHERE function=? AND energy > 0", (FLOATMATMULT,))
        data_rows = cursor.fetchall()
        if not data_rows:
            print("no data")
            return
        db_conn.close()
        
        #convert rows to dataframe
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=column_names)
        df['inputs'] = df['inputs'].str.strip('[]').str.split(', ')
        df['inputs'] = df['inputs'].str[0].str.strip('\"')
        #rename 'inputs' to 'file_name'
        df.rename(columns={'inputs': 'file_name'}, inplace=True)
        input_features_csv = pd.read_csv(INPUT_DIR_PATH + 'floatmatmult-inputs.csv').drop(['duration'], axis=1)
        
        #merge the input features csv with the dataframe on the 'file_name' column
        df = pd.merge(df, input_features_csv, on='file_name')       
        
        inputs = df[['row_size', 'col_size', 'density', 'max_cpu', 'frequency']]
        inputs = self.normalize_inputs(inputs, FLOATMATMULT)

        min_value = df['energy'].min()
        max_value = df['energy'].max()
        energy_min_max[FLOATMATMULT] = (min_value, max_value)
        if max_value == min_value:
            df['energy_normalized'] = df['energy']
        else:
            df['energy_normalized'] = (df['energy'] - min_value) / (max_value - min_value)
        
        target_energy = df['energy_normalized']
        # print(f"TARGET ENERGY{target_energy}")

        target_time = df['duration']
        
        #train the energy and time models
        with self.floatmatmult_lock:
            # print(f"TRAIN: energy_model id: {id(self.floatmatmult_energy_model)}")
            self.floatmatmult_energy_model.fit(inputs, target_energy)
            # print(f"TRAIN 2: energy_model id: {id(self.floatmatmult_energy_model)}")
            self.floatmatmult_time_model.fit(inputs, target_time)
        # print("Training FLOATMATMULT model finished")
            

        
    
    def train_imageprocess_model(self):
        # print("Training IMAGEPROCESS model")
        # Training logic for IMAGEPROCESS models
        # imagedb = "../datastore/lachesis-controller-image-half-completed.db"
        # db_conn = sqlite3.connect(imagedb)
        # print("7")
        db_conn = sqlite3.connect(CONTROLLER_DB)
        db_conn.row_factory = sqlite3.Row
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM fxn_exec_data WHERE function=? AND energy > 0", (IMAGEPROCESS,))
        data_rows = cursor.fetchall()
        if not data_rows:
            return
        db_conn.close()
        
        #convert rows to dataframe
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=column_names)
        #keep 500 rows
        df = df.head(1500)
        
        input_features_csv = pd.read_csv(INPUT_DIR_PATH + 'imageprocess-inputs.csv').drop(['duration'], axis=1)
        #rename df['inputs'] to df['file_name']
        df.rename(columns={'inputs': 'file_name'}, inplace=True)
        #strip the '[] and ' \"' from the inputs column and no spliting
        df['file_name'] = df['file_name'].str.strip('[]').str.strip('\"')
       
        #merge the input features csv with the dataframe on the 'file_name' column
        features = input_features_csv.columns.drop(['file_name'])
        df = pd.merge(df, input_features_csv, on='file_name')

        input_feature_list = features.tolist() + ['max_cpu', 'frequency']
        
        inputs = df[input_feature_list]
        inputs = self.normalize_inputs(inputs, IMAGEPROCESS)

        min_value = df['energy'].min()
        max_value = df['energy'].max()
        energy_min_max[IMAGEPROCESS] = (min_value, max_value)
        if max_value == min_value:
            df['energy_normalized'] = df['energy']
        else:
            df['energy_normalized'] = (df['energy'] - min_value) / (max_value - min_value)
        
        target_energy = df['energy_normalized']
        target_time = df['duration']
        
        #train the energy and time models
        with self.imageprocess_lock:
            self.imageprocess_energy_model.fit(inputs, target_energy)
            self.imageprocess_time_model.fit(inputs, target_time)
        # print("Training IMAGEPROCESS model finished")
        # print("7 Finish")
            
        
    def train_videoprocess_model(self):
        # print("Training VIDEOPROCESS model")
        # Training logic for VIDEOPROCESS models
        # videodb = "../datastore/lachesis-controller-partial-video.db"
        # db_conn = sqlite3.connect(videodb)
        db_conn = sqlite3.connect(CONTROLLER_DB)
        db_conn.row_factory = sqlite3.Row
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM fxn_exec_data WHERE function=? AND energy > 0", (VIDEOPROCESS,))
        data_rows = cursor.fetchall()
        if not data_rows:
            return
        db_conn.close()
        
        #convert rows to dataframe
        '''CHECK IF THIS IS ACTUALLY GOOD'''
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=column_names)
        # df = df.head()
        # print(f"DF: {df.head()}")
                
        input_features_csv = pd.read_csv(INPUT_DIR_PATH + 'videoprocess-inputs.csv').drop(['duration'], axis=1)
        
        #rename df['inputs'] to df['file_name']
        df.rename(columns={'inputs': 'file_name'}, inplace=True)
        #strip the '[] and ' \"' from the inputs column and no spliting
        df['file_name'] = df['file_name'].str.strip('[]').str.strip('\"')
        #merge the input features csv with the dataframe on the 'file_name' column
        features = input_features_csv.columns
        df = pd.merge(df, input_features_csv, on='file_name')
        
        #remove file_name from features
        features = features.drop(['file_name'])
        input_feature_list = features.tolist() + ['max_cpu', 'frequency']

        # print("INPUT FEATURES: ", input_feature_list)
        
        inputs = df[input_feature_list]
        # print("INPUTS: ", inputs)
        inputs = self.normalize_inputs(inputs, VIDEOPROCESS)
        min_value = df['energy'].min()
        max_value = df['energy'].max()
        # print(f"Min energy: {min_value}, Max energy: {max_value}")
        energy_min_max[VIDEOPROCESS] = (min_value, max_value)
        # print("DF[energy]: ", float(df['energy']))
        if max_value == min_value:
            df['energy_normalized'] = df['energy']
        else:
            df['energy_normalized'] = (df['energy'] - min_value) / (max_value - min_value)
        
        target_energy = df['energy_normalized']
        target_time = df['duration']
        
        #train the energy and time models
        with self.videoprocess_lock:
            # print(f"target energy: {target_energy}")
            self.videoprocess_energy_model.fit(inputs, target_energy)
            self.videoprocess_time_model.fit(inputs, target_time)
        # print("Training VIDEOPROCESS model finished")
            
        

    def train_linpack_model(self):
        # print("Training LINPACK model")
        # Training logic for LINPACK models
        # linpackdb = "../datastore/lachesis-controller-linpack-midway-snapshot.db"
        # db_conn = sqlite3.connect(linpackdb)
        db_conn = sqlite3.connect(CONTROLLER_DB)

        db_conn.row_factory = sqlite3.Row
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM fxn_exec_data WHERE function=? AND energy > 0", (LINPACK,))
        data_rows = cursor.fetchall()
        if not data_rows:
            return
        db_conn.close()
    
        #convert rows to dataframe
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=column_names)
        # df = df.head()

        df['matrix_size'] = df['inputs'].str.strip('[]').str.strip('\"').astype(float)

        inputs = df[['matrix_size', 'max_cpu', 'frequency']]
        inputs = self.normalize_inputs(inputs, LINPACK)
        min_value = df['energy'].min()
        max_value = df['energy'].max()
        energy_min_max[LINPACK] = (min_value, max_value)
        if max_value == min_value:
            df['energy_normalized'] = df['energy']
        else:
            df['energy_normalized'] = (df['energy'] - min_value) / (max_value - min_value)
        
        target_energy = df['energy_normalized']
        target_time = df['duration']
        
        #train the energy and time models
        with self.linpack_lock:
            self.linpack_energy_model.fit(inputs, target_energy)
            self.linpack_time_model.fit(inputs, target_time)
        # print("Training LINPACK model finished")
            
        
            
           

    def train_encrypt_model(self):
        # print("Training ENCRYPT model")
        # Training logic for ENCRYPT models
        # encryptdb = "../datastore/lachesis-controller-encrypt-completed.db"
        # db_conn = sqlite3.connect(encryptdb)
        db_conn = sqlite3.connect(CONTROLLER_DB)
        db_conn.row_factory = sqlite3.Row
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM fxn_exec_data WHERE function=? AND energy > 0", (ENCRYPT,))
        data_rows = cursor.fetchall()
        if not data_rows:
            return
        db_conn.close()
        
        #convert rows to dataframe
        column_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data_rows, columns=column_names)
        # df = df.head()
        
        df['length'] = df['inputs'].str.strip('[]').str.split(', ').str[0].str.strip('\"').astype(float)
        df['iterations'] = df['inputs'].str.strip('[]').str.split(', ').str[1].str.strip('\"').astype(float)
        
        
        inputs = df[['length', 'iterations', 'max_cpu', 'frequency']]
        inputs = self.normalize_inputs(inputs, ENCRYPT)
        min_value = df['energy'].min()
        max_value = df['energy'].max()
        energy_min_max[ENCRYPT] = (min_value, max_value)
        if max_value == min_value:
            df['energy_normalized'] = df['energy']
        else:
            df['energy_normalized'] = (df['energy'] - min_value) / (max_value - min_value)
        
        target_energy = df['energy_normalized']
        target_time = df['duration']
        
        #train the energy and time models
        with self.encrypt_lock:
            self.encrypt_energy_model.fit(inputs, target_energy)
            self.encrypt_time_model.fit(inputs, target_time)
        # print("Training ENCRYPT model finished")


def run_process():
    # Start the gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model = BayesianRegressor()
    bayesian_regressor_pb2_grpc.add_BayesianRegressorServicer_to_server(model, server)
    server.add_insecure_port('[::]:8080')
    server.start()

    # Start the model training thread
    training_thread = threading.Thread(target=model.train_models)
    training_thread.daemon = True  # Daemon thread so it stops when the main process exits
    training_thread.start()

    server.wait_for_termination()

if __name__ == '__main__':
    run_process()
