import pandas as pd
import socket
import subprocess
import os
import math
from sklearn.utils import shuffle

VW_IMAGE_MEMORY_MODEL_HOST = 'localhost'
VW_IMAGE_MEMORY_MODEL_PORT = 26542
MINIO_DATABASE_IP = '10.52.3.142'

image_util_path = '/home/cc/lachesis/src/datastore/imageprocess-collected-data.txt'
matmult_util_path = '/home/cc/lachesis/src/datastore/floatmatmult-collected-data.txt'

df_image_inputs = shuffle(pd.read_csv('../data/vw-prediction-inputs/imageprocess-inputs.csv'), random_state=0)
df_floatmatmult_inputs = shuffle(pd.read_csv('../data/vw-prediction-inputs/floatmatmult-inputs.csv'), random_state=0)

no_invocations = 0
prev_mem_limit = -1

# Memory units - MB
START_MEM = 2048
READY_INVOCATIONS = 10
MAX_MEM_CLASS = 160

def format_features(features):
    sample = '| '
    for feature in features:
        sample += '{0} '.format(feature)
    return sample

def invoke_fxn(invocation_no, fxn):
    if fxn == 'imageprocess':
        invocation_no = invocation_no % len(df_image_inputs)
        return df_image_inputs.iloc[invocation_no]
    elif fxn == 'floatmatmult':
        invocation_no = invocation_no % len(df_floatmatmult_inputs)
        return df_floatmatmult_inputs.iloc[invocation_no]

def launch_ow(mem_limit, invocation_no, mem_predicted, fxn, params):
    global prev_mem_limit
    results_path = ''
    fxn_registration_command = ''
    fxn_invocation_command = ''

    # Create commands
    if fxn == 'imageprocess':
        fxn_registration_command = 'cd ~/lachesis/benchmarks/functions/image-processing; wsk -i action update {} {}.py \
                                    --docker psinha25/main-python --web raw \
                                    --memory {} --cpu 2 \
                                    --param endpoint "{}:9002" \
                                    --param access_key "testkey" \
                                    --param secret_key "testsecret" \
                                    --param bucket "openwhisk"\n'.format(fxn, fxn, mem_limit, MINIO_DATABASE_IP)
        fxn_invocation_command = 'wsk -i action invoke {} \
                                --param image {} \
                                --param mem {} \
                                --param no_invocation {} \
                                --param predicted_ml {} \
                                -r -v\n'.format(fxn, params[0], mem_limit, invocation_no, mem_predicted)
        results_path = image_results_path
    elif fxn == 'floatmatmult':
        fxn_registration_command = 'cd ~/lachesis/benchmarks/functions/matmult; wsk -i action update {} {}.py \
                                    --docker psinha25/main-python --web raw \
                                    --memory {} --cpu 8 \
                                    --param endpoint "{}:9002" \
                                    --param access_key "testkey" \
                                    --param secret_key "testsecret" \
                                    --param bucket "openwhisk"\n'.format(fxn, fxn, mem_limit, MINIO_DATABASE_IP)
        fxn_invocation_command = 'wsk -i action invoke {} \
                                --param m1 {} \
                                --param m2 {} \
                                --param mem {} \
                                --param no_invocation {} \
                                --param predicted_ml {} \
                                -r -v\n'.format(fxn, params[0], params[1], mem_limit, invocation_no, mem_predicted)
        results_path = matmult_util_path

    # Update action if previous memory limit is different than current memory limit
    fxn_reg_out = 'ok: updated action'
    if no_invocations == 0 or prev_mem_limit != mem_limit:
        tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_reg_out, fxn_reg_err = tmp.communicate()
        fxn_reg_out = fxn_reg_out.decode()
        fxn_reg_err = fxn_reg_err.decode()

    # If action successfully updated, invoke
    if 'ok: updated action' in fxn_reg_out:
        prev_mem_limit = mem_limit
        results_size = os.path.getsize(results_path)

        tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_inv_out, fxn_inv_err = tmp.communicate()
        fxn_inv_out = fxn_inv_out.decode()
        fxn_inv_err = fxn_inv_err.decode()

        observed_lat = -1
        max_mem = -1
        while True:
            # If size of util file is greater than previous size, get util update
            curr_size = os.path.getsize(results_path)
            if curr_size > results_size:
                with open(results_path, 'r') as f:
                    f.seek(results_size)
                    new_lines = f.readlines()
                    for line in new_lines:
                        split_line = line.split(' ')
                        print(split_line)
                        observed_lat = float(split_line[3])
                        if fxn == 'imageprocess':
                            observed_lat = float(split_line[2])
                        max_mem = float(split_line[-3])
                        if observed_lat >= 100000.0:
                            max_mem = -1.0
                        print('Successfully launched and received data')
                    results_size = curr_size
                    return observed_lat, max_mem

def main():

    global no_invocations

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (VW_IMAGE_MEMORY_MODEL_HOST, VW_IMAGE_MEMORY_MODEL_PORT)
    s.connect(server_address)

    while True:

        # Get invocation information -- fxn and input
        fxn = 'floatmatmult'
        invocation = invoke_fxn(no_invocations, fxn)
        image = ''
        m1 = ''
        m2 = ''
        features = []
        if fxn == 'imageprocess':
            image = invocation.image
            features.append(invocation.width)
            features.append(invocation.height)
            features.append(invocation.no_channels)
            features.append(invocation.dpi_x)
            features.append(invocation.dpi_y)
            features.append(invocation.filesize)
        elif fxn == 'floatmatmult':
            m1 = 'matrix1_{}_{}.txt'.format(int(invocation.row_size), invocation.density)
            m2 = 'matrix2_{}_{}.txt'.format(int(invocation.row_size), invocation.density)
            features.append(invocation.row_size)
            features.append(invocation.col_size)
            features.append(invocation.density)
            features.append(invocation.datatype)
        
        # Get memory assignment
        mem_assigned = START_MEM
        vw_sample = format_features(features)
        vw_sample += '\n'
        s.sendall(vw_sample.encode())
        mem_predicted = int(s.recv(1024).decode().strip()) * 64
        if no_invocations >= READY_INVOCATIONS:
            mem_assigned = mem_predicted
        
        # Print some bookkeeping information
        if fxn == 'floatmatmult':
            print('Matrix: {}'.format(m1))
        elif (fxn == 'imageprocess') or (fxn == 'resnet'):
            print('Image: {}'.format(image))
        print('Invocation #: {}'.format(no_invocations))
        print('Mem Assigned: {}'.format(mem_assigned))
        print('Mem Predicted: {}'.format(mem_predicted))

        # Launch function on OpenWhisk
        observed_lat = -1
        max_mem = -1
        if fxn == 'imageprocess' or (fxn == 'resnet'):
            observed_lat, max_mem = launch_ow(mem_assigned, no_invocations, mem_predicted, fxn, [image])
        elif fxn == 'floatmatmult':
            observed_lat, max_mem = launch_ow(mem_assigned, no_invocations, mem_predicted, fxn, [m1, m2])
        no_invocations += 1
        print('Observed latency: {}'.format(observed_lat))
        print('Max memory used: {}'.format(max_mem))






if __name__=='__main__':
    main()


wsk -i action update imageprocess imageprocess.py \
    --docker psinha25/main-python --web raw \
    --memory 2048 --cpu 2 \
    --param endpoint "10.52.3.142:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"

wsk -i action invoke imageprocess \
    --param image 30M-river_landscape_515440.jpg \
    --param mem 2048 \
    --param no_invocation 1 \
    --param predicted_ml 64 \
    -r -v

wsk -i action update floatmatmult floatmatmult.py \
    --docker psinha25/main-python --web raw \
    --memory 4096 --cpu 8 \
    --param endpoint "10.52.3.142:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"

wsk -i action invoke floatmatmult \
    --param m1 matrix1_4000_0.7.txt \
    --param m2 matrix2_4000_0.7.txt \
    --param mem 4096 \
    --param cpu 8 \
    --param no_invocation 1 \
    --param predicted_ml 64 \
    -r -v

wsk -i action invoke floatmatmult_8 \
--param m1 matrix1_4000_0.7.txt \
--param m2 matrix2_4000_0.7.txt \
--param mem 4096 \
--param cpu 8 \
--param no_invocation 1 \
--param predicted_ml 64 \
-r -v