import pandas as pd
import socket
import subprocess
import os
import math
from sklearn.utils import shuffle

CPU_INCREASE = 8
START_CPU = 30
READY_INVOCATIONS = 5
MAX_CPU_CLASS = 64
UNDER_PREDICTION_SEVERITY = 35
HANDLE_NANS = 4

SLO_INCREASE = 0

no_invocations = 0
prev_cpu_limit = -1

VW_IMAGE_MODEL_HOST = 'localhost'
VW_IMAGE_MODEL_PORT = 26542

image_results_path = '/home/cc/irasc/imageprocess-collected-data.txt'
floatmatmult_results_path = '/home/cc/irasc/floatmatmult-collected-data-0-dur.txt'
resnet_results_path = '/home/cc/irasc/resnet-collected-data.txt'

df_image_inputs = shuffle(pd.read_csv('/home/cc/ow-modeling/data/imageprocess-inputs.csv'), random_state=0)
df_floatmatmult_inputs = shuffle(pd.read_csv('/home/cc/ow-modeling/data/floatmatmult.csv'), random_state=0)
df_resnet_inputs = shuffle(pd.read_csv('/home/cc/ow-modeling/data/resnet.csv'), random_state=0)

def compute_costs(max_cpu): 
    '''
        This function assumes that max_cpu is in 
        the form of # of cores, not Linux percentage.
    '''
    cost = [0] * (MAX_CPU_CLASS + 1)
    for i in range(1, MAX_CPU_CLASS + 1):
        if i < max_cpu:
            cost[i] = UNDER_PREDICTION_SEVERITY + (max_cpu - i)
        else:
            cost[i] = i - max_cpu + 1
    return cost

def format_costs(costs):
    # if 1 not in costs:
    #     print(costs)
    sample = ""
    for i in range(1, MAX_CPU_CLASS+1):
        sample += '{0}:{1} '.format(i, costs[i])
    return sample
    
def format_features(features):
    sample = '| '
    for feature in features:
        sample += '{0} '.format(feature)
    return sample

def vw_format_creator(costs, features):
    sample_cost = format_costs(costs)
    sample_features = format_features(features)
    sample = sample_cost + sample_features
    return sample

def invoke_fxn(invocation_no, fxn):

    if fxn == 'floatmatmult':
        sub_df = df_floatmatmult_inputs[['row_size', 'col_size', 'density', 'datatype']]
        dur_goals = df_floatmatmult_inputs['dur_goal'] + df_floatmatmult_inputs['dur_goal'] * SLO_INCREASE
        invocation_no = invocation_no % len(sub_df)
        return sub_df.iloc[invocation_no], dur_goals.iloc[invocation_no]
    elif fxn == 'imageprocess':
        invocation_no = invocation_no % len(df_image_inputs)
        return df_image_inputs.iloc[invocation_no], 5000
    elif fxn == 'resnet':
        sub_df = df_resnet_inputs[['image', 'width', 'height', 'quality', 'filesize']]
        dur_goals = df_resnet_inputs['slo'] + df_resnet_inputs['slo'] * SLO_INCREASE
        invocation_no = invocation_no % len(sub_df)
        return sub_df.iloc[invocation_no], dur_goals.iloc[invocation_no]

def launch_ow(cpu_limit, slo, invocation_no, cpu_predicted, fxn, params):
    global prev_cpu_limit
    results_path = ''
    # Create commands
    fxn_registration_command = ''
    if fxn == 'imageprocess':
        fxn_registration_command = 'cd ~/openwhisk-benchmarks/functions/image-processing; wsk -i action update {} {}.py \
                                    --docker psinha25/main-python --web raw \
                                    --memory 4096 --cpu {} \
                                    --param endpoint "10.52.3.148:9002" \
                                    --param access_key "testkey" \
                                    --param secret_key "testsecret" \
                                    --param bucket "openwhisk"\n'.format(fxn, fxn, cpu_limit)
        results_path = image_results_path
    elif fxn == 'floatmatmult':
        fxn_registration_command = 'cd ~/openwhisk-benchmarks/functions/matmult; wsk -i action update {} {}.py \
                            --docker psinha25/main-python --web raw \
                            --memory 4096 --cpu {} \
                            --param endpoint "10.52.3.148:9002" \
                            --param access_key "testkey" \
                            --param secret_key "testsecret" \
                            --param bucket "openwhisk"\n'.format(fxn, fxn, cpu_limit)
        results_path = floatmatmult_results_path
    elif fxn == 'resnet':
        fxn_registration_command = 'cd ~/openwhisk-benchmarks/functions/resnet-50; wsk -i action update resnet resnet-50.py \
                                    --docker psinha25/resnet-50-ow --web raw \
                                    --memory 5120 --cpu {} \
                                    --param endpoint "10.52.3.148:9002" \
                                    --param access_key "testkey" \
                                    --param secret_key "testsecret" \
                                    --param bucket "openwhisk"\n'.format(cpu_limit)
        results_path = resnet_results_path                            


    fxn_invocation_command = ''
    if fxn == 'imageprocess':
        fxn_invocation_command = 'wsk -i action invoke {} \
                                --param image {} \
                                --param cpu {} \
                                --param no_invocation {} \
                                --param predicted_cores {} \
                                --param slo {} \
                                -r -v\n'.format(fxn, params[0], cpu_limit, invocation_no, cpu_predicted, slo)
        # fxn_invocation_command = 'wsk -i action invoke {} \
        #                         --param image 30M-river_landscape_515440.jpg \
        #                         --param cpu {} \
        #                         --param slo {} \
        #                         -r -v\n'.format(fxn, cpu_limit, slo)
    elif fxn == 'floatmatmult':
        fxn_invocation_command = 'wsk -i action invoke {} \
                                    --param m1 {} \
                                    --param m2 {} \
                                    --param cpu {} \
                                    --param no_invocation {} \
                                    --param predicted_cores {} \
                                    --param slo {} \
                                    -r -v\n'.format(fxn, params[0], params[1], cpu_limit, invocation_no, cpu_predicted, slo)
        # fxn_invocation_command = 'wsk -i action invoke floatmatmult \
        #                             --param m1 matrix1_8000_0.9.txt \
        #                             --param m2 matrix2_8000_0.9.txt \
        #                             --param cpu 2 \
        #                             --param slo 10000 \
        #                             -r -v\n'
    elif fxn == 'resnet':
        fxn_invocation_command = 'wsk -i action invoke {} \
                                    --param image {} \
                                    --param cpu {} \
                                    --param no_invocation {} \
                                    --param predicted_cores {} \
                                    --param slo {} \
                                    -r -v\n'.format(fxn, params[0], cpu_limit, invocation_no, cpu_predicted, slo)

    fxn_reg_out = 'ok: updated action'
    if no_invocations == 0 or prev_cpu_limit != cpu_limit:
        tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_reg_out, fxn_reg_err = tmp.communicate()
        fxn_reg_out = fxn_reg_out.decode()
        fxn_reg_err = fxn_reg_err.decode()

    if 'ok: updated action' in fxn_reg_out:
        print("Successfully updated")
        prev_cpu_limit = cpu_limit
        results_size = os.path.getsize(results_path)

        tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_inv_out, fxn_inv_err = tmp.communicate()
        fxn_inv_out = fxn_inv_out.decode()
        fxn_inv_err = fxn_inv_err.decode()

        observed_lat = -1
        max_cpu = -1
        count = 0
        while True:
            if count == 0:
                print('Waiting on results')
                count += 1
            curr_size = os.path.getsize(results_path)
            
            if curr_size > results_size:
                with open(results_path, 'r') as f:
                    # if 'code 503' not in fxn_inv_out:
                    f.seek(results_size)
                    new_lines = f.readlines()
                    for line in new_lines:
                        split_line = line.split(' ')
                        observed_lat = float(split_line[2])
                        max_cpu = float(split_line[3])
                        if observed_lat >= 100000.0:
                            max_cpu = -1.0
                        print("Successfully launched and received data")
                    results_size = curr_size
                    return observed_lat, max_cpu
     
def main():

    global no_invocations

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (VW_IMAGE_MODEL_HOST, VW_IMAGE_MODEL_PORT)
    s.connect(server_address)

    while True:

        # Get invocation information -- fxn, input, and slo
        fxn = 'imageprocess'
        invocation, slo = invoke_fxn(no_invocations, fxn)
        image = ''
        m1 = ''
        m2 = ''
        features = []
        if fxn == 'imageprocess' or 'resnet':
            image = invocation.image
            features.append(invocation.width)
            features.append(invocation.height)
            features.append(invocation.quality)
            features.append(invocation.filesize)
            features.append(slo)
        elif fxn == 'floatmatmult':
            m1 = 'matrix1_{}_{}.txt'.format(int(invocation.row_size), invocation.density)
            m2 = 'matrix2_{}_{}.txt'.format(int(invocation.row_size), invocation.density)
            features.append(invocation.row_size)
            features.append(invocation.col_size)
            features.append(invocation.density)
            features.append(invocation.datatype)
            features.append(slo)
        
        # Get CPU assignment
        cpu_assigned = START_CPU
        vw_sample = format_features(features)
        vw_sample += '\n'
        # print(vw_sample)
        s.sendall(vw_sample.encode())
        data = s.recv(1024).decode().strip()
        if no_invocations >= READY_INVOCATIONS:
            cpu_assigned = data
        
        if fxn == 'floatmatmult':
            print('Matrix: {}'.format(m1))
        else:
            print('Image: {}'.format(image))
        print('Invocation #: {}'.format(no_invocations))
        print('CPU Assigned: {}'.format(cpu_assigned))
        print('CPU Predicted: {}'.format(data))
        print('SLO Given: {}'.format(slo))

        # Launch function on openwhisk
        observed_lat = -1
        max_cpu = -1
        if (fxn == 'imageprocess') or (fxn == 'resnet'):
            observed_lat, max_cpu = launch_ow(cpu_assigned, slo, no_invocations, data, fxn, [image])
        elif fxn == 'floatmatmult':
            observed_lat, max_cpu = launch_ow(cpu_assigned, slo, no_invocations, data, fxn, [m1, m2])
        no_invocations += 1

        print('Observed latency: {}'.format(observed_lat))
        print('Max CPU cores used: {}'.format(max_cpu))

        cpu_assigned = float(cpu_assigned)
        if max_cpu != -1.0:
            slack = observed_lat - slo
            # print("Slack is: {}".format(slack))
            # print("max cpu is: {}".format(max_cpu))
            # print("cpu assigned is: {}".format(cpu_assigned))
            if slack > 0:
                '''
                    Increase max_cpu that should've been used
                    by 1 for every 0.5 seconds over the SLO
                '''
                print("increasing")
                cpu_increase = min(math.ceil(slack / 500), 10)
                max_cpu += cpu_increase
                if max_cpu > MAX_CPU_CLASS:
                    max_cpu = 64
                print("Oh shit, didn't meet the SLO!")
            elif (max_cpu >= cpu_assigned):
                '''
                    Decrease max_cpu that should've been used 
                    by 1 for every 1.5 seconds we are under SLO
                '''
                print("decreasing")
                slack = abs(slack)
                max_cpu = cpu_assigned - math.floor(slack/1500)
                if max_cpu < 1:
                    max_cpu = 1
            print('Max CPU cores assigned: {}'.format(max_cpu))

            # Retrain VW CSOAA model
            costs = compute_costs(max_cpu)
            vw_sample = vw_format_creator(costs, features)
            vw_sample += '\n'
            
            s.sendall(vw_sample.encode())
            data = s.recv(1024).decode().strip()
            # print('Data received from vw daemon: {}'.format(data))
            print("")
        else:
            print("Probably ran into an OW error")
            print("")

if __name__=="__main__":
    main()