import pandas as pd
import socket
import subprocess
import os
import math
from sklearn.utils import shuffle

resnet_results_path = '/home/cc/irasc/imageprocess-collected-data-ow.txt'

prev_cpu_limit = -1

def launch_ow(cpu_limit, slo, fxn, params):
    global prev_cpu_limit
    fxn_registration_command = 'cd ~/openwhisk-benchmarks/functions/image-processing; wsk -i action update imageprocess imageprocess.py \
                                    --docker psinha25/main-python --web raw \
                                    --memory 4096 --cpu {} \
                                    --param endpoint "10.52.3.148:9002" \
                                    --param access_key "testkey" \
                                    --param secret_key "testsecret" \
                                    --param bucket "openwhisk"\n'.format(cpu_limit)   

    fxn_invocation_command = 'wsk -i action invoke {} \
                                --param image {} \
                                --param cpu {} \
                                --param slo {} \
                                -r -v\n'.format(fxn, params[0], cpu_limit, slo)                        
    
    fxn_reg_out = 'ok: updated action'
    if prev_cpu_limit != cpu_limit:
        tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_reg_out, fxn_reg_err = tmp.communicate()
        fxn_reg_out = fxn_reg_out.decode()
        fxn_reg_err = fxn_reg_err.decode()
    
    if 'ok: updated action' in fxn_reg_out:
        prev_cpu_limit = cpu_limit
        results_size = os.path.getsize(resnet_results_path)

        tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_inv_out, fxn_inv_err = tmp.communicate()
        fxn_inv_out = fxn_inv_out.decode()
        fxn_inv_err = fxn_inv_err.decode()

        count = 0
        while True:
            if count == 0:
                print('Waiting on results')
                count += 1
            curr_size = os.path.getsize(resnet_results_path)

            if curr_size > results_size:
                print("Completed image {} and cpu limit {}".format(params[0], cpu_limit))
                return

def main():

    df_resnet_inputs = shuffle(pd.read_csv('/home/cc/ow-modeling/data/imageprocess-inputs.csv'), random_state=0)
    fxn = 'imageprocess'
    cpus = [64, 32, 1]
    rows = [['6.8M-eiffel_ice_516309.jpg', 64], ['6.2M-dark_city_513227.jpg', 1], ['4.5M-more_toys_514506.jpg', 1]]
    for row in rows:
        launch_ow(row[1], 5000, fxn, [row[0]])
    # for cpu in cpus:
    #     for index, row in df_resnet_inputs.iterrows():
    #         launch_ow(cpu, 5000, fxn, [row.image])

if __name__ == '__main__':
    main()