import pandas as pd
import socket
import subprocess
import os
import math
from sklearn.utils import shuffle

image_results_path = '/home/cc/irasc/imageprocess-collected-data-ow.txt'
floatmatmult_results_path = '/home/cc/irasc/floatmatmult-collected-data-ow.txt'

prev_cpu_limit = -1

mapping = {'matrix1_2000_0.1.txt': 3000, 'matrix1_2000_0.3.txt': 3000, 'matrix1_2000_0.5.txt': 4000, 'matrix1_2000_0.7.txt': 4000, 'matrix1_2000_0.9.txt': 4000,
           'matrix1_4000_0.1.txt': 11500, 'matrix1_4000_0.3.txt': 12500, 'matrix1_4000_0.5.txt': 13000, 'matrix1_4000_0.7.txt': 14000, 'matrix1_4000_0.9.txt': 14500,
           'matrix1_6000_0.1.txt': 29000, 'matrix1_6000_0.3.txt': 31000, 'matrix1_6000_0.5.txt': 33500, 'matrix1_6000_0.7.txt': 35000, 'matrix1_6000_0.9.txt': 36000,
           'matrix1_8000_0.1.txt': 47000, 'matrix1_8000_0.3.txt': 54000, 'matrix1_8000_0.5.txt': 60000, 'matrix1_8000_0.7.txt': 63000, 'matrix1_8000_0.9.txt': 63000}



def launch_ow(cpu_limit, slo, fxn, params):
    global prev_cpu_limit
    results_path = ''
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

    fxn_invocation_command = ''
    if fxn == 'imageprocess':
        fxn_invocation_command = 'wsk -i action invoke {} \
                                --param image {} \
                                --param cpu {} \
                                --param slo {} \
                                -r -v\n'.format(fxn, params[0], cpu_limit, slo)
    elif fxn == 'floatmatmult':
        fxn_invocation_command = 'wsk -i action invoke {} \
                                    --param m1 {} \
                                    --param m2 {} \
                                    --param cpu {} \
                                    --param slo {} \
                                    -r -v\n'.format(fxn, params[0], params[1], cpu_limit, slo)
    
    fxn_reg_out = 'ok: updated action'
    if prev_cpu_limit != cpu_limit:
        tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_reg_out, fxn_reg_err = tmp.communicate()
        fxn_reg_out = fxn_reg_out.decode()
        fxn_reg_err = fxn_reg_err.decode()
    
    if 'ok: updated action' in fxn_reg_out:
        prev_cpu_limit = cpu_limit
        results_size = os.path.getsize(results_path)

        tmp = subprocess.Popen(fxn_invocation_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        fxn_inv_out, fxn_inv_err = tmp.communicate()
        fxn_inv_out = fxn_inv_out.decode()
        fxn_inv_err = fxn_inv_err.decode()

        count = 0
        while True:
            if count == 0:
                print('Waiting on results')
                count += 1
            curr_size = os.path.getsize(results_path)

            if curr_size > results_size:
                print("Completed matrix {} and cpu limit {}".format(params[0], cpu_limit))
                return

def main():

    matrix_size = [2000, 4000, 6000, 8000]
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    cpus = [64, 32, 1]
    fxn = 'floatmatmult'

    for cpu in cpus:
        for size in matrix_size:
            for density in densities:
                m1 = 'matrix1_{}_{}.txt'.format(size, density)
                m2 = 'matrix2_{}_{}.txt'.format(size, density)
                print("Starting matrix {} and cpu limit {} and slo {}".format(m1, 1, mapping[m1]))
                launch_ow(cpu, mapping[m1], fxn, [m1, m2])

if __name__ == '__main__':
    main()