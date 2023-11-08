import numpy as np
import pandas as pd
from time import time
from minio import Minio

def multiply_matrices(m1_path, m2_path):

    start = time()

    # Read matrices from files and convert them to numpy matrices
    matrix1 = pd.read_csv(m1_path, sep=' ', header=None).to_numpy()
    matrix2 = pd.read_csv(m2_path, sep=' ', header=None).to_numpy()

    # Perform matrix multiplication
    result_matrix = np.matmul(matrix1, matrix2)
    
    # Write results
    with open('/tmp/out.txt', 'w') as file:
        np.savetxt(file, result_matrix, fmt='%f')

    latency = time() - start
    
    return latency

def main(params):

    endpoint = params['endpoint']
    access_key = params['access_key']
    secret_key = params['secret_key']
    bucket = params['bucket']

    minio_client = Minio(endpoint=endpoint,
                     access_key=access_key,
                     secret_key=secret_key,
                     secure=False)
    found = minio_client.bucket_exists(bucket)
    if not found:
        print("Bucket '%s' does not exist" %bucket)
    
    m1 = params['input1']
    m2 = params['input2']
    m1_path = '/tmp/' + m1
    m2_path = '/tmp/' + m2

    minio_client.fget_object(bucket_name=bucket,
                             object_name=m1,
                             file_path=m1_path)

    minio_client.fget_object(bucket_name=bucket,
                             object_name=m2,
                             file_path=m2_path)

    lat = multiply_matrices(m1_path, m2_path)
    
    ret_val = {}
    ret_val['latency'] = lat
    return ret_val