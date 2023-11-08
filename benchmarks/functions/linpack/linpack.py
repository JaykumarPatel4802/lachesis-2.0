from numpy import matrix, linalg, random, amax
from time import time
# from minio import Minio
# import pandas as pd

def linpack(N):
    # eps = 2.22e-16

    # ops = (2.0*N)*N*N/3.0+(2.0*N)*N

    # Read matrices from files and convert them to numpy matrices
    # m1 = pd.read_csv(m1_path, sep=' ', header=None).to_numpy()
    # m2 = pd.read_csv(m2_path, sep=' ', header=None).to_numpy()

    # Create AxA array of random numbers -0.5 to 0.5
    A = random.random_sample((N, N))-0.5
    B = A.sum(axis=1)

    # Convert to matrices
    A = matrix(A)
    # A = matrix(m1)

    B = matrix(B.reshape((N, 1)))
    # B = matrix(m2)

    start = time()
    X = linalg.solve(A, B)
    latency = time() - start

    # mflops = (ops*1e-6/latency)

    result = {
        'latency': latency
    }

    return result

def main(params):
    # endpoint = params['endpoint']
    # access_key = params['access_key']
    # secret_key = params['secret_key']
    # bucket = params['bucket']

    # minio_client = Minio(endpoint=endpoint,
    #                      access_key=access_key,
    #                      secret_key=secret_key,
    #                      secure=False)
    # found = minio_client.bucket_exists(bucket)
    # if not found:
    #     print(f'Bucket {bucket} does not exist')
    
    # m1 = params['input1']
    # m2 = params['input2']
    # m1_path = '/tmp/' + m1
    # m2_path = '/tmp/' + m2

    # minio_client.fget_object(bucket_name=bucket,
    #                          object_name=m1,
    #                          file_path=m1_path)

    # minio_client.fget_object(bucket_name=bucket,
    #                          object_name=m2,
    #                          file_path=m2_path)
    
    return linpack(int(params['input1']))
    