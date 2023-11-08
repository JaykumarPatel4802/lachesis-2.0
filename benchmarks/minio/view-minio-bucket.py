import os
from minio import Minio
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--minio-config', dest='minio_config', type=str, required=True)
args = parser.parse_args()
minio_config_f = args.minio_config
with open(minio_config_f ,'r') as f:
    minio_config = json.load(f)
    endpoint = minio_config['endpoint']
    bucket = minio_config['bucket']
    access_key = minio_config['access_key']
    secret_key = minio_config['secret_key']

# Set up the Minio client
minio_client = Minio(endpoint=endpoint,
                     access_key=access_key,
                     secret_key=secret_key,
                     secure=False)

found = minio_client.bucket_exists(bucket)
if not found:
    minio_client.make_bucket(bucket)
else:
    print("Bucket '%s' already exists" %bucket)

# List objects in a bucket
objects = minio_client.list_objects(bucket, recursive=True)

# Print object names
for obj in objects:
    print(obj.object_name)