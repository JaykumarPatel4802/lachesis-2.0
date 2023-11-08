import sys
import os
import subprocess
import argparse
from minio import Minio
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--minio-config', dest='minio_config', type=str, required=True)
data_top_dir =  Path.cwd() / '..' / 'minio-data'
image_dir = data_top_dir / 'image-process'
sentiment_dir = data_top_dir / 'sentiment'
video_dir = data_top_dir / 'video'
lr_dir = data_top_dir / 'logistic-regression'
audio_dir = data_top_dir / 'audio'
mat_dir = data_top_dir / 'float-matrices'
resnet_dir = data_top_dir / 'resnet/model'
resnet_image_dir = data_top_dir / 'resnet-images'
linpack_dir = data_top_dir / 'linpack-matrices'

# print(data_top_dir)
# print(image_dir)
# print(sentiment_dir)
# print(video_dir)
# print(lr_dir)
# print(mat_dir)
# print(resnet_dir)

args = parser.parse_args()
minio_config_f = args.minio_config
with open(minio_config_f ,'r') as f:
    minio_config = json.load(f)
    endpoint = minio_config['endpoint']
    access_key = minio_config['access_key']
    secret_key = minio_config['secret_key']

# Deploy Minio
cmd = 'docker compose -f docker-compose.yml up -d'
subprocess.run(cmd, shell=True)
minio_client = Minio(endpoint=endpoint,
                     access_key=access_key,
                     secret_key=secret_key,
                     secure=False)
bucket_name = "openwhisk"
found = minio_client.bucket_exists(bucket_name)
if not found:
    minio_client.make_bucket(bucket_name)
else:
    print("Bucket '%s' already exists" %bucket_name)

# Images
# for img in os.listdir(str(image_dir)):
#     img_path = image_dir / img
#     print(img_path)
#     minio_client.fput_object(bucket_name=bucket_name,
#                        object_name=img,
#                        file_path=str(img_path))

# # Sentiment
# for sentence in os.listdir(str(sentiment_dir)):
#     if "json" in sentence:
#         sentence_path = sentiment_dir / sentence
#         print(sentence_path)
#         minio_client.fput_object(bucket_name=bucket_name,
#                              object_name=sentence,
#                              file_path=str(sentence_path))

# # Video
# for vid in os.listdir(str(video_dir)):
#     vid_path = video_dir / vid
#     print(vid_path)
#     minio_client.fput_object(bucket_name=bucket_name,
#                        object_name=vid,
#                        file_path=str(vid_path))

# # Logistic Regression
# for file in os.listdir(str(lr_dir)):
#     file_path = lr_dir / file
#     print(file_path)
#     minio_client.fput_object(bucket_name=bucket_name,
#                     object_name=file,
#                     file_path=str(file_path))

# # Speech2Text
# for audio in os.listdir(str(audio_dir)):
#     audio_path = audio_dir / audio
#     print(audio_path)
#     minio_client.fput_object(bucket_name=bucket_name,
#                     object_name=audio,
#                     file_path=str(audio_path))

# # Linpack
# for matrix in os.listdir(str(linpack_dir)):
#     linpack_path = linpack_dir / matrix
#     print(linpack_path)
#     minio_client.fput_object(bucket_name=bucket_name,
#                              object_name=matrix,
#                              file_path=str(linpack_path))


# Float matrix multiply
# for matrix in os.listdir(str(mat_dir)):
#     matrix_path = mat_dir / matrix
#     print(matrix_path)
#     minio_client.fput_object(bucket_name=bucket_name,
#                              object_name=matrix,
#                              file_path=str(matrix_path))

# # Resnet-50
# minio_client.fput_object(bucket_name=bucket_name,
#                          object_name='saved_model.pb',
#                          file_path=str(resnet_dir / 'saved_model.pb'))
# minio_client.fput_object(bucket_name=bucket_name,
#                          object_name='ImageNetLabels.txt',
#                          file_path=str(resnet_dir / 'ImageNetLabels.txt'))
# minio_client.fput_object(bucket_name=bucket_name,
#                          object_name='variables.index',
#                          file_path=str(resnet_dir / 'variables/variables.index'))
# minio_client.fput_object(bucket_name=bucket_name,
#                          object_name='variables.data-00000-of-00001',
#                          file_path=str(resnet_dir / 'variables/variables.data-00000-of-00001'))
for img in os.listdir(str(resnet_image_dir)):
    img_path = resnet_image_dir / img
    print(img_path)
    minio_client.fput_object(bucket_name=bucket_name,
                       object_name=img,
                       file_path=str(img_path))
