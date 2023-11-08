import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    

import json
import numpy as np
import PIL.Image as Image
from time import time
from minio import Minio

import tensorflow as tf
import tensorflow_hub as hub

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
    
    image_name = params['input1']
    image_path = '/tmp/' + image_name

    minio_client.fget_object(bucket_name=bucket,
                       object_name=image_name,
                       file_path=image_path)

    # Model
    model_pb = "saved_model.pb"
    model_pb_path = "/tmp/model/" + model_pb
    minio_client.fget_object(bucket_name=bucket,
                       object_name=model_pb,
                       file_path=model_pb_path)
    
    var1 = "variables.data-00000-of-00001"
    var1_path = "/tmp/model/variables/" + var1
    minio_client.fget_object(bucket_name=bucket,
                       object_name=var1,
                       file_path=var1_path)
    
    var2 = "variables.index"
    var2_path = "/tmp/model/variables/" + var2
    minio_client.fget_object(bucket_name=bucket,
                       object_name=var2,
                       file_path=var2_path)
    
    labels = "ImageNetLabels.txt"
    labels_path = "/tmp/model/" + labels
    minio_client.fget_object(bucket_name=bucket,
                       object_name=labels,
                       file_path=labels_path)

    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    model = tf.keras.Sequential([hub.KerasLayer('/tmp/model/')])
    model.build([None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    imagenet_labels= np.array(open('/tmp/model/ImageNetLabels.txt').read().splitlines())

    image = Image.open(image_path)

    start = time()
    img = np.array(image)/255.0
    prediction = ''
    for i in range(0, 8):
        prediction = model.predict(img[np.newaxis, ...] )
    top5_confidence_index = np.argsort(-prediction[0], axis=-1)[:5]

    for index in top5_confidence_index:
        prediction_confidence = prediction[0][index]
        predicted_class = imagenet_labels[index]
        print('ImageName: {0}, Prediction: {1}, Prediction confidence: {2}'.format(image, predicted_class, prediction_confidence))
    latency = time() - start

    ret_val = {}
    ret_val["latency"] = latency
    return ret_val

if __name__ == "__main__":
    main()