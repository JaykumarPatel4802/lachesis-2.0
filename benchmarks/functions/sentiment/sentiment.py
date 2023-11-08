# Dataset: https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format?resource=download&select=Train.csv
import sys
import json
from textblob import TextBlob
from time import time
from minio import Minio

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
    
    file_name = params['input1']
    path = '/tmp/' + file_name

    minio_client.fget_object(bucket_name=bucket,
                       object_name=file_name,
                       file_path=path)

    
    with open(path) as json_file:
        data = json.load(json_file)

        start = time()
        for sentence in data['analyse']:
            analysis = TextBlob(sentence)
            sentences = len(analysis.sentences)
            subjectivity =  sum([sentence.sentiment.subjectivity for sentence in analysis.sentences]) / sentences
            polarity = sum([sentence.sentiment.subjectivity for sentence in analysis.sentences]) / sentences
        latency = time() - start

    retVal = {}
    retVal["latency"] = latency

    return retVal
