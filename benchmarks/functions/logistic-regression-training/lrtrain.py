from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np 

from time import time
import re
from minio import Minio

cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence

def train(file_path):
    df = pd.read_csv(file_path)
    start = time()
    df['train'] = df['Text'].apply(cleanup)

    tfidf_vector = TfidfVectorizer(min_df=100).fit(df['train'])

    train = tfidf_vector.transform(df['train'])

    model = LogisticRegressionCV(n_jobs=-1)
    model.fit(train, df['Score'])
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

    file_name = params['input1']
    file_path = '/tmp/' + file_name

    minio_client.fget_object(bucket_name=bucket,
                       object_name=file_name,
                       file_path=file_path)
    
    latency = train(file_path)

    ret_val = {}
    ret_val['latency'] = latency
    return ret_val