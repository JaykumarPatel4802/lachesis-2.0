import pvleopard
from minio import Minio
from time import time

leopard = pvleopard.create(access_key='r6266cN4PB1uNIcdIT5zd6Ur3K6TOopa+K5JcsdpZ5i/GV7YykwtTQ==')

def transcribe(audio_path):
    start = time()
    transcript, words = leopard.process_file(audio_path)
    print(transcript)
    for word in words:
        print(
        "{word=\"%s\" start_sec=%.2f end_sec=%.2f confidence=%.2f}"
        % (word.word, word.start_sec, word.end_sec, word.confidence))
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

    audio_name = params['input1']
    audio_path = '/tmp/' + audio_name

    minio_client.fget_object(bucket_name=bucket,
                       object_name=audio_name,
                       file_path=audio_path)

    latency = transcribe(audio_path)
    ret_val = {}
    ret_val["latency"] = latency
    return ret_val

# find /home/cc/openwhisk-benchmarks/functions/speech2text/LibriSpeech/ -printf '%s %p\n'| sort -nr | head -20
# https://medium.com/picovoice/transcribe-speech-to-text-with-python-for-free-321b063b213c
# https://console.picovoice.ai/