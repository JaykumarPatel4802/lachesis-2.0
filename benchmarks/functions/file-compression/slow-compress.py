# SuperFastPython.com
# create a zip file and add files sequentially
from os import listdir
from os.path import join
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED
from time import time
 
# create a zip file
def main(path='/home/cc/openwhisk-benchmarks/minio-data/file-compression'):
    start = time()
    # list all files to add to the zip
    files = [join(path,f) for f in listdir(path)]
    # open the zip file
    with ZipFile('slow-testing.zip', 'w', compression=ZIP_DEFLATED) as handle:
        # add all files to the zip
        for filepath in files:
            # add file to the archive
            handle.write(filepath)
            # report progress
            print(f'.added {filepath}')
    latency = time() - start
    print(f'latemcy: {latency}')
 
# entry point
if __name__ == '__main__':
    main()