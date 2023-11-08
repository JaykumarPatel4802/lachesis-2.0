from os import listdir
from os.path import join
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from time import time
 
# load files as strings then add them to the zip in a thread safe manner
def add_files(lock, handle, filepaths):
    # load files first
    filedata = list()
    for filepath in filepaths:
        # load the data as a string
        with open(filepath, 'r') as file_handle:
            filedata.append(file_handle.read())
    # add all data to zip
    with lock:
        for filepath,data in zip(filepaths, filedata):
            handle.writestr(filepath, data)
            # report progress
            print(f'.added {filepath}')
 
# create a zip file
def main(path='/home/cc/openwhisk-benchmarks/minio-data/file-compression'):
    # list all files to add to the zip
    start = time()
    files = [join(path,f) for f in listdir(path)]
    # files = files[0:100]
    # create lock for adding files to the zip
    lock = Lock()
    # determine chunksize
    n_workers = 96
    chunksize = round(len(files) / n_workers)
    # open the zip file
    with ZipFile('testing.zip', 'w', compression=ZIP_DEFLATED) as handle:
        # create the thread pool
        with ThreadPoolExecutor(n_workers) as exe:
            # split the copy operations into chunks
            for i in range(0, len(files), chunksize):
                # select a chunk of filenames
                filepaths = files[i:(i + chunksize)]
                # submit the task
                _ = exe.submit(add_files, lock, handle, filepaths)
    latency = time() - start
    print(f'latency: {latency}')
 
# entry point
if __name__ == '__main__':
    main()