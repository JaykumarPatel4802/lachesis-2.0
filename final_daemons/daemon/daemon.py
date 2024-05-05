import multiprocessing
import subprocess

def run_aggregator_daemon():
    subprocess.run(["python", "aggregator-daemon.py"])

def run_util_daemon():
    subprocess.run(["python", "util-daemon.py"])

if __name__ == "__main__":
    aggregator_process = multiprocessing.Process(target=run_aggregator_daemon)
    util_process = multiprocessing.Process(target=run_util_daemon)

    aggregator_process.start()
    util_process.start()

    aggregator_process.join()
    util_process.join()