import pandas as pd
import os
from statistics import mean
import operator as op

directory = "./"
benchmark = []
cpu_lim = []
mem_lim = []
data_size = []
run_no = []
exec_time = []
mem_usage = []
mem_util = []
for file in os.listdir(directory):
# for file in ["./sentiment-96-2048-1500.txt", "./sentiment-64-2048-2000.txt", "./sentiment-80-2048-50.txt", "./sentiment-80-2048-1250.txt"]:
# for file in ["./sentiment-80-2048-1250.txt"]:
    # Check if file is .txt file - contains our raw data
    if file[-3:] == "txt":
        path = directory+file
        trunc_name = file[0:-4]
        name = trunc_name.split("-")

        with open(path) as f:
            lines = f.readlines()
            run = 1
            for line in lines:
                if "latency:" in line:
                    lat_line = line.split(" ")
                    if (lat_line[1] != "\n"):
                        exec_time.append(float(lat_line[1][:-1]))
                elif "max_mem" in line:
                    if len(mem_usage) + 1 == len(exec_time):
                        mem_line = line.split(" ")
                        mem_bytes = int(mem_line[1][:-1])
                        mem_megabytes = mem_bytes / (2**20)
                        mem_usage.append(mem_megabytes)
                        mem_util.append((mem_megabytes / 2048) * 100)
                        benchmark.append("sentiment")
                        cpu_lim.append(int(name[1]))
                        mem_lim.append(int(name[2]))
                        data_size.append(int(name[3]))
                        run_no.append(run)
                        run += 1


df = pd.DataFrame(columns=['benchmark', 'cpu_lim', 'mem_lim', 'data_size', 'run_no', 'exec_time', 'mem_usage', 'mem_util'])
df['benchmark'] = benchmark
df['cpu_lim'] = cpu_lim
df['mem_lim'] = mem_lim
df['data_size'] = data_size
df['run_no'] = run_no
df['exec_time'] = exec_time
df['mem_usage'] = mem_usage
df['mem_util'] = mem_util
print(df.to_string())
print(len(df))
df.to_csv("sentiment-data-no-load.csv", index=False)
