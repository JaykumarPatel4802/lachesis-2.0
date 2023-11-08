import pandas as pd
import os

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

    # Check if file is .txt file - contains our raw data
    filename = file.split(".")
    if len(filename) > 1 and filename[1] == "txt":
        path = directory+file

        name = filename[0].split("-")
        curr_benchmark = name[0]
        lim_cpu = int(name[1])
        lim_mem = int(name[2])
        size = int(name[3])

        with open(path) as f:
            lines = f.readlines()
            run = 1
            for line in lines:
                if "latency" in line:
                    lat_line = line.split(" ")
                    exec_time.append(float(lat_line[1][:-1]))
                elif "max_mem" in line:
                    mem_line = line.split(" ")
                    mem_bytes = int(mem_line[1][:-1])
                    mem_megabytes = mem_bytes / (2**20)
                    mem_usage.append(mem_megabytes)
                    mem_util.append((mem_megabytes / 2048) * 100)
                    benchmark.append(curr_benchmark)
                    cpu_lim.append(lim_cpu)
                    mem_lim.append(lim_mem)
                    data_size.append(size)
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

df.to_csv("./linpack-data-no-load.csv", index=False)