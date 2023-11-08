# Obtained most functionality from https://stackoverflow.com/questions/59082896/how-to-calculate-cpu-utilization-of-container-in-docker-using-http-api
import docker

# These initial values will seed the "last" cycle's saved values
containerCPU = 0
systemCPU = 0
client = docker.from_env()
function = "linpack"
container = ""
while container == "":
    running_containers = client.containers.list()
    for tainer in running_containers:
        if function in tainer.name:
            container = tainer

# container = client.containers.get(containerID)

print(container.id)
#This function is blocking; the loop will proceed when there's a new update to iterate
for stats in container.stats(decode=True):
    #Save the values from the last sample
    lastContainerCPU = containerCPU
    lastSystemCPU = systemCPU

    #Get the container's usage, the total system capacity, and the number of CPUs
    #The math returns a Linux-style %util, where 100.0 = 1 CPU core fully used
    containerCPU = stats.get('cpu_stats',{}).get('cpu_usage',{}).get('total_usage')
    systemCPU    = stats.get('cpu_stats',{}).get('system_cpu_usage')
    numCPU   = len(stats.get('cpu_stats',{}).get('cpu_usage',{}).get('percpu_usage',0))

    # Skip the first sample (result will be wrong because the saved values are 0)
    if lastContainerCPU and lastSystemCPU:
        cpuUtil = (containerCPU - lastContainerCPU) / (systemCPU - lastSystemCPU)
        cpuUtil = cpuUtil * numCPU * 100 / 2
        print(cpuUtil)
