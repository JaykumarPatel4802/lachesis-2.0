import docker
client = docker.from_env()
 # These initial values will seed the "last" cycle's saved values
# containerCPU = 0
# systemCPU = 0

# container = client.containers.get('817e76b32690')

#   #This function is blocking; the loop will proceed when there's a new update to iterate
# for stats in container.stats(decode=True):

#     #Save the values from the last sample
#   lastContainerCPU = containerCPU
#   lastSystemCPU = systemCPU

#     #Get the container's usage, the total system capacity, and the number of CPUs
#     #The math returns a Linux-style %util, where 100.0 = 1 CPU core fully used
#   containerCPU = stats.get('cpu_stats',{}).get('cpu_usage',{}).get('total_usage')
#   systemCPU    = stats.get('cpu_stats',{}).get('system_cpu_usage')
#   numCPU   = len(stats.get('cpu_stats',{}).get('cpu_usage',{}).get('percpu_usage',0))

#     # Skip the first sample (result will be wrong because the saved values are 0)
#   if lastContainerCPU and lastSystemCPU:
#     cpuUtil = (containerCPU - lastContainerCPU) / (systemCPU - lastSystemCPU)
#     cpuUtil = cpuUtil * numCPU * 100
#     print(cpuUtil)

stats = client.containers.get('817e76b32690').stats(stream=False)
UsageDelta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
# from informations : UsageDelta = 25382985593 - 25382168431

SystemDelta = stats['cpu_stats']['cpu_usage']['system_cpu_usage'] - stats['precpu_stats']['cpu_usage']['system_cpu_usage']
# from informations : SystemDelta = 75406420000000 - 75400410000000

len_cpu = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])
# from my informations : len_cpu = 2


percentage = (UsageDelta / SystemDelta) * len_cpu * 100
# this is a little big because the result is : 0.02719341098169717

percent = round(percentage, 2)
