# !/bin/bash

# Check if parameters were provided
if (( $# < 3 ))
  then
    echo "Usage: ./run-linpack.sh <cpu count> <memory count> <matrix size>"
    exit 1
fi

cd ~/openwhisk-benchmarks/functions/linpack

CPU=$1
MEM=$2
MAT_SIZE=$3
MACHINE="129.114.108.74"

# Register matrix multiplication function with our OpenWhisk setup
echo "Experiment Setup:"
echo "-----------------"
ACTIONS=$(wsk -i action list)
FXN='linpack'
ACTION_OP='create'
if [[ "$ACTIONS" == *"$FXN"* ]]; then
    ACTION_OP='update'
fi
wsk -i action $ACTION_OP $FXN $FXN.py --docker psinha25/main-python --web raw --memory $MEM --cpu $CPU

# Start up the faas profiler running other 4 functions as noise
# cd ~/faas-profiler
# ./WorkloadInvoker -c ~/openwhisk-benchmarks/runtests/linpack/linpack-config.json &
# sleep 10
# echo ""

echo "Experiment Details:"
echo "-------------------"
echo "CPU: $CPU"
echo "MEM: $MEM"
echo ""

# Begin experiment
echo "Experiment Begins:"
echo "------------------"
for size in $MAT_SIZE; do 
    echo "MATRIX SIZE: $size"
    # ssh 129.114.109.238 " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$MAT_SIZE-${FXN}.txt " &
    STATS_PID=$(ssh $MACHINE " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$MAT_SIZE-${FXN}.txt & jobs -p")

    # Do 10 runs per matrix
    for run in {1..10} ; do

        # 1. Get latency of critical section of function
        LAT_LINE=$(wsk action invoke linpack -i -p N ${size} -r | grep "latency")
        L_LINE=($LAT_LINE)
        LATENCY=${L_LINE[1]}
        echo "latency: $LATENCY"

        # 2. Collect max memory usage
        CONTAINER_LINE=$(ssh $MACHINE " docker ps --no-trunc | grep main-python ")
        C_LINE=($CONTAINER_LINE)
        CONTAINER_ID=${C_LINE[0]}
        MAX_MEM=$(ssh $MACHINE "cat /sys/fs/cgroup/memory/docker/$CONTAINER_ID/memory.max_usage_in_bytes")
        echo "max_mem: $MAX_MEM"
        sleep 5
    done
    # STATS_PID=$(ssh $MACHINE " ps -aux | grep \"python3 ~/stats-collector.py\" | cut -d ' ' -f8 ")
    ssh $MACHINE " kill $STATS_PID "
done

# Stop all container created by FaasProfiler
ssh $MACHINE " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "