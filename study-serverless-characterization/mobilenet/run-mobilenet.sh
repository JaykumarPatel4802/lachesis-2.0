# !/bin/bash

# Check if parameters were provided
if (( $# < 3 ))
  then
    echo "Usage: ./run-mobilenet.sh <cpu count> <memory count> <image>"
    exit 1
fi

cd ~/openwhisk-benchmarks/functions/mobilenet

CPU=$1
MEM=$2
FILE_IMAGE=$3
INVOKER="129.114.108.74"

# Register matrix multiplication function with our OpenWhisk setup
echo "Experiment Setup:"
echo "-----------------"
ACTIONS=$(wsk -i action list)
FXN='mobilenet'
ACTION_OP='create'
if [[ "$ACTIONS" == *"$FXN"* ]]; then
    ACTION_OP='update'
fi
wsk -i action $ACTION_OP $FXN $FXN.py --docker psinha25/main-python --web raw --memory $MEM --cpu $CPU
wsk -i action $ACTION_OP $FXN $FXN.py \
    --docker psinha25/mobilenet-ow \
    --web raw \
    --memory $MEM --cpu $CPU \
    --param endpoint "10.52.2.243:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk" \


# Start up the faas profiler running other 4 functions as noise
# cd ~/faas-profiler
# ./WorkloadInvoker -c ~/openwhisk-benchmarks/runtests/mobilenet/mobilenet-config.json &
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
for image in $FILE_IMAGE; do 
    echo "Image: $image"
    # ssh 129.114.109.238 " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$MAT_SIZE-${FXN}.txt " &
    STATS_PID=$(ssh $INVOKER " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$FILE_IMAGE-${FXN}.txt & jobs -p")

    # Do 10 runs per matrix
    for run in {1..10} ; do

        # 1. Get latency of critical section of function
        LAT_LINE=$(wsk action invoke ${FXN} -i -p image ${image} -r | grep "latency")
        L_LINE=($LAT_LINE)
        LATENCY=${L_LINE[1]}
        echo "latency: $LATENCY"

        # 2. Collect max memory usage
        CONTAINER_LINE=$(ssh $INVOKER " docker ps --no-trunc | grep mobilenet-ow ")
        C_LINE=($CONTAINER_LINE)
        CONTAINER_ID=${C_LINE[0]}
        MAX_MEM=$(ssh $INVOKER "cat /sys/fs/cgroup/memory/docker/$CONTAINER_ID/memory.max_usage_in_bytes")
        echo "max_mem: $MAX_MEM"
        sleep 5
    done
    # STATS_PID=$(ssh $INVOKER " ps -aux | grep \"python3 ~/stats-collector.py\" | cut -d ' ' -f8 ")
    ssh $INVOKER " kill $STATS_PID "
done

# Stop all container created by FaasProfiler
ssh $INVOKER " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
