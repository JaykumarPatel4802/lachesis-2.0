# !/bin/bash

# Check if parameters were provided
if (( $# < 3 ))
  then
    echo "Usage: ./run-image-process.sh <cpu count> <memory count> <image>"
    exit 1
fi

cd ~/openwhisk-benchmarks/functions/image-processing

CPU=$1
MEM=$2
FILE_IMAGE=$3
INVOKER="129.114.108.74"

# Register image process function with our OpenWhisk setup
echo "Experiment Setup:"
echo "-----------------"
ACTIONS=$(wsk -i action list)
FXN='image-process'
ACTION_OP='create'
if [[ "$ACTIONS" == *"$FXN"* ]]; then
    ACTION_OP='update'
fi
wsk -i action $ACTION_OP $FXN $FXN.py \
    --docker psinha25/main-python \
    --web raw \
    --memory $MEM --cpu $CPU \
    --param endpoint "10.52.2.243:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk" \

# Start up the faas profiler running other 4 functions as noise
# cd ~/faas-profiler
# ./WorkloadInvoker -c ~/openwhisk-benchmarks/runtests/image-process/image-process-config.json &
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
    STATS_PID=$(ssh $INVOKER " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$image-${FXN}.txt & jobs -p")

    # Do 10 runs per image
    for run in {1..10} ; do

        # 1. Get latency of critical section of function
        LAT_LINE=$(wsk action invoke $FXN -i -p image ${image} -r | grep "latency")
        L_LINE=($LAT_LINE)
        LATENCY=${L_LINE[1]}
        echo "latency: $LATENCY"

        # 2. Collect max memory usage
        CONTAINER_LINE=$(ssh $INVOKER " docker ps --no-trunc | grep main-python ")
        C_LINE=($CONTAINER_LINE)
        CONTAINER_ID=${C_LINE[0]}
        MAX_MEM=$(ssh $INVOKER "cat /sys/fs/cgroup/memory/docker/$CONTAINER_ID/memory.max_usage_in_bytes")
        echo "max_mem: $MAX_MEM"
        sleep 5
    done
    ssh $INVOKER " kill $STATS_PID "
done

# Stop all containers created FaasProfiler & the container of interest
ssh $INVOKER " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
# sleep 10
# ssh 129.114.109.238 " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
# sleep 10
# ssh 129.114.109.238 " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
# sleep 10
# ssh 129.114.109.238 " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "