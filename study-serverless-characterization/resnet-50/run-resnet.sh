#!/bin/bash

# Check if parameters were provided
if (( $# < 3 ))
  then
    echo "Usage: ./run-resnet.sh <cpu count> <memory count> <image>"
    exit 1
fi

cd ~/openwhisk-benchmarks/functions/resnet-50

CPU=$1
MEM=$2
IMAGE=$3
INVOKER="129.114.108.74"

# Register video process function with our OpenWhisk setup
echo "Experiment Setup:"
echo "-----------------"
ACTIONS=$(wsk -i action list)
FXN='resnet'
ACTION_OP='create'
if [[ "$ACTIONS" == *"$FXN"* ]]; then
    ACTION_OP='update'
fi

wsk -i action $ACTION_OP $FXN $FXN-50.py \
    --docker psinha25/resnet-50-ow \
    --web raw \
    --memory $MEM --cpu $CPU \
    --param endpoint "10.52.2.243:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"

echo "Experiment Details:"
echo "-------------------"
echo "CPU: $CPU"
echo "MEM: $MEM"
echo ""

# Begin experiment
echo "Experiment Begins:"
echo "------------------"
for image in $IMAGE; do
    echo "Image: $image"
    STATS_PID=$(ssh $INVOKER " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$image-${FXN}.txt & jobs -p")

    # Do 10 runs per video
    for run in {1..10}; do

        # 1. Get latency of critical section of function
        LAT_LINE=$(wsk action invoke $FXN -i -p "image" ${image} -r -v --blocking | grep "latency")
        L_LINE=($LAT_LINE)
        LATENCY=${L_LINE[2]}
        echo "latency: $LATENCY"

        # 2. Collect max memory usage
        CONTAINER_LINE=$(ssh $INVOKER " docker ps --no-trunc | grep resnet-50-ow ")
        C_LINE=($CONTAINER_LINE)
        CONTAINER_ID=${C_LINE[0]}
        MAX_MEM=$(ssh $INVOKER "cat /sys/fs/cgroup/memory/docker/$CONTAINER_ID/memory.max_usage_in_bytes")
        echo "max_mem: $MAX_MEM"

        ssh $INVOKER " echo x >> docker-stats-$CPU-$MEM-$image-${FXN}.txt "

        if [ -z "$LATENCY" ]; then
            # ssh $INVOKER " kill $STATS_PID "
            ssh $INVOKER " docker ps | grep psinha25/resnet-50-ow | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
            STATS_PID=$(ssh $INVOKER " python3 -u ~/stats-collector.py >> docker-stats-$CPU-$MEM-$image-${FXN}.txt & jobs -p")
        fi
        sleep 5
    done
    ssh $INVOKER " kill $STATS_PID "
done

# Stop all containers created FaasProfiler & the container of interest
ssh $INVOKER " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
