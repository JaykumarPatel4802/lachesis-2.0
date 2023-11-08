#!/bin/bash

# Check if parameters were provided
if (( $# < 3 ))
  then
    echo "Usage: ./run-sentiment.sh <cpu count> <memory count> <file>"
    exit 1
fi

cd ~/openwhisk-benchmarks/functions/sentiment

CPU=$1
MEM=$2
JSON_FILE=$3
INVOKER="129.114.108.74"

# Register sentiment analysis function with our OpenWhisk setup
echo "Experiment Setup:"
echo "-----------------"
ACTIONS=$(wsk -i action list)
FXN='sentiment'
ACTION_OP='create'
if [[ "$ACTIONS" == *"$FXN"* ]]; then
    ACTION_OP='update'
fi

wsk -i action $ACTION_OP $FXN $FXN.py \
    --docker psinha25/sentiment-ow \
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
for file in $JSON_FILE; do
    echo "File: $file"
    STATS_PID=$(ssh $INVOKER " python3 -u ~/stats-collector.py>> docker-stats-$CPU-$MEM-$file-${FXN}.txt & jobs -p")

    # Do 10 runs per json file
    for run in {1..10}; do

        # 1. Get latency of critical section of function
        LAT_LINE=$(wsk action invoke $FXN -i -p "file" ${file} -r -v --blocking | grep "latency")
        L_LINE=($LAT_LINE)
        LATENCY=${L_LINE[2]}
        echo "latency: $LATENCY"

        # 2. Collect max memory usage
        CONTAINER_LINE=$(ssh $INVOKER " docker ps --no-trunc | grep main-python ")
        C_LINE=($CONTAINER_LINE)
        CONTAINER_ID=${C_LINE[0]}
        MAX_MEM=$(ssh $INVOKER "cat /sys/fs/cgroup/memory/docker/$CONTAINER_ID/memory.max_usage_in_bytes")
        echo "max_mem: $MAX_MEM"

        if [ -z "$LATENCY" ]; then
            # ssh $INVOKER " kill $STATS_PID "
            ssh $INVOKER " docker ps | grep psinha25/sentiment-ow | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "
            STATS_PID=$(ssh $INVOKER " python3 -u ~/stats-collector.py>> docker-stats-$CPU-$MEM-$file-${FXN}.txt & jobs -p")
        fi
        sleep 5
    done
    ssh $INVOKER " kill $STATS_PID "
done

# Stop all containers created FaasProfiler & the container of interest
ssh $INVOKER " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "