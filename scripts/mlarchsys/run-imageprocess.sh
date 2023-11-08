# !/bin/bash

INVOKER="129.114.108.36"
CPU_LEVELS=( 2 4 6 8 10 )

cd /home/cc/openwhisk-benchmarks/minio-data/image-process
IMAGES=(*)
# IMAGES=( 30M-river_landscape_515440.jpg )
cd /home/cc/openwhisk-benchmarks/functions/image-processing

STATS_PID=$(ssh $INVOKER " nohup ~/getstats.sh >/dev/null 2>/dev/null </dev/null & jobs -p")
echo $STATS_PID
sleep 5

for cpu in "${CPU_LEVELS[@]}"
do
    echo $cpu
    # Register function with CPU and memory limit -- keep memory limit fixed for now
    wsk -i action update imageprocess image-process.py \
    --docker psinha25/main-python \
    --web raw \
    --memory 4096 --cpu $cpu \
    --param endpoint "10.52.3.37:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"

    for image in "${IMAGES[@]}"
    do
        wsk action invoke imageprocess \
            --param image $image \
            --param cpu $cpu \
            -r -v -i
        sleep 5
    done
done

sleep 10
ssh $INVOKER " kill $STATS_PID "

wsk action invoke imageprocess \
    --param image 30M-river_landscape_515440.jpg \
    --param cpu 1 \
    -r -v -i

wsk -i action update imageprocess imageprocess.py \
    --docker psinha25/main-python \
    --web raw \
    --memory 4096 --cpu 1 \
    --param endpoint "10.52.3.148:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"
