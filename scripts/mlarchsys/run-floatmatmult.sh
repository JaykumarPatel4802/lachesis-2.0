# !/bin/bash

INVOKER="129.114.108.36"
CPU_LEVELS=( 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 )
DENSITY_LEVELS=( 0.9 0.7 0.5 0.3 0.1 )
MATRIX_SIZES=( 1000 )

cd /home/cc/openwhisk-benchmarks/functions/matmult

STATS_PID=$(ssh $INVOKER " nohup ~/getstats.sh >/dev/null 2>/dev/null </dev/null & jobs -p")
echo $STATS_PID
sleep 5

for cpu in "${CPU_LEVELS[@]}"
do
    echo $cpu
    # Register function with CPU and memory limit -- keep memory limit fixed for now
    wsk -i action update floatmatmult floatmatmult.py \
    --docker psinha25/main-python \
    --web raw \
    --memory 4096 --cpu $cpu \
    --param endpoint "10.52.3.37:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"

    for size in "${MATRIX_SIZES[@]}"
    do
        for density in "${DENSITY_LEVELS[@]}"
        do
            wsk action invoke floatmatmult \
                --param m1 matrix1_${size}_${density}.txt \
                --param m2 matrix2_${size}_${density}.txt \
                --param cpu $cpu \
                -r -v -i
        done
    done
done

sleep 240
ssh $INVOKER " kill $STATS_PID "

wsk -i action update floatmatmult floatmatmult.py \
    --docker psinha25/main-python \
    --web raw \
    --memory 4096 --cpu 17 \
    --param endpoint "10.52.3.148:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"

wsk action invoke floatmatmult \
    --param m1 matrix1_4000_0.3.txt \
    --param m2 matrix2_4000_0.3.txt \
    --param cpu 17 \
    -r -v -i

wsk -i action update resnet resnet-50.py \
    --docker psinha25/resnet-50-ow \
    --web raw \
    --memory 5120 --cpu 32 \
    --param endpoint "10.52.3.148:9002" \
    --param access_key "testkey" \
    --param secret_key "testsecret" \
    --param bucket "openwhisk"


wsk -i action invoke resnet \
    --param image 2.4M-building.jpg \
    --param cpu 24 \
    --param no_invocation 0 \
    --param predicted_cores 12 \
    --param slo 10000 \
    -r -v