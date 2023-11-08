#!/bin/bash

CONTAINER_NAMES=("psinha25-python" "psinha25-resnet" "psinha25-mobilenet" "psinha25-redis" "psinha25-nginx" "psinha25-mysql" "psinha25-prometheus")
# CONTAINER_NAMES=("psinha25-python"  )
IMAGES=("psinha25/python3-ow" "psinha25/resnet-50-ow" "psinha25/mobilenet-ow" "redis:latest" "nginx:latest" "mysql:latest" "prom/prometheus:latest")
CPU_LEVELS=(2 4 8 16 32 48)
# CPU_LEVELS=(32)
MEMORY_LEVELS=(512 1024 2048 4096)
# MEMORY_LEVELS=("2048")

for ((i = 0; i < ${#CONTAINER_NAMES[@]}; i++)); do
    container_name="${CONTAINER_NAMES[i]}"
    image_name="${IMAGES[i]}"

    for cpu in "${CPU_LEVELS[@]}"; do
        for mem in "${MEMORY_LEVELS[@]}"; do
            
            pwr_output=$(likwid-powermeter docker run -d --name "$container_name" --cpus $cpu --memory "$mem"m "$image_name")
            # Stop and remove container
            docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {}
            docker ps -a | grep psinha25- | cut -d ' ' -f1 | xargs -I {} docker rm {}

            # Extract runtime value (10.1151s)
            runtime=$(echo "$pwr_output" | grep -oE 'Runtime: [0-9.]+ s' | awk '{print $2}')

            # Extract energy and power consumed for Domain PKG on each socket
            energy_socket_0=$(echo "$pwr_output" | grep -A 4 'Measure for socket 0 on CPU 0' | grep 'Energy consumed:' | awk '{print $3}')
            power_socket_0=$(echo "$pwr_output" | grep -A 4 'Measure for socket 0 on CPU 0' | grep 'Power consumed:' | awk '{print $3}')

            energy_socket_1=$(echo "$pwr_output" | grep -A 4 'Measure for socket 1 on CPU 1' | grep 'Energy consumed:' | awk '{print $3}')
            power_socket_1=$(echo "$pwr_output" | grep -A 4 'Measure for socket 1 on CPU 1' | grep 'Power consumed:' | awk '{print $3}')

            echo "$container_name,$cpu,$mem,$runtime,$energy_socket_0,$power_socket_0,$energy_socket_1,$power_socket_1" >> ../../data/energy-study/energy-container-creation.csv
        done
    done
done




