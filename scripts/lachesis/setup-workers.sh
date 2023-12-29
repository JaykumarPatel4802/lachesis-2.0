#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <0 or 1> <full-lachesis-azure-rps-{}, full-lachesis-alibaba-rps-{}, ..."
    exit 1
fi

# Argument: 0 to perform installations, 1 to launch daemons
ACTION="$1"
SYSTEM="$2"

# Parse the config.yaml file to get cluster configuration data
CONTROLLER_IP=$(grep 'controller_ip' ../../config.yaml | cut -d "'" -f 2)
CONTROLLER_PORT=$(grep 'controller_port' ../../config.yaml | cut -d "'" -f 2)

# worker_ips=($(grep -E 'w[1-16]_ip' ../../config.yaml | cut -d "'" -f 2))
# worker_names=($(grep -E 'w[1-16]_name' ../../config.yaml | cut -d "'" -f 2))
# worker_usernames=($(grep -E 'w[1-16]_username' ../../config.yaml | cut -d "'" -f 2))

worker_ips=($(awk '/w[1-9][0-9]*_ip:/ {print $2}' ../../config.yaml | sed "s/[\"']//g"))
worker_names=($(awk '/w[1-9][0-9]*_name:/ {print $2}' ../../config.yaml | sed "s/[\"']//g"))
worker_usernames=($(awk '/w[1-9][0-9]*_username:/ {print $2}' ../../config.yaml | sed "s/[\"']//g"))

UTIL_PIDS=()
AGG_PIDS=()

declare -a threadpids

# cleanup() {
#     echo 'Received Ctrl+C. Cleaning up...'
#     for ((j = 0; j < ${#UTIL_PIDS[@]}; j++)); do
#         util_pid="${UTIL_PIDS[j]}"
#         agg_pid="${AGG_PIDS[j]}"
#         ip="${worker_ips[j]}"
#         name="${worker_names[j]}"
#         username="${worker_usernames[j]}"

#         # echo "Killing util process $util_pid on $name with ip $ip"
#         # ssh $username@$ip " kill $util_pid "
#         # echo "Killing aggregation process $agg_pid on $name with ip $ip"
#         # ssh $username@$ip " kill $agg_pid "
#         ssh $username@$ip " pkill -f ./util-daemon "
#         ssh $username@$ip " pkill -f aggregator-daemon.py "

#         # Remove containers
#         ssh $username@$ip " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "

#         # At each invoker, create CPU core utilization graph and copy the graph back to central controller
#         ssh $username@$ip " cd  ~/daemon; python3 util-timeline-graph.py --invoker-name $name --system $SYSTEM"
#         scp $username@$ip:~/daemon/*-cpu-util-timeline.png ~/lachesis/study-full-experiments

#     done
#     exit
# }

# Function to copy daemon directory to each worker node
copy_daemon_dir() {
    local ip="$1"
    local name="$2"
    local username="$3"

    # Recompile util-daemon
    gcc ../../src/daemon/util-daemon.c -o ../../src/daemon/util-daemon -lsqlite3

    # Copy daemon directory to each worker node
    scp -r -q ../../src/daemon/ $username@$ip:~
    echo "Copied daemon folder to worker $name"

    # Delete WAL and SHM and DB
    ssh $username@$ip " cd ~/daemon; rm invoker_data.*"
}

# Function to set up worker node (installations)
setup_installations() {
    local ip="$1"
    local name="$2"
    local username="$3"

    echo "Setting up worker $name with ip $ip for installations"
    echo "-------------------------------------------------------"

    # Copy daemon directory to each worker node
    scp -r -q ../../src/daemon/ $username@$ip:~
    echo "Copied daemon folder"

    # Install Docker on Linux node
    ssh $username@$ip " ./daemon/install-docker.sh "

    # Install other OW packages on Linux node
    ssh $username@$ip " ./daemon/install-wsk-packages.sh "

    # Install necessary packages on each worker node
    ssh $username@$ip " pip3 -q install -r ./daemon/requirements.txt "

    echo "Installed python packages"

    echo "Completed installations on worker $name with ip $ip"
}

# Function to launch daemons
launch_daemons() {
    local ip="$1"
    local name="$2"
    local username="$3"

    # Set up local database on each worker node
    ssh $username@$ip " cd ~/daemon; python3 datastore-daemon.py "
    echo "Set up local database on $name"

    # Launch utilization daemon and aggregation daemon
    UTIL_PID=$(ssh $username@$ip " cd ~/daemon; ./util-daemon >/dev/null 2>&1 & jobs -p ")
    AGG_PID=$(ssh $username@$ip " cd ~/daemon; python3 aggregator-daemon.py --controller-ip $CONTROLLER_IP --controller-port $CONTROLLER_PORT --invoker-ip $ip --invoker-name $name>/dev/null 2>&1 & jobs -p ")
    # python3 aggregator-daemon.py --controller-ip '10.52.3.142' --controller-port '50051' --invoker-ip '129.114.109.153' --invoker-name 'w8'
    # python3 aggregator-daemon.py --controller-ip '10.52.3.142' --controller-port '50051' --invoker-ip '129.114.108.59' --invoker-name 'w7' --system 'cypress'
    UTIL_PIDS+=("$UTIL_PID")
    AGG_PIDS+=("$AGG_PID")
    echo "Launched util and aggregator daemon on $name"
    echo ""
}

# Cleanup function to remove daemon processes, kill docker containers created, and plot core utilization plot
cleanup() {
    local ip="$1"
    local name="$2"
    local username="$3"

    # Kill daemon processes
    ssh $username@$ip " pkill -f ./util-daemon "
    ssh $username@$ip " pkill -f aggregator-daemon.py "
    
    # Remove containers that we might've created
    ssh $username@$ip " docker ps | grep psinha25 | cut -d ' ' -f1 | xargs -I {} docker stop -t 1 {} "

    # At each invoker, create CPU core utilization graph and copy the graph back to central controller
    ssh $username@$ip " cd ~/daemon; python3 util-timeline-graph.py --invoker-name $name --system $SYSTEM"
    scp $username@$ip:~/daemon/*-cpu-util-timeline.png ~/lachesis/study-full-experiments
    ssh $username@$ip " cd ~/daemon; rm *-cpu-util-timeline.png "
}

if [ "$ACTION" -eq 0 ]; then
    # Copy daemon directory to each worker node
    for ((i = 0; i < ${#worker_ips[@]}; i++)); do
        ip="${worker_ips[i]}"
        name="${worker_names[i]}"
        username="${worker_usernames[i]}"

        copy_daemon_dir "$ip" "$name" "$username" &
        threadpids+=($!)
    done

    for pid in "${threadpids[@]}"; do
        wait "$pid"
    done

elif [ "$ACTION" -eq 1 ]; then
    # Perform installations on each worker node using parallel threads
    for ((i = 0; i < ${#worker_ips[@]}; i++)); do
        ip="${worker_ips[i]}"
        name="${worker_names[i]}"
        username="${worker_usernames[i]}"

        setup_installations "$ip" "$name" "$username"
    done

    # Exit the script after installations
    exit 0
elif [ "$ACTION" -eq 2 ]; then
    for ((i = 0; i < ${#worker_ips[@]}; i++)); do
        ip="${worker_ips[i]}"
        name="${worker_names[i]}"
        username="${worker_usernames[i]}"

        launch_daemons "$ip" "$name" "$username" &
        threadpids+=($!)
    done

    for pid in "${threadpids[@]}"; do
        wait "$pid"
    done

    echo "Completed launching worker node daemons"
elif [ "$ACTION" -eq 3 ]; then
    for ((i = 0; i < ${#worker_ips[@]}; i++)); do
        ip="${worker_ips[i]}"
        name="${worker_names[i]}"
        username="${worker_usernames[i]}"

        cleanup "$ip" "$name" "$username" &
        threadpids+=($!)
    done

    for pid in "${threadpids[@]}"; do
        wait "$pid"
    done

    echo "Completed cleaning up daemons"
else
    echo "Invalid argument. Use 0 to copy daemon directory, 1 to perform installations, or 2 to launch daemons."
    exit 1
fi