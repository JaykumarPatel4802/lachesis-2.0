#!/bin/bash

worker_ips=($(grep 'w[0-9]_ip' /home/$USER/lachesis/config.yaml | cut -d "'" -f 2))
worker_names=($(grep 'w[0-9]_name' /home/$USER/lachesis/config.yaml | cut -d "'" -f 2))
worker_usernames=($(grep 'w[0-9]_username' /home/$USER/lachesis/config.yaml | cut -d "'" -f 2))

for ((i = 0; i < ${#worker_ips[@]}; i++)); do
    ip="${worker_ips[i]}"
    name="${worker_names[i]}"
    username="${worker_usernames[i]}"

    scp $username@$ip:~/daemon/*.db /home/$USER/lachesis/src/tmp-daemon-data/$name.db
done