frequency="$1"

num_cores=$(nproc --all)

for ((i = 0; i < "$num_cores"; i++)); do
    # echo "Setting frequency $frequency for core $i"
    # Print the CPU number
    sudo cpufreq-set -c "$i" -g userspace
    sudo cpufreq-set -c "$i" -f "$frequency"
done

echo "Done setting frequency for all cores to $frequency"