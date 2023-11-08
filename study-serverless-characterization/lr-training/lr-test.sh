# !/bin/bash

# CPU: 16, Memory: 2GB
./run-lr.sh 16 2048 20M-reviews.csv >> lr-16-2048-20M.txt
./run-lr.sh 16 2048 10M-reviews.csv >> lr-16-2048-10M.txt

# CPU: 32, Memory: 2GB
./run-lr.sh 32 2048 20M-reviews.csv >> lr-32-2048-20M.txt
./run-lr.sh 32 2048 10M-reviews.csv >> lr-32-2048-10M.txt

# CPU: 48, Memory: 2GB
./run-lr.sh 48 2048 20M-reviews.csv >> lr-48-2048-20M.txt
./run-lr.sh 48 2048 10M-reviews.csv >> lr-48-2048-10M.txt

# CPU: 64, Memory: 2GB
./run-lr.sh 64 2048 20M-reviews.csv >> lr-64-2048-20M.txt
./run-lr.sh 64 2048 10M-reviews.csv >> lr-64-2048-10M.txt

# CPU: 80, Memory: 2GB
./run-lr.sh 80 2048 20M-reviews.csv >> lr-80-2048-20M.txt
./run-lr.sh 80 2048 10M-reviews.csv >> lr-80-2048-10M.txt

# CPU: 96, Memory: 2GB
./run-lr.sh 96 2048 20M-reviews.csv >> lr-96-2048-20M.txt
./run-lr.sh 96 2048 10M-reviews.csv >> lr-96-2048-10M.txt

