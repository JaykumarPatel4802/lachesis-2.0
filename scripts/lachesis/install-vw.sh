#!/bin/bash

# First install cmake 3.20+
sudo apt update
sudo apt install build-essential libtool autoconf unzip wget
sudo apt remove --purge --auto-remove cmake
version=3.26
build=1
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/
./bootstrap
make -j$(nproc)
sudo make install
# Logout and log back in
cmake --version

# Second install ninja
sudo wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
sudo gunzip /usr/local/bin/ninja.gz
sudo chmod a+x /usr/local/bin/ninja
ninja --version

# Third build VW
git clone --recursive https://github.com/VowpalWabbit/vowpal_wabbit.git
cd vowpal_wabbit
cmake --preset=vcpkg-release -DBUILD_TESTING=OFF
cmake --build --preset=vcpkg-release

# Fourth install VW
cmake --install build

# Helpful links: 
# https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line -- installing cmake
# https://lindevs.com/install-ninja-build-system-on-ubuntu -- intalling ninja
