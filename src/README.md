### Installing Vowpal Wabbit

```
sudo apt-get update
sudo apt-get install vowpal-wabbit
```

### Using Vowpal Wabbit

```
# Set up daemon with 10 child processes on port 26542

# Memory: number of classes is 160 (each class represents a 64 MB interval)
vw --csoaa 160 --daemon --quiet --port 26542

# CPU: number of classes is 48 (each class represents number of cores)
vw -csoaa 48 --daemon --quiet --port 26543

# View number of vw processes
pgrep vw | wc -l

# Kill the vowpal processes
pkill -9 -f 'vw.*--port 26542'
```

### Installing gRPC

Follow the below steps to install gRCP or just run the `install-grpc.sh` script in `../scripts/lachesis`

```
# Download and Unzip compiler
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
unzip protoc-3.11.4-linux-x86_64.zip -d protoc3

# Move the binary to directory which is PATH
sudo mv protoc3/bin/* /usr/local/bin/

sudo mv protoc3/include/* /usr/local/include/

# Change owner
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google

# Test if it works
protoc --version

# Install grpcio and grpcio-tools for python
pip3 install grpcio grpcio-tools
```

### Compiling gRPC Proto

```
python3 -m grpc_tools.protoc \
    -I /home/$USER/lachesis/src/proto \
    --python_out=/home/$USER/lachesis/src/generated \
    --grpc_python_out=/home/$USER/lachesis/src/generated \
    /home/$USER/lachesis/src/proto/*.proto

sed -i -E 's/^import.*_pb2/from . \0/' /home/$USER/lachesis/src/generated/*.py
```

You might see the warning:
```
/home/cc/.local/lib/python3.8/site-packages/grpc_tools/protoc.py:21: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  import pkg_resources
```
Don't worry about it, the generated files should still be in `~/lachesis/src/generated`

