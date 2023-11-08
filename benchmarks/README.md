# Usage
Before registering functions, make sure the custom docker image is available on Docker Hub. See [this doc](https://github.com/neerajas-group/openwhisk-benchmarks/blob/master/docs/docker-actions.md) for how to create a custom Docker image and use it with OpenWhisk actions.

- matmult
```console
$ wsk action create matmult matmult.py --docker psinha25/python3-ow --web raw --memory 512
$ wsk action invoke matmult -i -p N 100 -r -v
```
- image processing
```console
$ wsk action create image_process image-process.py --docker psinha25/python3-ow --web raw --memory 512
$ wsk action invoke image_process -i -p endpoint "10.52.2.166:9002" -p access_key "testkey" -p secret_key "testsecret" -p bucket "openwhisk" -p image "mj.jpg" -r -v
```
- encryption
```console
$ wsk action create encrypt encrypt.py --docker psinha25/python3-ow --web raw --memory 512 -i
$ wsk action invoke encrypt -i -p length 10 -p iteration 10 -r -v
```

- linpack
```console
$ wsk action create linpack linpack.py --docker psinha25/python3-ow --web raw --memory 512 -i
$ wsk action invoke linpack -i -p N 10 -r -v
```

- mobilenet
```console
$ wsk action create mobilenet mobilenet.py --docker psinha25/mobilenet-ow --web raw --memory 512
$ wsk action invoke mobilenet -i -p endpoint "10.52.2.166:9002" -p access_key "testkey" -p secret_key "testsecret" -p bucket "openwhisk" -p image "mj.jpg" -r -v
```
