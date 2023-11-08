# Docker Actions
This repo of OpenWhisk serverless uses a custom Docker image as the action runtime. We have the OpenWhisk platform dynamically inject the source files into the custom runtime during initialization.

## Creating Custom Docker Image & Register in Docker Hub

We build our own custom Docker image for our various python applications (i.e., `matmul`, `image-process`). The Dockerfile and the libraries to bake into the image are located in `python3-openwhisk-docker`. Below is the series of commands we use to build the image and register it in Docker Hub. 

- Build the Docker image (in `python3-openwhisk-docker`)
```console
$ docker build -t python3-ow .
```
- Tag the Docker image with Docker Hub Username
```console
$ docker tag python3-ow psinha25/python3-ow
```
- Push the Docker image to Docker Hub
```console
$ docker push psinha25/python3-ow
```

Note that you might run into a the following issue upon trying to push:
```console
$ docker push psinha25/python3-ow
Using default tag: latest
The push refers to repository [docker.io/psinha25/python3-ow]
70a9fa58e57a: Preparing 
27518f2f9994: Preparing 
b934e3166bfe: Preparing 
a432518212dc: Preparing 
f7679aa2eeb6: Preparing 
237f71f33e46: Waiting 
a030eff2328e: Waiting 
5ac01c082b52: Waiting 
52e609561bfc: Waiting 
d7366bbef1d9: Waiting 
d7aa66fec7c0: Waiting 
2e517d68c391: Waiting 
5f3a5adb8e97: Waiting 
73bfa217d66f: Waiting 
91ecdd7165d3: Waiting 
e4b20fcc48f4: Waiting 
denied: requested access to the resource is denied
```
You need to login on the command line:
```console
$ docker login -u "psinha25" --password-stdin "<password>" docker.io
```
Then retry pushing and it should work.

## Using Docker Images with OpenWhisk Actions

Now that we have our custom Docker image on the public Docker Hub, we can use it as our runtime when registering the action with OpenWhisk.

```console
$ wsk action create matmult matmult.py --docker psinha25/python3-ow --web raw --memory 512
ok: created action matmult
$ wsk action invoke matmult -i -p N 10 -r -v
```

## Resources
We obtained all this information from the official [OpenWhisk Github Docs](https://github.com/apache/openwhisk/blob/master/docs/actions-docker.md).