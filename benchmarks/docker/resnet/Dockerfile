FROM psinha25/mobilenet-ow 

RUN pip install --upgrade tensorflow_hub

RUN mkdir model
RUN curl -L https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4?tf-hub-format=compressed -o ./model/resnet.tar.gz
RUN tar -xf model/resnet.tar.gz -C model/
RUN rm -r model/resnet.tar.gz

# Download ImageNet labels
RUN curl https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt -o ./model/ImageNetLabels.txt
