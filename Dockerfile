# Start from an image with CUDA and pytorch
# https://hub.docker.com/r/ceshine/cuda-pytorch/dockerfile/
FROM ceshine/cuda-pytorch:1.7.1

# Install dependencies
RUN pip install tensorboardX pyyaml

# Copy code and run evaluation
COPY . /pq-sgd
WORKDIR /pq-sgd

# RUN python -m src.train_resnet --config ../config/train_resnet50.yaml
# RUN python -m src.evaluate_resnet
