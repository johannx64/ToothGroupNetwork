# Use the official PyTorch image as a base
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Install necessary packages
RUN apt update && \
    apt install -y git && \
    pip install wandb && \
    pip install --ignore-installed PyYAML && \
    pip install open3d && \
    pip install multimethod && \
    pip install termcolor && \
    pip install trimesh && \
    pip install easydict

# Copy the external libraries and install them
COPY external_libs/pointops /workspace/external_libs/pointops
WORKDIR /workspace/external_libs/pointops
RUN python setup.py install

# Set the entry point for the container
CMD ["/bin/bash"]
