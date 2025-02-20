# Start with CUDA 11.8 base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Avoid interactive dialog during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Create symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Verify Python version
RUN python --version

# Set working directory
WORKDIR /app

# COPY pyproject.toml
COPY ./ /app/lerobot
WORKDIR /app/lerobot

RUN pip3 install -e .
RUN pip3 install transformers
RUN pip3 install datasets
RUN pip3 install pytest

# Install PyTorch with CUDA 11.8 support
# RUN pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
# RUN pip3 install \
#     timm==0.9.10 \
#     tokenizers==0.19.1 \
#     transformers==4.40.1

# Set the nvidia runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default command
CMD ["/bin/bash"]