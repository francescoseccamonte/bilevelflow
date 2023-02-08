# Parametrize base image
ARG BASE=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM ${BASE}
# Other tags: https://hub.docker.com/r/pytorch/pytorch/tags

LABEL author="Francesco Seccamonte <fseccamonte@ucsb.edu>"
LABEL maintainer="Francesco Seccamonte <fseccamonte@ucsb.edu>"
LABEL version="0.1"
LABEL description="Dockerfile to build a pytorch-based image with dependencies to run flow estimation codes"

# Upgrade pip
RUN pip3 install --upgrade pip

# Downgrade setuptools https://github.com/pypa/setuptools/issues/3278
RUN pip3 install --upgrade setuptools==58.3.0

# Parametrize cuda version
ARG CUDA=cu113

# Install torch geometric (main torch dependencies already installed in base image)
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html

# Install additional dependencies from requirements.txt file
COPY requirements.txt /pipdeps/
RUN pip3 install --requirement /pipdeps/requirements.txt
