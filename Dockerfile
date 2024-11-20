# Start from a base CUDA image with NVCC 11.7
# FROM nvidia/cuda:11.7.1-devel-ubuntu18.04
FROM sameli/manylinux2014_x86_64_cuda_11.8

# RUN apt update && apt upgrade -y
# RUN apt install software-properties-common -y
# RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt install -y python3.10
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 

COPY ../ /app

WORKDIR /app

RUN python3.10 -m pip install .

# Default command
CMD ["/bin/bash"]