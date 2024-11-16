# Start from a base CUDA image with NVCC 11.7
# FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
FROM sameli/manylinux2014_x86_64_cuda_11.7

# RUN yum install  python3 python3-pip

COPY ../ /app

WORKDIR /app

RUN python3.10 -m pip install .

# Default command
CMD ["/bin/bash"]