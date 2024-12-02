# Start from a base CUDA image with NVCC 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# FROM sameli/manylinux2014_x86_64_cuda_12.3 

RUN apt update
RUN apt install -y python3 python3-pip ninja-build

# RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# RUN apt install software-properties-common curl -y
# RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt update
# RUN apt install -y python3.10
# RUN /usr/bin/python3.10 --version
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.10

COPY ../ /app

WORKDIR /app

RUN python3 -m pip install .

# Default command
CMD ["/bin/bash"]