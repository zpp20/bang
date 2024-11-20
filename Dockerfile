# Start from a base CUDA image with NVCC 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip  && apt-get clean
RUN pip install python-libsbml

COPY ../ /app

WORKDIR /app

RUN pip install .

# Default command
CMD ["/bin/bash"]