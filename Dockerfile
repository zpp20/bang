# Start from a base CUDA image with NVCC 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip && apt-get clean

# Verify the installation
RUN gcc --version
RUN g++ --version
# Optionally, copy your project files here



COPY ../ /app

# Default command
CMD ["/bin/bash"]