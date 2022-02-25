FROM ubuntu:20.04

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
                           make gcc g++ build-essential automake git \
                           python3 python3-distutils python3-dev \
                           libboost-dev libgmp-dev libtbb-dev ocl-icd-opencl-dev

# adding the gosdt user
RUN useradd -m gosdt
