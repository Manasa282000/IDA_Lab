FROM ubuntu:20.04

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y git \
                                                          make \
                                                          gcc \
                                                          g++ \
                                                          build-essential \
                                                          automake \
                                                          gzip \
                                                          lshw \
                                                          python3 \
                                                          python3-dev \
                                                          python3-pip \
                                                          python3-distutils \
                                                          python3-numpy \
                                                          python3-pandas \
                                                          python3-sklearn \
                                                          python3-matplotlib \
                                                          python3-sortedcontainers \
                                                          python3-gmpy2 \
                                                          libboost-dev \
                                                          libgmp-dev \
                                                          libtbb-dev \
                                                          ocl-icd-opencl-dev && \
    pip3 install -q Cython && \
    pip3 install -q dl8.5 && \
    pip3 install -q corels && \
    pip3 install -q pyarrow

# adding the gosdt user
RUN useradd -m gosdt
