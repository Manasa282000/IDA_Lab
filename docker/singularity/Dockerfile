FROM ubuntu:20.04

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
                                    build-essential \
                                    libssl-dev \
                                    uuid-dev \
                                    libgpgme11-dev \
                                    squashfs-tools \
                                    libseccomp-dev \
                                    wget \
                                    pkg-config \
                                    git \
                                    cryptsetup \
                                    golang-go \
                                    debootstrap


RUN wget https://github.com/sylabs/singularity/releases/download/v3.8.0/singularity-ce-3.8.0.tar.gz && \
    tar -xzf singularity-ce-3.8.0.tar.gz

RUN (cd singularity-ce-3.8.0 && ./mconfig && \
    make -C builddir && \
    make -C builddir install)

RUN rm -rf ./singularity-ce-3.8.0* && singularity help

# adding the gosdt user
RUN useradd -m gosdt
