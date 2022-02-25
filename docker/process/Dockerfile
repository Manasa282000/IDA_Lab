FROM ubuntu:20.04

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
                           python3 python3-distutils python3-pandas \
                           python3-numpy python3-pandas  \
                           python3-matplotlib python3-sortedcontainers python3-gmpy2 \
                           texlive texlive-pictures latexmk git gzip

# adding the gosdt user
RUN useradd -m gosdt
