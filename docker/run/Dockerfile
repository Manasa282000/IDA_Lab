FROM ubuntu:20.04

# Avoid interactive prompts during builds
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
        git \
        build-essential \
        gcc \
        g++ \
        python3 \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-pandas \
        python3-sklearn \
        python3-matplotlib \
        python3-gmpy2 \
        libgmp-dev \
        libboost-dev \
        dos2unix \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Install pip dependencies
RUN pip3 install --no-cache-dir -q Cython==0.29.36
RUN pip3 install --no-cache-dir -q corels pyarrow sortedcontainers

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Fix CRLF issues in scripts (if needed)
#RUN dos2unix gosdt/dependencies.sh scripts/compile_and_install.sh

# Run compile and install script (assumes dl8.5 build is done here)
RUN bash ./scripts/compile_and_install.sh

# Default command (optional)
# CMD ["python3", "your_script.py"]
