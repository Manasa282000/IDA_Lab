#!/bin/bash

set -euo pipefail

echo "Setting up docker container (DEPLOY)"
echo "==============================================="

# set the debian frontent to be noninteractive
export DEBIAN_FRONTEND=noninteractive

echo "-----------------------------------------------------------------------"
echo "installing dependencies"

echo " + running apt update"
# update the package repositories
apt-get -qq update > /dev/null

echo " + installing build and runtime dependencies"
# build and run dependencies
apt-get -qq install -y \
    make \
    gcc \
    g++ \
    build-essential \
    automake \
    git \
    gzip \
    bash \
    libboost-dev \
    libgmp-dev \
    libtbb-dev > /dev/null


echo " + installing python dependencies"
# python dependencies
apt-get -qq install -y \
    python3 \
    ipython3 \
    python3-distutils \
    python3-dev \
    python3-numpy \
    python3-pandas \
    python3-sklearn \
    python3-pip \
    python3-matplotlib \
    python3-sortedcontainers \
    python3-gmpy2  > /dev/null

echo " + installing pip packages"
# install pip kackages
pip3 install -q Cython > /dev/null
pip3 install corels > /dev/null
pip3 install pyarrow > /dev/null

echo " + cleaning apt"
apt-get autoremove
apt-get autoclean
apt-get clean


# ---------------------------------------------------------------------------
# Building GOSDT
# ---------------------------------------------------------------------------

cd /gosdt-src/gosdt

# make a hard git reset
git reset --hard

echo "-----------------------------------------------------------------------"
echo "building gosdt..."

# make sure we have the autoconf dependencies all set up
echo " + autoreconf"
autoreconf -f -i &> /dev/null

# build it
echo " + configure"
./configure  &> /dev/null

echo " + building gosdt"
make -j `nproc` gosdt  &> /dev/null

# install gosdt
echo " + installing gosdt"
make install  &> /dev/null

echo "gosdt installed"

echo "-----------------------------------------------------------------------"
echo "building pygosdt..."

# XXX: that's not so nice here, but otherwise building on a more recente machine
#      fails due to "unknown arch"
export GOSDT_BUILD_OPT_FLAGS="-O3;-march=broadwell"

echo " + cleaning directory"
# clean the directory
make clean &> /dev/null
python3 setup.py clean  &> /dev/null

# build it in a machine-local directory
echo " + build pygosdt module"
python3 setup.py build -q -j `nproc` &> /dev/null

# install the gosdt python module in the mounted home directory
echo " + installing python module"
python3 setup.py install --skip-build  &> /dev/null

echo "pygosdt installed"

# ---------------------------------------------------------------------------
# Building DL8.5
# ---------------------------------------------------------------------------

echo "-----------------------------------------------------------------------"
echo "building pydl8.5..."

# go into the bind-mounted directory
cd /gosdt-src/dl85

# clean the directory
echo " + cleaning directory"
python3 setup.py clean  &> /dev/null

echo " + building pydl85 module"
python3 setup.py build -q -j `nproc` &> /dev/null

# install the dl85 python module in the home directory
echo " + installig pydl85"
python3 setup.py install --skip-build  &> /dev/null

echo "pydl85 installed"

echo "-----------------------------------------------------------------------"
echo "cleanup dl85 build"

# clean the directory again
python3 setup.py clean

# make a hard git reset
git reset --hard

# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

echo "-----------------------------------------------------------------------"
echo "finalize"

# go to the root directory
cd /

# create the gosdt directory
mkdir -p /gosdt

# copy the directories
echo "copy to /gosdt"
for d in datasets; do
    mv /gosdt-src/$d /gosdt/$d
done

# we can't have netherlands in the public docker image.
rm -rf /gosdt/datasets/original_datasets/netherlands*
rm -rf /gosdt/datasets/binarized_datasets/netherlands*

# go to the root
echo "remove /gosdt-src"

# remove the gosdt src
rm -rf /gosdt-src

# set the entry point executable
chmod 755 /entrypoint.sh

echo "-----------------------------------------------------------------------"
echo "done"
