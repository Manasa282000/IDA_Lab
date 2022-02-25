# Tree Benchmarks

Benchmarks used to compare different learning algorithms for and their tree classifiers

## Paper

This repository was used to evaluate the gosdt-guesses algorithms as presented in

- McTavish, H., Zhong, C., Achermann, R., Karimalis, I., Chen, J., Rudin, C., & Seltzer, M. Fast Sparse Decision Tree Optimization via Reference Ensembles. In AAAI. 2022.

The [paper](paper) directory contains the corresponding plotting scripts for the graphs in the
paper.

## Experiment Configurations

See the `config-aaaai-paper.csv` file, and the [configurations](configurations) directory.

## Datasets

The datasets including scripts for binarization can be found in the [datasets](datasets)
directory.



## Submodules: gosdt and dl8.5

The directory `gosdt` contains a git submodule of the version of gosdt that is intended to be
run with this version of `tree-benchmark`. Similarly, the directory `dl8.5` contains the version
of dl8.5.

```
git submodule init
git submodule update
```

To compile and install the used version execute `bash ./scripts/compile_and_install.sh`
This will perform the submodule initialization and compiles gosdt and dl8.5

## Dependencies

The following dependencies are needed to run on Ubuntu 20.04.
```
$ sudo apt-get install git \
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
                       ocl-icd-opencl-dev

pip3 install --system  -q Cython
pip3 install --system  -q dl8.5 corels
pip3 install --system  -q pyarrow
```

We provide docker and singularity images for running the benchmarks. See the `docker/run` directory.


## Running on a Slurm Cluster

The scripts provide an environment to run the benchmarks in a
[Slurm](https://slurm.schedmd.com/overview.html) environment.

For Compute Canada, you will need to clone the repository somewhere in the `/project` directory,
as you can't submit jobs from your home directory.

**Init Repository:** initialize the repository, with the submodules etc.
```
git submodule init
git submodule update
```

**Prepare Run:** To build the python modules, and create the dataset in the Apache Feather format,
run the following job.

```
bash scripts/prepare-cc-start.sh  -r <RESULTS_DIRECTORY>
```

This will create a home directory in `<RESULTS_DIRECTORY>/gosdt-homedir` that contains the installed
python modules.

**Enqueue Experiment Jobs:**
To enqueue jobs, specify the configuration CSV and the `<RESULTS_DIRECTORY>` directory that you have
prepared using the command above:

```
bash scripts/singularity-start.sh -c <CONFIG.csv> -r <RESULTS_DIRECTORY>
```

Sub directories of interest:
 * `<RESULTS_DIRECTORY>/slurmlogs:` contains the runlogs indicating errors etc.
 * `<RESULTS_DIRECTORY>/configmap:` contains a map from config id (row in CSV) to hash
 * `<RESULTS_DIRECTORY>/runlogs:` results and computed trees.

**Enqueue Single Experiment Job:**
To enqueue a specific configuration in a CSV you can execute the following command:
```
bash scripts/singularity-start-single.sh -c <CONFIG.csv> -r <RESULTS_DIRECTORY> -i <ID>
```
Where `<ID>` is the configuration index. This is the row offset into the CSV file, e.g.,
line `<ID> + 2` (including the header), or dataset row `<ID> + 1`. Example: `i = 2` is the 3rd row
in the CSV, or the 4th line of the file.

**Collecting Results:**
To process and check the results for problems execute the finish script on the same results directory.
You may want to `rsync` the results first.
```
bash scripts/finish-cc-exec.sh <RESULTS_DIRECTORY>
```

## Manually Running Experiments

To run a configuration from the CSV file use the command:

```
python3 python/run.py csv ${CFG_FILE} ${CFG_INDEX}
```
