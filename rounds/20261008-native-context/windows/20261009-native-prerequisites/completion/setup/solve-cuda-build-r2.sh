#!/bin/bash
export CONDA_PKGS_DIRS=/opt/4d-native-package-cache
/opt/conda/bin/conda create --dry-run --json --override-channels -p /opt/4d-cuda121-build -c https://conda.anaconda.org/nvidia cuda-cudart=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1 > /root/rivermind-data/4d-native-prerequisites-20261009/cuda-build-solve-r2.json 2> /root/rivermind-data/4d-native-prerequisites-20261009/cuda-build-solve-r2.stderr
printf '%s\n' "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/cuda-build-solve-r2.exit
