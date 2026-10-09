#!/bin/bash
export CONDA_PKGS_DIRS=/opt/4d-native-package-cache
/opt/conda/bin/conda create --yes --override-channels -c https://conda.anaconda.org/nvidia -p /opt/4d-cuda121-build --file /root/rivermind-data/4d-native-prerequisites-20261009/cuda-build-explicit.txt > /root/rivermind-data/4d-native-prerequisites-20261009/cuda-build-install.log 2>&1
printf "%s\n" "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/cuda-build-install.exit
