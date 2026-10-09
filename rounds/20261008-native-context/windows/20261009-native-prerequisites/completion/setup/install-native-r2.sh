#!/bin/bash
export CONDA_PKGS_DIRS=/opt/4d-native-package-cache
/opt/conda/bin/conda create --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge --yes --json -p /opt/4d-native-cu121 --file /root/rivermind-data/4d-native-prerequisites-20261009/native-conda-explicit.txt > /root/rivermind-data/4d-native-prerequisites-20261009/conda-install-r2.json 2> /root/rivermind-data/4d-native-prerequisites-20261009/conda-install-r2.stderr
printf "%s\n" "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/conda-install-r2.exit
