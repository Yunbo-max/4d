#!/bin/bash
export CONDA_PKGS_DIRS=/opt/4d-native-package-cache
/opt/conda/bin/conda create --dry-run --json --override-channels -p /opt/4d-native-cu121 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch -c https://conda.anaconda.org/iopath -c https://conda.anaconda.org/pytorch3d -c https://conda.anaconda.org/nvidia -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main python=3.11 pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 pytorch3d=0.7.8=py311_cu121_pyt240 numpy=1.26.4 scipy=1.13.1 pip > /root/rivermind-data/4d-native-prerequisites-20261009/conda-solve-r2.json 2> /root/rivermind-data/4d-native-prerequisites-20261009/conda-solve-r2.stderr
printf '%s\n' "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/conda-solve-r2.exit
