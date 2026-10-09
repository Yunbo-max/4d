#!/bin/bash
export CUDA_VISIBLE_DEVICES=''
export CUDA_HOME=/opt/4d-cuda121-build
export PATH=/opt/4d-cuda121-build/bin:/opt/4d-native-cu121/bin:$PATH
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=2
/opt/4d-native-cu121/bin/python -m pip wheel --no-build-isolation --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple diso==0.1.4 -w /root/rivermind-data/4d-native-prerequisites-20261009/wheels > /root/rivermind-data/4d-native-prerequisites-20261009/diso-build.log 2>&1
printf '%s\n' "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/diso-build.exit
