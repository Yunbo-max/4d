#!/bin/bash
export CONDA_PKGS_DIRS=/opt/4d-native-package-cache
/opt/conda/bin/conda create --yes --json -p /opt/4d-native-cu121 --file /root/rivermind-data/4d-native-prerequisites-20261009/native-conda-explicit.txt > /root/rivermind-data/4d-native-prerequisites-20261009/conda-install.json 2> /root/rivermind-data/4d-native-prerequisites-20261009/conda-install.stderr
printf "%s\n" "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/conda-install.exit
