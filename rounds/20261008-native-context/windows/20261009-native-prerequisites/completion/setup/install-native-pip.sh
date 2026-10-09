#!/bin/bash
/opt/4d-native-cu121/bin/python -m pip install --no-build-isolation --no-deps --require-hashes --report /root/rivermind-data/4d-native-prerequisites-20261009/pip-install-report.json -i https://pypi.tuna.tsinghua.edu.cn/simple -r /root/rivermind-data/4d-native-prerequisites-20261009/native-pip-resolved-lock.txt > /root/rivermind-data/4d-native-prerequisites-20261009/pip-install.log 2>&1
printf "%s\n" "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/pip-install.exit
