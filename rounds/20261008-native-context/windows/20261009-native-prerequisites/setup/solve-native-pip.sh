#!/bin/bash
/opt/4d-native-cu121/bin/python -m pip install --dry-run --report /root/rivermind-data/4d-native-prerequisites-20261009/pip-solve-report.json -i https://pypi.tuna.tsinghua.edu.cn/simple -r /root/rivermind-data/4d-native-prerequisites-20261009/native-pip-requirements.txt -c /root/rivermind-data/4d-native-prerequisites-20261009/native-core-constraints.txt > /root/rivermind-data/4d-native-prerequisites-20261009/pip-solve.log 2>&1
printf "%s\n" "$?" > /root/rivermind-data/4d-native-prerequisites-20261009/pip-solve.exit
