#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
inference_python=""
if [[ -n "${OVERNIGHT_PYTHON:-}" ]]; then
  if [[ ! -x "${OVERNIGHT_PYTHON}" ]]; then
    echo 'OVERNIGHT_PYTHON must point to an executable inference Python.' >&2
    exit 2
  fi
  inference_python="${OVERNIGHT_PYTHON}"
elif [[ -n "${ACTIONMESH_WORKDIR:-}" && -x "${ACTIONMESH_WORKDIR}/inference-env/bin/python" ]]; then
  inference_python="${ACTIONMESH_WORKDIR}/inference-env/bin/python"
elif [[ -x /root/rivermind-data/actionmesh-repro/inference-env/bin/python ]]; then
  inference_python=/root/rivermind-data/actionmesh-repro/inference-env/bin/python
elif [[ -x "${project_dir}/actionmesh/inference-env/bin/python" ]]; then
  inference_python="${project_dir}/actionmesh/inference-env/bin/python"
else
  inference_python="$(command -v python3)"
fi
export PYTHONPATH="${project_dir}/actionmesh${PYTHONPATH:+:${PYTHONPATH}}"
if [[ $# -eq 0 ]]; then
  set -- run
fi
exec "${inference_python}" -u -m research_overnight "$@"
