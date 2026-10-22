#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_dir/.venv/bin/activate"

export LD_LIBRARY_PATH="$(python -c 'import os, nvidia.cublas, nvidia.cudnn; print(os.path.join(nvidia.cublas.__path__[0], "lib") + ":" + os.path.join(nvidia.cudnn.__path__[0], "lib"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$project_dir/.venv/bin/subtitle-creator" "$@"
