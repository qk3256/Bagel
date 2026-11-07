#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-bagel}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

"$CONDA_BIN" create -n "$ENV_NAME" --override-channels -c conda-forge \
  python=3.10 pip -y

ENV_PYTHON="$HOME/miniconda3/envs/$ENV_NAME/bin/python"

# Gradio is not used by the training entrypoint and its unpinned dependency set
# currently causes extensive resolver backtracking against BAGEL's pinned stack.
sed '/^gradio$/d' requirements.txt | \
  "$ENV_PYTHON" -m pip install -i "$PYPI_INDEX" -r /dev/stdin

# The host's system nvcc may be older than the CUDA runtime bundled with torch.
# Use the official wheel matching Python 3.10, CUDA 12, torch 2.5 and ABI=false.
"$ENV_PYTHON" -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

"$ENV_PYTHON" - <<'PY'
import flash_attn
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flash_attn": flash_attn.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "smoke_shape": tuple(y.shape),
})
PY
