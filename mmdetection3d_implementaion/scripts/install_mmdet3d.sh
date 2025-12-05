#!/usr/bin/env bash
# Create a conda env with an older Python and install MMDetection3D + deps.
# Run from any directory: bash mmdetection3d_implementaion/scripts/install_mmdet3d.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-mmdet3d38}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
# Match the CUDA toolkit you have installed. For CPU-only, set CUDA_VERSION=cpu.
CUDA_VERSION="${CUDA_VERSION:-11.7}"
TORCH_VERSION="${TORCH_VERSION:-1.13.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.14.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-0.13.1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
MMDET3D_DIR="${ROOT_DIR}/mmdetection3d"

if [ ! -d "${MMDET3D_DIR}" ]; then
  echo "MMDetection3D repo not found at ${MMDET3D_DIR}" >&2
  exit 1
fi

echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "Activating env..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing PyTorch ${TORCH_VERSION} (CUDA ${CUDA_VERSION})..."
if [ "${CUDA_VERSION}" = "cpu" ]; then
  conda install -y pytorch="${TORCH_VERSION}" torchvision="${TORCHVISION_VERSION}" torchaudio="${TORCHAUDIO_VERSION}" cpuonly -c pytorch
else
  conda install -y pytorch="${TORCH_VERSION}" torchvision="${TORCHVISION_VERSION}" torchaudio="${TORCHAUDIO_VERSION}" pytorch-cuda="${CUDA_VERSION}" -c pytorch -c nvidia
fi

echo "Installing OpenMIM and core OpenMMLab deps..."
pip install -U openmim
mim install "mmengine>=0.7.1,<1.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"

echo "Installing MMDetection3D runtime requirements..."
pip install -r "${MMDET3D_DIR}/requirements/runtime.txt"

echo "Installing MMDetection3D in editable mode..."
pip install -v -e "${MMDET3D_DIR}"

echo "Done. Activate with: conda activate ${ENV_NAME}"
