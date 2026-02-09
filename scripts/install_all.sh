# scripts/install_all.sh
#!/usr/bin/env bash
set -euo pipefail

# Configuration
WITH_POSE=${WITH_POSE:-"false"}   # set WITH_POSE=true to install OpenMMLab stack
TORCH_CUDA=${TORCH_CUDA:-"cu121"} # cu118|cu121|cpu
PYTHON_BIN=${PYTHON_BIN:-"python3"}
USE_CONDA=${USE_CONDA:-"auto"}    # auto|true|false - auto detects ARM64

# Detect ARM64
ARCH=$(uname -m)
IS_ARM64=false
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  IS_ARM64=true
fi

echo "============================================"
echo "wifi-3d-fusion Installation"
echo "============================================"
echo "Architecture: $ARCH (ARM64: $IS_ARM64)"
echo "Python: $PYTHON_BIN"
echo ""

# Auto-detect conda preference for ARM64
if [[ "$USE_CONDA" == "auto" && "$IS_ARM64" == "true" ]]; then
  if command -v conda &> /dev/null; then
    echo "✓ ARM64 detected + conda found → using conda (RECOMMENDED)"
    USE_CONDA=true
  else
    echo "⚠ ARM64 detected but conda not found → falling back to pip"
    echo "  For best results on Raspberry Pi, install conda-forge:"
    echo "  https://conda-forge.org/docs/user/install_linux_aarch64/"
    USE_CONDA=false
  fi
elif [[ "$USE_CONDA" == "auto" ]]; then
  USE_CONDA=false
fi

if [[ "$USE_CONDA" == "true" ]]; then
  # ============== CONDA INSTALLATION (RECOMMENDED FOR ARM64) ==============
  echo ""
  echo "Using conda for ARM64-optimized packages..."
  
  if [ ! -f environment-arm64.yml ]; then
    echo "ERROR: environment-arm64.yml not found!"
    exit 1
  fi
  
  # Create or update the environment
  CONDA_ENV_NAME=${CONDA_ENV_NAME:-"wifi3d"}
  echo "Creating conda environment: $CONDA_ENV_NAME"
  conda env create -f environment-arm64.yml --name $CONDA_ENV_NAME --yes
  
  echo ""
  echo "✓ Conda environment created successfully!"
  echo ""
  echo "Activate with:"
  echo "  conda activate $CONDA_ENV_NAME"
  echo ""
  
else
  # ============== PIP INSTALLATION (FALLBACK) ==============
  echo ""
  echo "Using pip installer..."
  
  if [[ "$IS_ARM64" == "true" ]]; then
    echo "⚠️  WARNING: Installing on ARM64 via pip"
    echo "   open3d==0.19.0 has no prebuilt ARM64 wheels on PyPI"
    echo "   Attempting installation - may fail or fall back to source build"
    echo ""
    echo "   For best results, install conda-forge:"
    echo "   https://conda-forge.org/docs/user/install_linux_aarch64/"
    echo ""
  fi
  
  $PYTHON_BIN -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip wheel setuptools
  
  # Base deps (PyPI default index)
  echo "Installing base dependencies..."
  pip install -r requirements.txt
fi

# --- Clone third-party repos ---
echo ""
echo "Cloning third-party repositories..."
mkdir -p third_party
cd third_party
[ -d Person-in-WiFi-3D-repo ] || git clone https://github.com/aiotgroup/Person-in-WiFi-3D-repo
[ -d NeRF2 ] || git clone https://github.com/XPengZhao/NeRF2
[ -d 3D_wifi_scanner ] || git clone https://github.com/Neumi/3D_wifi_scanner
cd ..

# --- Optional: OpenMMLab stack for Person-in-WiFi-3D ---
if [[ "$WITH_POSE" == "true" ]]; then
  echo ""
  echo "Installing OpenMMLab stack (WITH_POSE=true)..."
  
  if [[ "$USE_CONDA" == "true" ]]; then
    CONDA_ENV_NAME=${CONDA_ENV_NAME:-"wifi3d"}
    # Install via conda env
    echo "To install pose stack, run:"
    echo "  conda activate $CONDA_ENV_NAME"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/$TORCH_CUDA"
    echo "  pip install --upgrade openmim"
    echo "  mim install 'mmengine>=0.10.3' 'mmcv>=2.0.0' 'mmdet>=3.2.0'"
  else
    # pip env
    case "$TORCH_CUDA" in
      cu118) TORCH_SPEC="torch==2.2.2+cu118 torchvision==0.17.2+cu118";;
      cu121) TORCH_SPEC="torch==2.2.2+cu121 torchvision==0.17.2+cu121";;
      cpu)   TORCH_SPEC="torch==2.2.2+cpu torchvision==0.17.2+cpu";;
      *) echo "Unknown TORCH_CUDA: $TORCH_CUDA"; exit 1;;
    esac
    pip install $TORCH_SPEC --index-url https://download.pytorch.org/whl/$TORCH_CUDA
    pip install --upgrade openmim
    mim install "mmengine>=0.10.3"
    mim install "mmcv>=2.0.0"
    mim install "mmdet>=3.2.0"
  fi
fi

echo ""
echo "============================================"
echo "Install complete!"
echo "============================================"
echo ""

if [[ "$USE_CONDA" == "true" ]]; then
  CONDA_ENV_NAME=${CONDA_ENV_NAME:-"wifi3d"}
  echo "Next steps:"
  echo "  1. Activate: conda activate $CONDA_ENV_NAME"
  echo "  2. Run: python scripts/your_script.py"
else
  echo "Next steps:"
  echo "  1. Activate: source .venv/bin/activate"
  echo "  2. Run: python scripts/your_script.py"
fi

echo ""
echo "Documentation & Configuration:"
echo "  ARM64 setup with conda: conda env create -f environment-arm64.yml"
echo "  WITH_POSE=true: bash scripts/install_all.sh WITH_POSE=true"
echo "  USE_CONDA=true: bash scripts/install_all.sh USE_CONDA=true"
