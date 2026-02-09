#!/bin/bash
# install_conda_and_deps.sh
# Installs miniforge conda and wifi-3d-fusion dependencies on Raspberry Pi

set -euo pipefail

echo "=================================================="
echo "WiFi-3D Fusion - Raspberry Pi Conda Setup"
echo "=================================================="
echo ""
echo "This script will install:"
echo "  1. Miniforge (conda for ARM64)"
echo "  2. Open3d with full visualization support"
echo "  3. All other dependencies"
echo ""

# Step 1: Install Miniforge
echo "Step 1: Installing Miniforge..."
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-aarch64.sh"
MINIFORGE_FILE="$HOME/miniforge_installer.sh"

echo "Downloading Miniforge..."
if command -v wget &> /dev/null; then
  wget -O "$MINIFORGE_FILE" "$MINIFORGE_URL" 2>&1 | tail -5
elif command -v curl &> /dev/null; then
  curl -L -o "$MINIFORGE_FILE" "$MINIFORGE_URL" 2>&1 | tail -5
else
  echo "ERROR: Neither wget nor curl found. Please install one."
  exit 1
fi

if [ ! -f "$MINIFORGE_FILE" ]; then
  echo "ERROR: Failed to download Miniforge"
  exit 1
fi

echo "Installing Miniforge to $HOME/miniforge3..."
bash "$MINIFORGE_FILE" -b -p "$HOME/miniforge3"

# Initialize conda
echo "Initializing conda..."
export PATH="$HOME/miniforge3/bin:$PATH"
conda init bash
source ~/.bashrc || true

echo "✓ Miniforge installed successfully"
echo ""

# Step 2: Create environment
echo "Step 2: Creating conda environment..."
cd /home/pi/wifi-3d-fusion

if [ -f environment-arm64.yml ]; then
  conda env create -f environment-arm64.yml --name wifi3d -y
  echo "✓ Environment created successfully"
else
  echo "ERROR: environment-arm64.yml not found"
  exit 1
fi

# Step 3: Install third-party repos
echo ""
echo "Step 3: Cloning third-party repositories..."
mkdir -p third_party
cd third_party
[ -d Person-in-WiFi-3D-repo ] || git clone https://github.com/aiotgroup/Person-in-WiFi-3D-repo
[ -d NeRF2 ] || git clone https://github.com/XPengZhao/NeRF2
[ -d 3D_wifi_scanner ] || git clone https://github.com/Neumi/3D_wifi_scanner
cd ..

echo "✓ Third-party repositories cloned"
echo ""

echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  conda activate wifi3d"
echo ""
echo "Test visualization:"
echo "  conda activate wifi3d"
echo "  python -c \"import open3d; print('✓ Open3D ready')\""
echo ""
