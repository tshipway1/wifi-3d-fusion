#!/bin/bash
# verify_installation.sh - Check installation status

echo "=================================================="
echo "WiFi-3D Fusion - Installation Verification"
echo "=================================================="
echo ""

# Check Python environment
echo "1. Checking Python environments..."
if [ -d ".venv" ]; then
  echo "   ✓ Pip venv found: .venv/"
  source .venv/bin/activate
  python -c "import sys; print(f'   ✓ Python {sys.version.split()[0]}')" 2>/dev/null || echo "   ✗ Venv activation failed"
else
  echo "   ✗ Pip venv not found"
fi

if command -v conda &> /dev/null; then
  echo "   ✓ Conda found: $(conda --version)"
else
  echo "   ⚠ Conda not installed yet (run: bash install_conda.sh)"
fi

echo ""

# Check core packages (pip)
echo "2. Checking pip venv packages..."
if [ -d ".venv" ]; then
  source .venv/bin/activate
  
  packages=("numpy" "scipy" "pandas" "cv2" "matplotlib" "sklearn" "pyvista")
  for pkg in "${packages[@]}"; do
    python -c "import $pkg" 2>/dev/null && echo "   ✓ $pkg" || echo "   ✗ $pkg"
  done
  
  python -c "import open3d" 2>/dev/null && echo "   ✓ open3d (via pip)" || echo "   ✗ open3d (expected - use conda)"
fi

echo ""

# Check third-party repos
echo "3. Checking third-party repositories..."
repos=("Person-in-WiFi-3D-repo" "NeRF2" "3D_wifi_scanner")
for repo in "${repos[@]}"; do
  [ -d "third_party/$repo" ] && echo "   ✓ $repo" || echo "   ✗ $repo"
done

echo ""

# Check conda environment
echo "4. Checking conda environment..."
if command -v conda &> /dev/null; then
  if conda env list | grep -q wifi3d; then
    echo "   ✓ wifi3d environment exists"
    conda run -n wifi3d python -c "import open3d; print('   ✓ open3d available in conda')" 2>/dev/null || echo "   ✗ open3d check failed"
  else
    echo "   ⚠ wifi3d environment not found (run: bash install_conda.sh)"
  fi
else
  echo "   ⚠ Conda not yet installed"
fi

echo ""
echo "=================================================="
echo "Quick Start:"
echo "=================================================="
echo ""
echo "For visualization (recommended):"
echo "  bash install_conda.sh"
echo "  conda activate wifi3d"
echo "  python run_realtime_hop.py"
echo ""
echo "For data processing only:"
echo "  source .venv/bin/activate"
echo "  python tools/train_reid.py"
echo ""
