# WiFi-3D Fusion - Raspberry Pi Setup Guide

## Quick Start (AI Hat+ 2)

For Raspberry Pi with AI Hat+ 2 and visualization support, use **conda-forge** (recommended):

### Step 1: Install Conda-Forge

```bash
# Download miniforge (lightweight conda for ARM64)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Latest-Linux-aarch64.sh

# Install
bash Miniforge3-Latest-Linux-aarch64.sh
# Follow prompts, say yes to initialization

# Restart shell or run:
source ~/.bashrc
```

### Step 2: Install wifi-3d-fusion via Conda

```bash
cd ~/wifi-3d-fusion

# Create conda environment with all dependencies + open3d 
conda env create -f environment-arm64.yml

# Activate the environment
conda activate wifi3d
```

That's it! Now all visualization and dependencies work correctly.

### Step 3: Run Your Scripts

```bash
conda activate wifi3d
python run_realtime_hop.py
# or your other scripts
```

---

## Alternative: Using the Install Script

If you already have conda installed:

```bash
cd ~/wifi-3d-fusion
bash scripts/install_all.sh
# Script will auto-detect ARM64 + conda and use them
```

---

## Troubleshooting

### Issue: `open3d` installation fails with pip

**Solution**: Use conda instead (see above)
- Conda-forge has prebuilt ARM64 wheels
- PyPI's `open3d==0.19.0` doesn't have ARM64 support

### Issue: Python says "module open3d not found"

```bash
# Make sure you activated the conda environment:
conda activate wifi3d

# Verify it's installed:
python -c "import open3d; print(open3d.__version__)"
```

### Issue: Out of memory during installation

Conda is better optimized for ARM64. If pip is slow or fails:
```bash
# Switch to conda method (Step 1-2 above)
```

---

## Conda vs Pip Comparison

| Feature | Conda | Pip |
|---------|-------|-----|
| ARM64 wheels | ✅ Excellent | ❌ Missing (open3d) |
| CUDA packages | ⚠️ Limited | ❌ Not on Pi anyway |
| Installation speed | ✅ Faster | ⚠️ Slower/fails |
| Visualization | ✅ Works | ❌ open3d fails |

---

## Optional: Enable Pose Detection

```bash
conda activate wifi3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade openmim
mim install "mmengine>=0.10.3" "mmcv>=2.0.0" "mmdet>=3.2.0"
```

---

## Notes for AI Hat+ 2 Users

- The AI Hat+ 2 adds compute power, making complex ML tasks feasible
- Use CPU PyTorch (already in conda environment)
- Open3D visualization should work smoothly with conda-forge version
- Check `/usr/bin/python*` or conda's Python for best AI Hat integration

---

## Questions?

Check the main [README.md](../README.md) for architecture and other details.
