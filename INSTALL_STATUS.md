# WiFi-3D Fusion - Installation Status & Next Steps

## ✅ Current Status

### Installed Components (via pip in `.venv`):
- ✅ **Core Libraries**: numpy, scipy, pandas, scikit-learn
- ✅ **Image Processing**: opencv-python, matplotlib, pillow
- ✅ **Data Processing**: csiread, PyYAML, loguru, tqdm
- ✅ **ML/Compute**: tensorflow/torch-ready environment
- ✅ **Visualization (partial)**: plotly, pyvista, ipython
- ✅ **Jupyter Support**: jupyter_core, ipywidgets
- ✅ **Third-party Repos**: 
  - Person-in-WiFi-3D-repo
  - NeRF2  
  - 3D_wifi_scanner

### NOT Yet Installed:
- ❌ **open3d** (required for Live 3D visualization) - No ARM64 wheels on PyPI
- ⚠️ **PyTorch** (optional, for advanced ML) - Recommend CPU version

---

## 🚀 To Enable Full Visualization (Recommended)

### Option 1: Install via Conda (RECOMMENDED - 3 min)

This is **the recommended approach** for Raspberry Pi. Conda-forge has proper ARM64 wheels for open3d.

```bash
cd ~/wifi-3d-fusion
bash install_conda.sh
```

This script will:
1. ✓ Download & install Miniforge (lightweight conda for ARM64)
2. ✓ Create `wifi3d` conda environment with open3d + all dependencies
3. ✓ Clone/update third-party repos

**Then activate it:**
```bash
conda activate wifi3d
python your_script.py  # Full visualization support!
```

**Verify it worked:**
```bash
conda activate wifi3d
python -c "import open3d as o3d; print('✓ Open3D ready!')"
```

---

### Option 2: Manual Conda Installation

If you prefer manual control:

```bash
# Install miniforge (one-time)
wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-aarch64.sh
bash Miniforge3-24.3.0-0-Linux-aarch64.sh
# Say YES to initialization, restart shell

# Create environment
cd ~/wifi-3d-fusion
conda env create -f environment-arm64.yml

# Activate
conda activate wifi3d
```

---

## 📋 What You Have Now (pip venv)

You can use the pip-installed `.venv` for:
- ✅ Data processing & analysis
- ✅ Model training (CPU-based)
- ✅ CSI signal processing
- ✅ 2D plotting (matplotlib, plotly)
- ✅ Development/testing without visualization
- ⚠️ Not for 3D visualization (missing open3d)

**Activate pip venv:**
```bash
source .venv/bin/activate
python your_script.py
```

---

## ⚠️ Why WiFi-3D Scripts Need Open3D

The project uses open3d for real-time 3D visualization:
- `src/pipeline/realtime_viewer.py` - Live point cloud viewer
- `src/pipeline/hyper_view.py` - 3D spatial viewer

**If you try to use these without open3d**, you'll get:
```
ImportError: open3d is required for visualization.
On Raspberry Pi, use conda:
  bash install_conda.sh && conda activate wifi3d
```

---

## 🔧 Running Scripts

### With Full Visualization (conda):
```bash
conda activate wifi3d
python run_realtime_hop.py          # Works! 3D visualization enabled
python run_skeleton_demo.py         # Works!
```

### Without Visualization (pip venv):
```bash
source .venv/bin/activate
python tools/train_reid.py          # Data processing works
python tools/simulate_csi.py        # Algorithms work
# Some visualization features will error
```

---

## 🍓 Raspberry Pi AI Hat+ 2 Notes

- **Open3D with conda-forge**: Fully optimized for ARM64, uses hardware acceleration where available
- **PyTorch CPU**: If you need ML models, install via conda with CPU support
- **Memory**: On Pi 5, 8GB variant recommended for smooth operation
- **Storage**: Project needs ~2GB with all third-party repos

---

## ❓ Troubleshooting

### Q: Installation fails or hangs?
**A:** Network issues on Pi are common. Run with verbose output:
```bash
bash install_conda.sh -v  # If using script
```
Or restart and retry - many timeouts are transient.

### Q: "conda: command not found"
**A:** Miniforge not installed. Run:
```bash
bash install_conda.sh  # This will install it
```

### Q: Can I use pip to install open3d?
**A:** Unfortunately, `open3d==0.19.0` has no ARM64 wheels on PyPI. You must use conda-forge (which is what `install_conda.sh` does).

### Q: Virtual env conflicts?
**A:** You have two separate envs - they don't conflict:
- `.venv/` - pip-based (no open3d)
- `~/miniforge3/envs/wifi3d/` - conda-based (has open3d)

Only source one at a time.

---

## 📊 Environment Comparison

| Feature | pip venv | conda wifi3d |
|---------|----------|-------------|
| Core dependencies | ✅ OK | ✅ OK |
| Open3D visualization | ❌ Missing | ✅ Works |
| Data processing | ✅ Works | ✅ Works |
| PyTorch/ML | ⚠️ (CPU only) | ✅ (CPU optimized) |
| Installation time | ~2 min | ~10 min |
| Disk space | ~1.2 GB | ~2.5 GB |

---

## ✨ Next Steps

**1. For visualization (recommended):**
```bash
cd ~/wifi-3d-fusion
bash install_conda.sh
conda activate wifi3d
python run_realtime_hop.py
```

**2. To train models (data-only):**
```bash
source .venv/bin/activate
python tools/train_reid_v2.py
```

**3. For development:**
Use either environment depending on your needs.

---

## 📚 More Information

- See [RPI_SETUP.md](RPI_SETUP.md) for detailed Raspberry Pi setup
- Main [README.md](../README.md) for project overview
- Check [requirements.txt](requirements.txt) for all pip dependencies
- See [environment-arm64.yml](environment-arm64.yml) for conda dependencies

---

**Installation Status: PARTIAL ✓/✅**
- Pip environment: Ready
- Conda environment: Follow script to complete
- Third-party repos: ✅ Cloned
