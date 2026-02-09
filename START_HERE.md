# 🚀 Quick Start - Choose Your Path

## You Have Two Options:

### **Option A: Get Visualization Working (Recommended - 3 min)**

```bash
cd ~/wifi-3d-fusion
bash install_conda.sh
conda activate wifi3d
python run_realtime_hop.py
```

### **Option B: Start Using It Now (Data Processing)**

```bash
cd ~/wifi-3d-fusion
source .venv/bin/activate
python tools/train_reid_v2.py
```

---

## What's the Difference?

| Feature | Option A (Conda) | Option B (Pip) |
|---------|------------------|----------------|
| 3D Visualization | ✅ Works | ❌ No |
| Data Processing | ✅ Works | ✅ Works |
| Installation Time | 3-5 min | Ready now! |
| Disk Space | +1.3 GB | Already used |

---

## Status Check

```bash
cd ~/wifi-3d-fusion
bash verify_installation.sh
```

---

## Why Two Options?

**open3d problem**: The 3D visualization library (`open3d==0.19.0`) has no ARM64 wheels on PyPI.

**Solution**: Use conda-forge (which has ARM64 wheels) - Option A does this automatically.

**Current state**: Pip environment (.venv) is ready NOW without open3d. Add conda anytime.

---

## Still Stuck?

- See [INSTALLATION_COMPLETE.md](INSTALLATION_COMPLETE.md) for full details
- See [INSTALL_STATUS.md](INSTALL_STATUS.md) for troubleshooting
- See [RPI_SETUP.md](RPI_SETUP.md) for Pi-specific help

---

That's it! Pick an option above and go. 🎯
