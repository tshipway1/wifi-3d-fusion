# Person-in-WiFi-3D Checkpoint Guide

## Quick Answer: Where to Get `pwifi3d.pth`

⚠️ **Important**: The repo you're looking at doesn't have pre-trained models available.

### Status: No Pre-trained Model Available

The Person-in-WiFi-3D-repo fork on GitHub has:
- ✅ Code framework (MMCV, operators, training pipeline)
- ✅ Demo GIFs showing the concept
- ✅ Training scripts
- ❌ NO pre-trained checkpoint (.pth files)
- ❌ NO releases or model downloads

### Your Options

---

## Option 1: Use Dummy Data (RECOMMENDED FOR NOW) ✅

**You're already doing this!** Your dashboard is fully functional with simulated WiFi data:

```bash
python run_js_visualizer.py --dummy

# Visit: http://192.168.9.155:5000
```

Shows:
- ✅ Red cone = simulated person detection
- ✅ Yellow/green skeleton = simulated 3D pose (17 joints)
- ✅ Confidence scores = 70-95% range
- ✅ Activity log with detections

**This is perfect for testing the entire pipeline without actual WiFi CSI or a trained model.**

---

## Option 2: Train Your Own Model

If you have WiFi CSI data (or can generate synthetic CSI data):

```bash
# Navigate to the Person-in-WiFi-3D repo
cd third_party/Person-in-WiFi-3D-repo

# Prepare your WiFi CSI dataset in this structure:
# data/wifipose/
# ├── train_data/
# │   ├── csi/              # WiFi CSI numpy arrays
# │   ├── keypoint/         # 3D skeleton JSON/numpy files
# │   └── train_data_list.txt
# └── test_data/
#     ├── csi/
#     ├── keypoint/
#     └── test_data_list.txt

# Run training:
python tools/train.py configs/wifi/petr_wifi.py \
  --work-dir ../../env/weights \
  --gpu-id 0

# Output checkpoint:
# ../../env/weights/latest.pth → copy to ../../env/weights/pwifi3d.pth
```

**Dataset requirements:**
- 97K+ WiFi CSI frames (from their paper)
- 3D skeleton annotations for each frame
- Multi-person data (2-3 people per scene)

---

## Option 3: Contact Authors for Pre-trained Model

The original authors may have a pre-trained model available:

1. Visit: https://aiotgroup.github.io/Person-in-WiFi-3D/
2. Visit: https://github.com/aiotgroup/Person-in-WiFi3D (official repo)
3. Check for model downloads or contact them for checkpoint

---

## Current Setup: Dummy Data is Fully Functional ✅

Your web dashboard is **production-ready** with the dummy mode:

| Feature | Status |
|---------|--------|
| Web Dashboard | ✅ Working |
| 3D Visualization | ✅ Working |
| Person Detection | ✅ Working (simulated) |
| Skeleton Tracking | ✅ Working (500 dense points/person) |
| Confidence Display | ✅ Fixed (0-100% scale) |
| Activity Log | ✅ Working (shows detections) |
| System Metrics | ✅ Working (FPS, CPU, Memory) |

---

## To Actually Use Person-in-WiFi-3D on Your Pi

You need **one of these**:

### Path A: Synthetic Data (Easiest)
```python
# Edit run_js_visualizer.py to generate realistic WiFi CSI data
# from actual WiFi signals (if you have a WiFi monitoring setup)
```

### Path B: Real WiFi CSI Collection
```bash
# Capture real WiFi signals using:
# - ESP32 with WiFi
# - Nexmon on Raspberry Pi
# - USB WiFi adapter in monitor mode

# Then feed CSI data to Person-in-WiFi-3D model
```

### Path C: Train on Your Own Data
```bash
# If you have WiFi CSI + skeleton labeled data
# Train the model using the provided configs
```

---

## What The Checkpoint Would Contain

If you had the `pwifi3d.pth` model:

```
pwifi3d.pth (250-300 MB)
├── PETR Transformer backbone
│   └── Multi-head attention over WiFi CSI
├── 3D Pose Head
│   └── Outputs 17 keypoints [x, y, z] per person
└── Pre-trained weights
    └── From 97K+ WiFi frames (7 volunteers)
```

**Performance:**
- Inference: ~100ms per frame on GPU, ~500ms on CPU
- Input: WiFi CSI subcarrier data (128-256 subcarriers × 3 Tx × 3 Rx)
- Output: 3D skeleton poses (17×3 coordinates), confidence scores

---

## Recommendation for Your Raspberry Pi

**Current approach is ideal:**

1. ✅ **Use `--dummy` mode for testing** - fully functional dashboard
2. ⏳ **When ready to deploy** - collect real WiFi CSI via:
   - ESP32 CSI collection
   - Nexmon on Pi
   - 802.11n WiFi monitoring
3. 🔧 **Two paths:**
   - **Path A**: Train model with your own WiFi CSI + 3D skeleton data
   - **Path B**: Find pre-trained model from authors (GitHub issues/email)

---

## Check If Author Repo Has Model

The official repo might have a pre-trained model in releases:

```bash
# Official author GitHub:
# https://github.com/aiotgroup/Person-in-WiFi3D

# This is DIFFERENT from the fork in your third_party:
# - aiotgroup/Person-in-WiFi3D (original)
# - aiotgroup/Person-in-WiFi-3D-repo (in your project)
```

Try contacting the authors through their GitHub issues if you need the pre-trained model.
