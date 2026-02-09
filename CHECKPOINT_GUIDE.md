# Person-in-WiFi-3D Checkpoint Guide

## Quick Answer: Where to Get `pwifi3d.pth`

The checkpoint file is **NOT included** in the repo. You need to either:

1. **Download pre-trained model** from the official Person-in-WiFi-3D GitHub release
2. **Train it yourself** with WiFi CSI data
3. **Use the provided dummy version** for testing

---

## Option 1: Download Pre-trained Model (Recommended)

### Get it from GitHub

```bash
# The official repo with pre-trained models:
# https://github.com/aiotgroup/Person-in-WiFi3D

# Look in the "Releases" section for model checkpoints:
# https://github.com/aiotgroup/Person-in-WiFi3D/releases
```

### Download and Setup

```bash
cd /home/pi/wifi-3d-fusion

# Create weights directory
mkdir -p env/weights

# Download the checkpoint (example - check actual GitHub release for link)
cd env/weights

# Option A: Using wget (if URL is available)
wget https://github.com/aiotgroup/Person-in-WiFi3D/releases/download/v1.0/pwifi3d_model.pth -O pwifi3d.pth

# Option B: Manual download
# 1. Visit: https://github.com/aiotgroup/Person-in-WiFi3D/releases
# 2. Download the .pth file
# 3. Move to: /home/pi/wifi-3d-fusion/env/weights/pwifi3d.pth
```

### Verify

```bash
ls -lh /home/pi/wifi-3d-fusion/env/weights/pwifi3d.pth
# File should be 100MB+ (typically 200-500MB)
```

---

## Option 2: Train Your Own

If you have WiFi CSI training data:

```bash
# Training script (inside Person-in-WiFi-3D repo)
cd third_party/Person-in-WiFi-3D-repo

# Run training with WiFi config
python tools/train.py configs/wifi/petr_wifi.py \
  --work-dir env/weights \
  --gpu-id 0

# Output will be saved as: env/weights/latest.pth
# Copy to expected location:
cp env/weights/latest.pth ../../env/weights/pwifi3d.pth
```

---

## Option 3: Use Dummy Model (For Testing)

The web dashboard works with **dummy data** even without the checkpoint:

```bash
cd /home/pi/wifi-3d-fusion

# Run with simulated WiFi data
python run_js_visualizer.py --dummy

# Visit: http://192.168.9.155:5000
# You'll see simulated 3D poses with skeleton tracking
```

---

## What the Checkpoint Contains

The `pwifi3d.pth` file has:
- **PETR Transformer backbone** - processes WiFi CSI signals
- **3D pose head** - outputs 3D skeleton coordinates (17-25 keypoints)
- **Pre-trained weights** - trained on WiFi pose dataset (97K+ frames)

**File size**: ~250-300 MB (relatively small for a 3D pose model)

---

## WiFi CSI Data Format (if training)

If you want to train your own, you need:

```
data/
├── wifipose/
│   ├── train_data/
│   │   ├── csi/              # WiFi CSI arrays (numpy)
│   │   ├── keypoint/         # 3D skeleton annotations (JSON/numpy)
│   │   └── train_data_list.txt
│   └── test_data/
│       ├── csi/
│       ├── keypoint/
│       └── test_data_list.txt
```

Each CSI file: ~1KB-10KB (depending on subcarrier count)
Each keypoint file: JSON with 17-25 3D joint coordinates

---

## Troubleshooting

**Q: File is too large (>1GB)**
A: You probably downloaded the wrong file. Look for `pwifi3d.pth` or `model.pth`, not dataset files.

**Q: 404 error downloading from GitHub**
A: The release might have moved. Check the actual GitHub repo releases page.

**Q: "FileNotFoundError: No such file or directory: 'pwifi3d.pth'"**
A: Make sure the file exists:
```bash
test -f env/weights/pwifi3d.pth && echo "Found!" || echo "Missing!"
```

**Q: Can I use it without a GPU?**
A: Yes! PyTorch can run inference on CPU, but it will be slower (~0.5-2 FPS instead of 10+ FPS).

---

## References

- **Official Project**: https://aiotgroup.github.io/Person-in-WiFi-3D/
- **GitHub Repo**: https://github.com/aiotgroup/Person-in-WiFi3D
- **Paper**: https://arxiv.org/abs/2306.08987
- **Dataset**: Available in the GitHub repo releases

---

## Current Status

✅ Web dashboard: Working (showing dummy data)
❌ pwifi3d.pth: Not included (needs to be downloaded or trained)
⏳ After adding checkpoint: Will show actual WiFi-based 3D poses

Once you have the checkpoint, enable it in `configs/fusion.yaml`:
```yaml
bridges:
  person_in_wifi_3d:
    enabled: true
    checkpoint: env/weights/pwifi3d.pth
```
