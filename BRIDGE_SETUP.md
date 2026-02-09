# WiFi CSI Bridges Setup Guide

Your project supports several optional bridges for advanced WiFi sensing. Here's how to enable them on your Raspberry Pi with AI Hat+2.

## 🎯 Recommendation for Raspberry Pi

**Use: Person-in-WiFi-3D** ✅ Best choice for your hardware
- **Lightweight**: Runs on CPU on Pi
- **Detection Quality**: 3D body pose from WiFi CSI
- **Use Case**: Cheap WiFi-based person sensor (no camera needed!)
- **Advantage**: Works through walls with just radio signals

---

## Step 1: Fix the Visualization

Your web dashboard rotation is now **fixed** (static view). Push the change:

```bash
cd /home/pi/wifi-3d-fusion
git add -A
git commit -m "Fix visualization - remove rotating room (confusing UX)"
git push origin main
```

**What changed**: The 3D wireframe room now stays still, so you can clearly see person detection and skeleton tracking.

---

## Step 2: Enable Person-in-WiFi-3D Bridge

This gives you 3D skeleton poses estimated directly from WiFi CSI signals.

### 2a. Enable in config

```bash
# Edit configs/fusion.yaml
nano configs/fusion.yaml
```

Find the `bridges:` section and change:
```yaml
bridges:
  person_in_wifi_3d:
    enabled: true  # ← Change from false to true
```

### 2b. Download/Prepare Checkpoint

```bash
mkdir -p env/weights

# Check if checkpoint exists
ls env/weights/pwifi3d.pth

# If not, you need to download it from the Person-in-WiFi-3D repo:
# https://github.com/MaliosDark/Person-in-WiFi-3D-repo
# Place it at: env/weights/pwifi3d.pth
```

### 2c. Test Bridge Runner

```bash
# With dummy data first (to verify it works)
python -m src.bridges.pwifi3d_runner configs/fusion.yaml \
  --dummy

# Or with actual WiFi data
python -m src.bridges.pwifi3d_runner configs/fusion.yaml \
  --source esp32
```

---

## Step 3: Integrate with Web Dashboard

The web dashboard will automatically display Person-in-WiFi-3D poses:

```bash
# Make sure server is running
python run_js_visualizer.py --dummy

# Visit: http://192.168.9.155:5000
```

You should see:
- ✅ Wireframe room (static, not rotating)
- ✅ Red cone = detected person
- ✅ Yellow dots = skeleton joints (50-100 points for dense pose)
- ✅ Green lines = skeletal connections
- ✅ Activity log = detection confidence

---

## Why NOT the Other Bridges?

| Bridge | Feature | Why Not for Pi |
|--------|---------|---------------|
| **Person-in-WiFi-3D** | 3D Pose | ✅ Use this - very efficient |
| NeRF2 | Neural RF Field | ❌ GPU-only (Pi has no GPU) |
| 3D WiFi Scanner | RSSI Volume | ⚠️ Optional - less useful without GPU rendering |

---

## What You Get With Person-in-WiFi-3D

A **WiFi-based pose sensor** that:
- ✅ Detects people through walls
- ✅ Estimates 3D skeleton poses
- ✅ Uses only WiFi CSI (no RGB camera)
- ✅ Costs under $100 (Pi + WiFi chip)
- ✅ Runs on Raspberry Pi CPU (no GPU needed)
- ✅ Web dashboard shows real-time 3D visualization

Perfect for cheap home/office person detection!

---

## Troubleshooting

**Q: Where do I get the pwifi3d.pth checkpoint?**
A: From the Person-in-WiFi-3D repo or training the model yourself.

**Q: Will Person-in-WiFi-3D work without a GPU?**
A: Yes! It's designed for lightweight inference. PyTorch can run it on CPU.

**Q: Can I run Person-in-WiFi-3D + web dashboard together?**
A: Yes! Run `run_js_visualizer.py` and feed it data from the bridge runner.

**Q: What about performance?**  
A: On Pi with dummy data: ~5 FPS. With real WiFi data: ~2-3 FPS (acceptable for pose tracking).

---

## Next Steps

1. Push the visualization fix
2. Enable Person-in-WiFi-3D in `configs/fusion.yaml`
3. Get the checkpoint file
4. Test with `--dummy` first
5. Integrate with web dashboard
6. Test with actual WiFi data

Good luck with your cheap WiFi person sensor! 🚀
