# Live WiFi CSI Data Setup Guide

This guide explains how to switch from dummy data to live WiFi Channel State Information (CSI) data for real-time 3D person detection.

## Available CSI Data Sources

The system supports 4 data sources:

| Source | Hardware Required | Difficulty | Status | Speed |
|--------|------------------|-----------|---------|--------|
| **monitor** | Linux WiFi card in monitor mode | Medium | ✅ Supported | Real-time |
| **esp32** | ESP32 with custom CSI firmware | Medium | ✅ Supported | Real-time |
| **nexmon** | Nexmon-capable WiFi card (BCM43xx) | Hard | ✅ Supported | High-speed |
| **dummy** (default) | None (simulated) | Easy | ✅ Built-in | Fixed rate |

## Option 1: WiFi Monitor Mode (Linux Monitor Interface)

### Prerequisites
- Linux WiFi adapter capable of monitor mode
- `airmon-ng` tool installed (from `aircrack-ng` package)
- Root/sudo privileges

### Setup Steps

```bash
# 1. Install required tools (if not already installed)
sudo apt-get install aircrack-ng

# 2. Put WiFi interface into monitor mode
# First, find your interface name:
iwconfig

# Then put it into monitor mode (example: wlan0 → mon0)
sudo airmon-ng start wlan0

# 3. Verify monitor mode is active
iwconfig mon0

# 4. Run visualization with monitor source
conda activate wifi3d
python run_js_visualizer.py --source monitor --interface mon0
```

### Stop Monitor Mode
```bash
sudo airmon-ng stop mon0
```

**Pros:**
- Works with standard Linux WiFi adapters
- Real-time WiFi packet capture
- No external hardware needed

**Cons:**
- Requires root privileges
- Disables normal WiFi connectivity
- CSI data may be limited depending on adapter

---

## Option 2: ESP32 CSI Receiver (over UDP)

### Prerequisites
- ESP32 microcontroller ($10-15)
- Custom ESP32 CSI firmware
- USB serial cable (micro-USB for ESP32)

### Setup Steps

#### A. Flash ESP32 with CSI Firmware

```bash
# 1. Install esptool
pip install esptool

# 2. Connect ESP32 to Pi via USB
# Device should appear as /dev/ttyUSB0 or /dev/ttyACM0

# 3. Flash the ESP32 CSI firmware
cd /home/pi/wifi-3d-fusion/third_party/3D_wifi_scanner/node_mcu_code
esptool.py -p /dev/ttyUSB0 -b 460800 write_flash 0x1000 node_mcu_code.bin

# 4. Configure ESP32 WiFi settings
# Edit the Arduino code to set your SSID/password before flashing
```

#### B. Run Visualization with ESP32 Source

```bash
# Start the visualization server listening for ESP32 UDP packets
conda activate wifi3d
python run_js_visualizer.py --source esp32 --interface 0.0.0.0:5555
```

The ESP32 will send CSI data over UDP to port 5555 on the Pi.

**Pros:**
- Dedicated hardware for CSI collection
- Doesn't interfere with normal WiFi
- Can cover multiple frequencies
- Low power consumption

**Cons:**
- Requires additional hardware ($15-20)
- Need to flash custom firmware
- Single AP limitation per ESP32

---

## Option 3: Nexmon CSI (Advanced)

### Prerequisites
- Nexmon-compatible WiFi card (Broadcom BCM43xx)
- Custom Nexmon firmware compiled for your adapter
- Advanced Linux knowledge

### Setup Steps

```bash
# This is a complex setup - see NEXMON_SETUP.md for detailed instructions
# Nexmon provides raw CSI at very high sampling rates

conda activate wifi3d
python run_js_visualizer.py --source nexmon --interface wlan0
```

**Pros:**
- Highest CSI data quality
- Very high sampling rates
- Most research-grade option

**Cons:**
- Complex setup process
- Limited to specific WiFi adapters
- Requires Linux kernel manipulation
- Most difficult to debug

---

## Current Hardware Detection

Your system has:
```
WiFi Adapter: wlan0
Operating Mode: Managed (connected to network)
Frequency: 5.66 GHz
Signal Strength: -32 dBm (good signal)
```

### Recommended Next Steps (in order of difficulty):

1. **Try Monitor Mode (Medium)**
   ```bash
   sudo airmon-ng start wlan0
   python run_js_visualizer.py --source monitor --interface mon0
   ```
   This is easiest to test immediately.

2. **Get an ESP32 (Medium)**
   - Order ESP32 DevKit ($10-15 with shipping)
   - Flash custom CSI firmware
   - Provides dedicated CSI collection

3. **Use Nexmon (Hard)**
   - Check if your adapter is compatible
   - Compile Nexmon for your system
   - Run advanced CSI collection

---

## Troubleshooting

### Monitor Mode Issues

**"No such interface" / "Interface not found"**
```bash
# Check current interfaces
iwconfig
ip link show

# Make sure monitor interface was created
sudo airmon-ng check kill  # Kill interfering processes
sudo airmon-ng start wlan0
```

**"Permission denied" / "Operation not permitted"**
```bash
# Run with sudo
sudo python run_js_visualizer.py --source monitor --interface mon0
```

### ESP32 Connection Issues

**USB device not recognized**
```bash
# Check if device is connected
ls /dev/ttyUSB*
dmesg | tail -20

# Try different baud rate
esptool.py -p /dev/ttyUSB0 -b 115200 flash_id
```

**No CSI packets received**
- Verify ESP32 is connected to WiFi (has LED indicator)
- Check Pi firewall allows UDP port 5555
- Verify ESP32 firmware flashed correctly

---

## Switching Between Sources

To switch data sources, just change the command-line arguments:

```bash
# Dummy data (testing, no hardware)
python run_js_visualizer.py --source dummy

# Monitor mode
python run_js_visualizer.py --source monitor --interface mon0

# ESP32 
python run_js_visualizer.py --source esp32

# Nexmon (if supported)
python run_js_visualizer.py --source nexmon --interface wlan0
```

---

## Performance Notes

| Source | Data Rate | Latency | CPU Load | Comments |
|--------|-----------|---------|----------|----------|
| Dummy | 10 Hz | 100ms | Low | Good for UI testing |
| Monitor | 10-50 Hz | 50-200ms | Medium | Depends on traffic |
| ESP32 | 10-100 Hz | Variable | Low | Very deterministic |
| Nexmon | 100+ Hz | <10ms | High | For research only |

---

## Next Steps

1. Choose your data source (start with Monitor mode)
2. Set up the hardware/software prerequisites
3. Run `python run_js_visualizer.py --source <your_source>`
4. Check logs for CSI data reception
5. Verify persons are detected in 3D visualization

For detailed CSI analysis, see [CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md).
