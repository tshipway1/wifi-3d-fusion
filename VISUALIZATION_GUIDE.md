# WiFi CSI Monitor v3.0 - Visualization Guide

## What You're Looking At

The 3D visualization shows a real-world WiFi-based indoor environment with person detection. Here's how to interpret it:

### 🎯 Coordinate System

The visualization uses a **3D Cartesian coordinate system**:
- **Red Line (X-axis)**: Left-Right direction (0-18 meters)
- **Green Line (Y-axis)**: Height/Vertical direction (0-8 meters)  
- **Blue Line (Z-axis)**: Front-Back/Depth direction (0-18 meters)
- **White Sphere**: Origin point (0, 0, 0)

### 🏠 Room Layout

The scene represents an **18m × 8m × 18m indoor space** with:

- **Green Grid on Floor**: 1-meter cells showing the floor plan
- **Green Perimeter**: Room boundaries (walls)
- **Dark Green Ceiling**: Upper grid lines at 8m height
- **Small Dots**: Reference markers every 5 meters for scale
- **Brown Wireframe Boxes**: Furniture and obstacles (Shelf, Cabinet, Table, Chair)

Think of it as looking down at a room from an angle - you can see the floor layout combined with a 3D perspective.

### 👤 Person Detection

When people are detected:

1. **Red Cone**: Represents a person - the larger the cone visibility, the higher confidence
2. **Red Vertical Line**: Shows detection height (people detected at ~1m = chest/torso height)
3. **Red Label Box**: Shows:
   - Person ID number (e.g., "Person #1")
   - 3D Position coordinates `[X, Y, Z]` in meters
   - Detection confidence percentage (0-100%)

Multiple people show as red (primary) and orange (secondary) cones.

### 🦴 Skeleton Tracking

When skeleton joints are detected:

- **Yellow Dots**: Individual joint positions (25 joints tracked)
- **Green Lines**: Connections between joints forming the skeleton
- Represents the pose/body position of each detected person
- The denser the yellow dots, the more detailed the pose tracking

### 📊 Understanding Person Position

Example reading: `Pos: [9.2, 0.8, 12.5]` means:
- **9.2m** along X-axis (left-right) - towards right side of room
- **0.8m** height (Y) - low, near the ground (sitting or kneeling)
- **12.5m** along Z-axis (front-back) - towards back of room

**Room Reference**:
- X=0 is left wall, X=18 is right wall
- Y=0 is floor, Y=8 is ceiling
- Z=0 is front wall, Z=18 is back wall

### 📈 Confidence Score

- **95-100%**: Very confident detection (clear signal)
- **70-94%**: Good detection (typical WiFi detection)
- **50-69%**: Moderate confidence (possible detection)
- **Below 50%**: Low confidence detection

Higher confidence means the WiFi CSI data strongly indicates a person at that location.

### 🎮 Camera Controls

The view is **fixed isometric perspective** showing:
- 25m north-east observation point
- 18m high vantage
- 25m north-east offset in depth

This angle provides clear 3D visibility of:
1. Floor layout (X and Z axes)
2. Height information (Y axis)
3. Person positions relative to furniture
4. Full skeleton poses for detected individuals

### 📏 Scale Reference

Use these landmarks to judge 3D distances:

- **Grid cells**: 1 meter each (10 cells = 10 meters)
- **Furniture sizes**: Vary from 1-3 meters
- **Person height**: ~2 meters (knee to head)
- **Room dimensions**: Marked by green perimeter walls

### 🔍 What Different Colors Mean

| Color | Element | Meaning |
|-------|---------|---------|
| 🔴 Red | X-Axis, Primary Person | Horizontal position, main person detected |
| 🟢 Green | Y-Axis, Grid, Skeleton Connections | Vertical/height, floor structure, joint links |
| 🔵 Blue | Z-Axis | Depth direction (front-back) |
| 🟡 Yellow | Skeleton Joints | Body joint positions (knees, elbows, wrists, etc.) |
| 🟠 Orange | Secondary Persons | Additional people detected |
| 🟤 Brown | Furniture | Room obstacles and objects |
| ⚫ Dark Green | Ceiling Grid, Reference Markers | Upper structure, distance markers |

### 💡 Tips for Interpretation

1. **Check the legend**: Bottom-right corner has a quick reference
2. **Person confidence**: Higher = more reliable detection
3. **Skeleton density**: More yellow dots = more detailed pose tracking
4. **Multiple people**: Look for multiple colored cones (red, orange, etc.)
5. **Room layout**: Furniture shows realistic obstacles (like a real office space)
6. **Stationary view**: Camera doesn't rotate, making it easier to track movement

### 🚀 Using This for Development

This visualization is designed to help you:

- **Debug WiFi CSI detection**: See if persons are being detected at expected coordinates
- **Understand pose estimation**: Visualize skeleton tracking accuracy
- **Test in environments**: Place furniture to simulate office/home layouts
- **Verify confidence scores**: Correlate with actual person presence
- **Optimize detection**: See patterns in person detection across the room

### 📡 Real Data Pipeline

In production mode (not dummy):

1. **WiFi CSI Collection**: From ESP32, Nexmon, or USB adapter
2. **Signal Processing**: Extract WiFi signal features
3. **Person Detection**: Neural network estimates position + confidence
4. **Pose Estimation**: Person-in-WiFi-3D model outputs skeleton (if enabled)
5. **3D Visualization**: Renders everything in real-time on the browser

Currently running in **`--dummy` mode** with synthetic data for testing. Switch to real WiFi CSI for actual person detection.

---

**Version**: WiFi CSI Monitor v3.0  
**Hardware**: Raspberry Pi with WiFi CSI collection  
**Purpose**: Cheap WiFi-based person/pose sensing for indoor environments
