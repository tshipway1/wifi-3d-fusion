# Interactive 3D Visualization Controls

## Mouse & Touch Interactions

### Desktop
- **Left-Click + Drag**: Rotate the camera around the scene
  - Drag horizontally: Rotate left/right (X-axis spin)
  - Drag vertically: Rotate up/down (Y-axis spin)
- **Mouse Wheel / Scroll**: Zoom in/out
  - Scroll up: Zoom in (closer view)
  - Scroll down: Zoom out (wider view)

### Mobile/Tablet
- **Single Finger Drag**: Rotate camera (same as mouse drag)
- **Two Finger Pinch**: Zoom in/out
  - Pinch together: Zoom out
  - Pinch apart: Zoom in

## Understanding the 3D Scene

### What You're Looking At

The visualization shows a **real-time 3D spatial view** of:
- **18m × 18m floor space** with 1-meter grid markers
- **8-meter ceiling height** showing the vertical space
- **Furniture/obstacles** as brown wireframe boxes (realistic room)
- **Detected persons** as red/orange cones with position labels
- **Skeleton joints** as yellow dots connected by green lines

### Reference Frame

- **Red Line (X)**: Left-right direction (0-18m)
- **Green Line (Y)**: Floor-to-ceiling height (0-8m)
- **Blue Line (Z)**: Front-back depth (0-18m)
- **White Sphere**: Origin point (0, 0, 0)

### Interpreting Person Positions

When a person is detected:
- **Red Cone**: The person location
- **White Label Box**: Shows `Person #ID`, coordinates `[X, Y, Z]`, and confidence %
- **Yellow Dots**: Skeleton joints (body pose)
- **Green Lines**: Connections between joints

### Confidence Scores

- **95-100%**: Highly confident detection (clear WiFi signature)
- **70-94%**: Good detection quality (typical WiFi sensing)
- **Below 70%**: Lower confidence (ambiguous or weak signal)

## Camera Controls Tips

1. **Explore Angles**: Drag to see the room from different perspectives
2. **Zoom to Focus**: Scroll to zoom in on a specific area or person
3. **Track Movement**: Watch as persons move across the room - they appear as moving red cones
4. **Verify Layout**: The furniture arrangement shows realistic obstacles
5. **Distance Reference**: Grid cells are exactly 1 meter, use them to judge distances

## Common Use Cases

### Debug WiFi Detection
1. Zoom into an area where you expect a detection
2. Look for red cones - these indicate detected persons
3. High confidence = reliable detection; low confidence = questionable

### Verify Pose Estimation
1. Look at yellow skeleton dots around detected persons
2. Dense dots = detailed pose tracking; sparse = lower resolution
3. Rotate view to check if skeleton aligns with person cone

### Understand Coverage
1. Move around the room (rotate camera) to see coverage gaps
2. Places with many detections = good WiFi coverage
3. Check furniture doesn't block detections

### Test Room Layout
1. Furniture obstacles are shown in brown
2. Verify CSI signal can propagate around obstacles
3. Test detection in different parts of the room

## Performance Notes

- **Desktop**: Smooth interaction with modern browsers (Chrome, Firefox, Safari)
- **Mobile**: Single-finger drag and pinch-zoom work well on tablets
- **Network**: Works on local network (192.168.x.x or localhost:5000)
- **FPS**: Typically 30-60 FPS depending on system load

## Troubleshooting

| Issue | Solution |
|-------|----------|
| View is too zoomed in | Scroll down to zoom out |
| Camera is upside down | This shouldn't happen - rotation is clamped; try refreshing |
| Movement is slow | Normal on Raspberry Pi; CPU-bound not GPU-bound |
| Scene appears flat | Rotate view using drag - the isometric angle only shows one side |
| Persons not visible | They might be off-screen; zoom out or rotate view |

## Technical Details

### Camera System
- Uses spherical coordinates for smooth rotation
- Distance: 10-80 meters (clamped)
- Vertical angle: ±72° (clamped to prevent flipping)
- Looks at room center: [9, 4, 9]

### Scene Geometry
- Floor grid: 1m × 1m cells (18×18 total)
- Ceiling: 8m high with sparse grid
- Room perimeter: Green wireframe walls
- Reference markers: Every 5m for scale

### Data Integration
- Updates at **5 FPS** (200ms polling)
- Real-time person detection overlay
- Live skeleton visualization
- Confidence score updates

---

**Version**: WiFi CSI Monitor v3.0 with Interactive Controls  
**Status**: ✅ Fully functional on Raspberry Pi  
**Purpose**: Explore WiFi-based person sensing in 3D space
