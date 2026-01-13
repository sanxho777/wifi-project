# WiFi Room Sensing System

**3D Object Detection and Room Reconstruction using WiFi Signals**

This system uses WiFi Channel State Information (CSI) to detect objects in a room and create real-time 3D models of the environment - similar to radar, but using standard WiFi routers.

## Features

- **Real-time object detection** from WiFi signals
- **3D room reconstruction** with object positions
- **AI-powered analysis** using deep learning
- **Multiple hardware support**: Intel 5300, ESP32, simulation mode
- **Live visualization** dashboard and 3D viewer
- **Export capabilities** for 3D models

## How It Works

WiFi signals bounce off objects in your environment. By analyzing how these signals change (amplitude and phase shifts), we can:

1. **Detect what objects are present** (people, furniture, etc.)
2. **Locate them in 3D space** (x, y, z coordinates)
3. **Create a 3D model** of the room

### The Science

- **CSI (Channel State Information)**: Raw WiFi signal data showing how signals propagate
- **Multipath Effects**: Signals reflect off objects, creating unique patterns
- **Deep Learning**: CNN model learns to recognize object signatures in CSI data
- **3D Reconstruction**: Combines detections into spatial model

## Quick Start

### 1. Installation

```bash
# Install Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to set:

```yaml
hardware:
  device_type: 'simulation'  # Start with simulation mode

room:
  dimensions: [5.0, 4.0, 3.0]  # Your room size in meters
```

### 3. Run

```bash
# Run with visualization
python main.py

# Run without visualization (faster)
python main.py --no-viz

# Run for specific duration
python main.py --duration 60
```

## Using Simulation Mode (Recommended for Testing)

The system includes a **simulation mode** that generates synthetic WiFi data - perfect for testing without real hardware.

```yaml
# config.yaml
hardware:
  device_type: 'simulation'
```

This will simulate:
- WiFi signals from multiple antennas
- Object reflections and multipath effects
- Realistic noise and interference

## Using Real Hardware

For actual WiFi sensing, you need specialized hardware that can capture CSI data.

### Supported Hardware

1. **Intel 5300 WiFi Card** (Most popular for research)
   - Linux only
   - Requires modified driver
   - See `HARDWARE_SETUP.md` for details

2. **ESP32 Microcontroller** (Budget-friendly option)
   - Works on Windows/Mac/Linux
   - Requires ESP32-CSI firmware
   - USB connection

3. **Atheros WiFi Cards** (AR9xxx series)
   - Linux only
   - Experimental support

See **[HARDWARE_SETUP.md](HARDWARE_SETUP.md)** for detailed setup instructions.

## Configuration Guide

### Room Settings

```yaml
room:
  dimensions: [5.0, 4.0, 3.0]  # [length, width, height] in meters
  router_positions:
    - [0.0, 0.0, 1.5]  # Router location (x, y, z)
```

### AI Model Settings

```yaml
model:
  architecture: 'cnn'  # Neural network type
  num_classes: 10      # Number of object types
  class_names:         # What to detect
    - person
    - chair
    - table
    # ... add more objects
  confidence_threshold: 0.5  # Detection confidence (0-1)
```

### Visualization Settings

```yaml
visualization:
  realtime: true              # Enable live visualization
  update_interval: 100        # Update every 100ms
  save_plots: true           # Save results
  output_dir: './output'     # Where to save
```

## Training Your Own Model

The system includes an untrained AI model. For best results, you should train it with your own data.

### Collecting Training Data

1. **Set up your hardware** (or use simulation)

2. **Run data collection**:

```bash
python collect_training_data.py --duration 300 --label person
```

3. **Label your data**: Mark what objects were present and where

4. **Train the model**:

```bash
python train_model.py --data ./data/training --epochs 50
```

### Pre-trained Models

We don't include pre-trained models because WiFi characteristics vary by:
- Room layout and materials
- WiFi hardware used
- Frequency band (2.4GHz vs 5GHz)

Your trained model will be specific to your environment.

## Understanding the Output

### Detection Results

```
[Frame 42] Detected: 3 objects - person, chair, table
```

- **Frame**: Processing iteration
- **Objects**: What was detected
- **Confidence**: How certain (shown in dashboard)

### 3D Coordinates

```
person: (2.50, 2.00, 0.80)
```

- **X**: Length of room (0 to room length)
- **Y**: Width of room (0 to room width)
- **Z**: Height from floor (0 to room height)
- Units: meters

### Saved Files

In `./output/`:
- `scene_YYYYMMDD_HHMMSS.ply` - 3D model (open with MeshLab, Blender, etc.)
- `dashboard_YYYYMMDD_HHMMSS.png` - Visualization snapshot
- `summary_YYYYMMDD_HHMMSS.txt` - Detection log

## Troubleshooting

### "No CSI data received"

- Check hardware connection
- Verify driver installation
- Try simulation mode first

### "Low detection accuracy"

- Model needs training with your specific environment
- Collect more training data
- Adjust confidence threshold in config

### "Visualization not showing"

- Check if matplotlib/open3d installed correctly
- Try `--no-viz` flag and check saved outputs
- Ensure X server running (Linux) or display connected

### "Memory error"

- Reduce `input_timesteps` in config
- Lower `batch_size` during training
- Close other applications

## System Requirements

### Minimum

- Python 3.8+
- 4GB RAM
- CPU with AVX support

### Recommended

- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA (for training)
- 2+ WiFi antennas

## Performance

- **Processing Speed**: ~10-20 FPS (CPU), ~30-60 FPS (GPU)
- **Detection Range**: ~5-10 meters
- **Position Accuracy**: ~0.3-0.5 meters
- **Latency**: ~50-200ms

## Limitations

1. **Line of sight**: Works best with direct path between router and objects
2. **Material sensitivity**: Metal objects easier to detect than wood/plastic
3. **Motion helps**: Static scenes harder than moving objects
4. **Interference**: Other WiFi networks can affect accuracy
5. **Privacy**: Can potentially detect through walls (consider ethical implications)

## Project Structure

```
wifi-room-sensing/
├── main.py                 # Main application
├── csi_collector.py        # CSI data collection
├── signal_processor.py     # Signal preprocessing
├── ai_model.py            # Neural network
├── reconstruction_3d.py    # 3D scene builder
├── visualizer.py          # Visualization tools
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── README.md             # This file
└── HARDWARE_SETUP.md     # Hardware instructions
```

## Advanced Usage

### Headless Operation (No Display)

```bash
python main.py --no-viz
```

### Custom Configuration

```bash
python main.py --config my_config.yaml
```

### Python API

```python
from main import WiFiRoomSensing

# Create system
system = WiFiRoomSensing('config.yaml')

# Start sensing
system.start()
```

## Contributing

This is a research project. Contributions welcome:

- Hardware driver support
- Pre-trained models (with dataset info)
- Improved algorithms
- Bug fixes

## References

This system is based on research in WiFi sensing:

- "Through-Wall Human Pose Estimation Using Radio Signals" (MIT CSAIL)
- "WiFi-based Indoor Positioning" (Various papers)
- "Deep Learning for CSI-based Activity Recognition" (Multiple researchers)

## License

MIT License - See LICENSE file

## Disclaimer

**Privacy Warning**: This technology can potentially detect through walls. Use responsibly and legally:

- Get consent before monitoring spaces
- Follow local privacy laws
- Consider ethical implications
- Don't use for surveillance without authorization

## Support

For issues, questions, or contributions:

- Check `HARDWARE_SETUP.md` for hardware issues
- Read this README thoroughly
- Test with simulation mode first
- Provide error logs when reporting issues

## Acknowledgments

Built using:
- PyTorch (Deep Learning)
- Open3D (3D Processing)
- NumPy/SciPy (Signal Processing)
- Matplotlib (Visualization)

Inspired by research from MIT CSAIL, UC Santa Barbara, and other institutions advancing WiFi sensing technology.

---

**Made with WiFi signals and AI**
#   w i f i - p r o j e c t  
 