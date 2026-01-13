# Quick Start Guide

Get up and running with WiFi Room Sensing in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Test with Simulation

The easiest way to see the system in action is with simulation mode:

```bash
python main.py
```

You should see:
- **Console output** showing detected objects
- **3D visualization** window with room and objects
- **Dashboard** with detection statistics

## 3. Understanding the Output

### Console

```
[Frame 42] Detected: 2 objects - person, chair
```

### Saved Files

Check `./output/` folder for:
- `scene_*.ply` - 3D model files
- `dashboard_*.png` - Visualization snapshots
- `summary_*.txt` - Detection logs

## 4. Customize Your Room

Edit `config.yaml`:

```yaml
room:
  dimensions: [5.0, 4.0, 3.0]  # Your room size: length, width, height (meters)
```

## 5. Next Steps

### Option A: Use Real Hardware

See **[HARDWARE_SETUP.md](HARDWARE_SETUP.md)** for:
- Intel 5300 setup (Linux)
- ESP32 setup (any platform)

### Option B: Train Custom Model

1. **Collect training data**:
```bash
# Example: Collect data with a person in the room
python collect_training_data.py --label person --duration 60

# Collect data with empty room
python collect_training_data.py --label empty_room --duration 60

# Collect data with furniture
python collect_training_data.py --label chair --duration 60
```

2. **Train model**:
```bash
python train_model.py --data ./data/training --epochs 50
```

3. **Update config** to use trained model:
```yaml
model:
  pretrained_path: './models/best_model.pth'
```

4. **Run with trained model**:
```bash
python main.py
```

## Common Commands

```bash
# Run with visualization
python main.py

# Run without visualization (faster, headless)
python main.py --no-viz

# Run for 60 seconds then stop
python main.py --duration 60

# Use custom config
python main.py --config my_config.yaml

# Test individual components
python csi_collector.py
python signal_processor.py
python ai_model.py
```

## Troubleshooting

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"No display" error**
```bash
python main.py --no-viz
```

**Need help?**
- Read [README.md](README.md) for full documentation
- Check [HARDWARE_SETUP.md](HARDWARE_SETUP.md) for hardware issues

## Example Workflow

### Simulation Testing
```bash
# 1. Run system
python main.py

# 2. Observe console and visualizations

# 3. Check output
ls output/
```

### Real Hardware
```bash
# 1. Set up hardware (see HARDWARE_SETUP.md)

# 2. Update config.yaml
#    device_type: 'intel5300' or 'esp32'

# 3. Collect training data
python collect_training_data.py --label person --duration 120
python collect_training_data.py --label chair --duration 120
python collect_training_data.py --label table --duration 120

# 4. Train model
python train_model.py --epochs 50

# 5. Update config with trained model path

# 6. Run system
python main.py
```

---

**That's it! You're now sensing with WiFi!**

For more details, see [README.md](README.md)
