# Hardware Setup Guide

This guide explains how to set up real WiFi hardware for CSI (Channel State Information) capture.

## Overview

To capture real WiFi CSI data, you need special hardware that can access low-level WiFi information. Regular WiFi adapters don't expose this data.

## Option 1: Intel 5300 WiFi Card (Recommended for Serious Use)

The Intel 5300 is the most popular option for WiFi sensing research.

### What You Need

- **Intel WiFi Link 5300 AGN** network card (~$20-50 on eBay)
- **Compatible computer**: Desktop with PCIe slot or laptop with mini-PCIe
- **Linux OS**: Ubuntu 18.04/20.04 recommended (won't work on Windows)
- **Modified driver**: Linux 802.11n CSI Tool

### Step-by-Step Setup

#### 1. Install the Hardware

```bash
# Power off computer
# Insert Intel 5300 into PCIe or mini-PCIe slot
# Power on
```

#### 2. Verify Detection

```bash
lspci | grep -i wireless
# Should see: Intel Corporation WiFi Link 5300
```

#### 3. Install Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    git \
    iw \
    tcpdump
```

#### 4. Download and Install CSI Tool

```bash
# Clone the repository
cd ~
git clone https://github.com/dhalperi/linux-80211n-csitool.git
cd linux-80211n-csitool

# Build the modified driver
make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi modules

# Install the driver
sudo make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi INSTALL_MOD_DIR=updates modules_install

# Update module dependencies
sudo depmod

# Reboot
sudo reboot
```

#### 5. Configure for CSI Capture

```bash
# Load the driver with CSI logging enabled
sudo modprobe -r iwlwifi
sudo modprobe iwlwifi connector_log=0x1

# Verify CSI interface exists
ls /sys/kernel/debug/ieee80211/
# Should show phy0 or similar
```

#### 6. Install CSI Extraction Tool

```bash
cd ~/linux-80211n-csitool/supplementary/linux

# Build the logger
make

# Test CSI capture
sudo ./log_to_file /tmp/csi.dat &

# Generate some WiFi traffic to capture CSI
ping -c 100 8.8.8.8

# Stop logging
sudo killall log_to_file

# Check if data was captured
ls -lh /tmp/csi.dat
# Should show file with data
```

#### 7. Configure WiFi Room Sensing

```yaml
# config.yaml
hardware:
  device_type: 'intel5300'
  num_antennas: 3          # Intel 5300 has 3 antennas
  num_subcarriers: 30      # Standard for Intel 5300
  sampling_rate: 100       # Adjust based on traffic
```

#### 8. Run the System

```bash
python main.py
```

### Troubleshooting Intel 5300

**Driver not loading**:
```bash
# Check kernel logs
dmesg | grep iwl

# Try older kernel version
# CSI tool may not work with very new kernels (>5.4)
```

**No CSI data**:
```bash
# Ensure monitor mode is enabled
sudo iw dev wlan0 set monitor none
sudo ifconfig wlan0 up

# Check CSI logging is enabled
cat /sys/kernel/debug/ieee80211/phy0/iwlwifi/iwlmvm/fw_dbg_conf
```

**Permission denied**:
```bash
# Add user to necessary groups
sudo usermod -a -G dialout $USER
sudo usermod -a -G plugdev $USER

# Logout and login again
```

---

## Option 2: ESP32 Microcontroller (Budget-Friendly)

ESP32 is a $5-10 microcontroller with WiFi that can provide CSI data.

### What You Need

- **ESP32 development board** (ESP32-DevKitC or similar)
- **USB cable** (for connection to computer)
- **Computer**: Windows, Mac, or Linux

### Step-by-Step Setup

#### 1. Install ESP-IDF (ESP32 Development Framework)

**Linux/Mac**:
```bash
# Install prerequisites
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util

# Get ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32

# Set up environment
. ./export.sh
```

**Windows**:
Download ESP-IDF installer from: https://dl.espressif.com/dl/esp-idf/

#### 2. Install ESP32-CSI-Tool

```bash
cd ~/esp
git clone https://github.com/StevenMHernandez/ESP32-CSI-Tool.git
cd ESP32-CSI-Tool

# Configure the project
idf.py menuconfig
# Set WiFi SSID and password in config

# Build and flash
idf.py build
idf.py -p /dev/ttyUSB0 flash  # Replace with your port (COM3 on Windows)
```

#### 3. Verify CSI Output

```bash
# Monitor serial output
idf.py -p /dev/ttyUSB0 monitor

# Should see CSI data streaming:
# CSI_DATA,type,seq,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,...
```

#### 4. Configure WiFi Room Sensing

```yaml
# config.yaml
hardware:
  device_type: 'esp32'
  num_antennas: 1          # ESP32 typically has 1 antenna
  num_subcarriers: 52      # 802.11n standard
  sampling_rate: 50        # Lower than Intel 5300
```

Update `csi_collector.py` if needed:
```python
# Adjust COM port for your system
self.serial_port = serial.Serial('COM3', 115200, timeout=1)  # Windows
# or
self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Linux
```

#### 5. Run the System

```bash
python main.py
```

### Troubleshooting ESP32

**Device not detected**:
```bash
# Linux: Check USB permissions
ls -l /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB0

# Windows: Install CP210x or CH340 USB driver
```

**Flash fails**:
```bash
# Hold BOOT button while flashing
# Try different USB cable
# Reduce baud rate: idf.py -p COM3 -b 115200 flash
```

**No CSI data**:
- Ensure ESP32 is connected to WiFi network
- Generate traffic on the network (ping, streaming, etc.)
- CSI is only captured when receiving packets

---

## Option 3: Atheros WiFi Cards (Experimental)

Atheros AR9xxx series cards can provide CSI with modified drivers.

### What You Need

- **Atheros AR93xx WiFi card** (AR9380, AR9390, etc.)
- **Linux system** (Ubuntu recommended)
- **Atheros CSI Tool**

### Setup

```bash
# Clone Atheros CSI Tool
git clone https://github.com/xieyaxiongfly/Atheros-CSI-Tool.git
cd Atheros-CSI-Tool

# Follow instructions in repository
# (Setup varies by card model)
```

**Note**: Atheros support is experimental and not fully implemented in this codebase. Contributions welcome!

---

## Option 4: Commercial WiFi Sensing Solutions

Some commercial options exist but are expensive ($1000+):

- **Origin AI**: Pre-built WiFi sensing platform
- **Cognitive Systems**: Commercial radar platform
- **Various research-grade SDRs**: Universal Software Radio Peripheral (USRP)

---

## Testing Your Setup

### 1. Test Data Collection

```bash
# Test each module independently
python csi_collector.py
# Should see CSI samples printed

python signal_processor.py
# Should process test data successfully

python ai_model.py
# Should load model and make test prediction
```

### 2. Test Full System

```bash
# Start with simulation
python main.py

# Should see:
# [CSI Collector] Starting SIMULATION mode
# [Frame X] Detected: Y objects - ...
```

### 3. Switch to Real Hardware

```yaml
# config.yaml
hardware:
  device_type: 'intel5300'  # or 'esp32'
```

```bash
python main.py
# Should now use real CSI data
```

---

## Best Practices

### 1. Hardware Placement

- **Router position**: Corner or wall of room, 1.5m height
- **Multiple routers**: Better coverage and accuracy
- **Avoid metal obstacles**: Between router and detection area

### 2. Environment Considerations

- **Room materials**: Drywall better than concrete
- **Furniture**: More reflections = better detection
- **WiFi interference**: Use less congested channels

### 3. Data Collection

- **Consistent setup**: Same router positions for training and testing
- **Traffic generation**: Continuous WiFi packets needed
  ```bash
  # Generate traffic
  ping -i 0.01 <router_ip>  # 100 packets/sec
  ```

### 4. Optimization

- **Channel selection**: Use WiFi analyzer to find clear channel
- **Bandwidth**: 20MHz vs 40MHz (more subcarriers = better)
- **Antenna configuration**: Multiple antennas improve accuracy

---

## Comparison Table

| Hardware | Cost | Setup Difficulty | Accuracy | Antennas | Platform |
|----------|------|-----------------|----------|----------|----------|
| Intel 5300 | $20-50 | Hard | Excellent | 3 | Linux only |
| ESP32 | $5-10 | Medium | Good | 1 | All platforms |
| Atheros | $15-30 | Hard | Good | 2-3 | Linux only |
| Simulation | Free | Easy | N/A | 3 | All platforms |

---

## Additional Resources

### Intel 5300
- CSI Tool Documentation: https://dhalperi.github.io/linux-80211n-csitool/
- FAQ: https://dhalperi.github.io/linux-80211n-csitool/faq.html
- Papers: Multiple research papers available

### ESP32
- ESP32-CSI-Tool: https://github.com/StevenMHernandez/ESP32-CSI-Tool
- ESP-IDF Documentation: https://docs.espressif.com/projects/esp-idf/
- CSI Format: Espressif documentation on WiFi CSI

### General WiFi Sensing
- "WiFi-based Indoor Positioning" papers
- IEEE papers on CSI-based sensing
- GitHub: Search "WiFi CSI" for various projects

---

## Getting Help

**Hardware not working?**
1. Start with simulation mode
2. Test hardware independently
3. Check hardware-specific forums
4. Verify driver versions

**Still stuck?**
- Intel 5300: Check CSI Tool GitHub issues
- ESP32: Check ESP-IDF forum
- This project: Check our documentation

---

**Remember**: WiFi sensing is experimental technology. Hardware setup can be tricky. Be patient and start with simulation mode!
