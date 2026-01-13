"""
CSI Data Collector Module
Handles WiFi Channel State Information collection from various hardware.
"""

import numpy as np
import time
import subprocess
import os
from typing import Optional, Tuple
import threading
import queue


class CSICollector:
    """
    Collects CSI data from WiFi hardware.
    Supports: Intel 5300, ESP32, Atheros, and simulation mode.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device_type = config['hardware']['device_type']
        self.num_antennas = config['hardware']['num_antennas']
        self.num_subcarriers = config['hardware']['num_subcarriers']
        self.sampling_rate = config['hardware']['sampling_rate']
        self.is_collecting = False
        self.data_queue = queue.Queue(maxsize=1000)

        print(f"[CSI Collector] Initialized for device type: {self.device_type}")

    def start_collection(self) -> bool:
        """Start CSI data collection."""
        if self.device_type == 'simulation':
            return self._start_simulation()
        elif self.device_type == 'intel5300':
            return self._start_intel5300()
        elif self.device_type == 'esp32':
            return self._start_esp32()
        elif self.device_type == 'atheros':
            return self._start_atheros()
        else:
            print(f"[ERROR] Unknown device type: {self.device_type}")
            return False

    def stop_collection(self):
        """Stop CSI data collection."""
        self.is_collecting = False
        print("[CSI Collector] Stopped collection")

    def get_csi_data(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get one CSI data packet.
        Returns: CSI matrix of shape (num_antennas, num_subcarriers, 2)
                 where last dimension is [amplitude, phase]
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ==================== SIMULATION MODE ====================
    def _start_simulation(self) -> bool:
        """
        Simulation mode: generates synthetic CSI data.
        Useful for testing without real hardware.
        """
        print("[CSI Collector] Starting SIMULATION mode")
        print("  This generates synthetic WiFi CSI data for testing.")
        print("  To use real hardware, change 'device_type' in config.yaml")

        self.is_collecting = True

        # Start simulation thread
        sim_thread = threading.Thread(target=self._simulation_worker, daemon=True)
        sim_thread.start()

        return True

    def _simulation_worker(self):
        """Worker thread that generates simulated CSI data."""
        sample_interval = 1.0 / self.sampling_rate

        # Simulate some objects in the room
        time_step = 0

        while self.is_collecting:
            start_time = time.time()

            # Generate synthetic CSI data
            # Simulate multipath effects and object reflections
            csi_amplitude = np.zeros((self.num_antennas, self.num_subcarriers))
            csi_phase = np.zeros((self.num_antennas, self.num_subcarriers))

            for ant in range(self.num_antennas):
                for sub in range(self.num_subcarriers):
                    # Base signal
                    freq = 2.4e9 + sub * 0.3125e6  # WiFi channel spacing

                    # Simulate object reflections (moving person example)
                    person_pos = 2.5 + 1.0 * np.sin(time_step * 0.1)  # Moving person
                    distance = person_pos + ant * 0.5

                    # Path loss and multipath
                    amplitude = 40 + 10 * np.sin(2 * np.pi * freq * distance / 3e8)
                    amplitude += np.random.normal(0, 2)  # Noise

                    phase = (2 * np.pi * freq * distance / 3e8) % (2 * np.pi)
                    phase += np.random.normal(0, 0.1)  # Phase noise

                    csi_amplitude[ant, sub] = amplitude
                    csi_phase[ant, sub] = phase

            # Combine amplitude and phase
            csi_data = np.stack([csi_amplitude, csi_phase], axis=-1)

            # Add to queue
            if not self.data_queue.full():
                self.data_queue.put(csi_data)

            time_step += 1

            # Maintain sampling rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            time.sleep(sleep_time)

    # ==================== INTEL 5300 MODE ====================
    def _start_intel5300(self) -> bool:
        """
        Intel 5300 WiFi card with Linux CSI Tool.
        Requires: https://github.com/spanev/linux-80211n-csitool
        """
        print("[CSI Collector] Starting Intel 5300 mode")
        print("  Requirements:")
        print("  - Intel 5300 WiFi card")
        print("  - Modified driver installed")
        print("  - Linux CSI Tool: https://github.com/spanev/linux-80211n-csitool")

        # Check if CSI tool is available
        if not os.path.exists('/sys/kernel/debug/ieee80211'):
            print("[ERROR] CSI tool not found. Please install Linux 802.11n CSI Tool.")
            print("  See HARDWARE_SETUP.md for instructions")
            return False

        self.is_collecting = True

        # Start collection thread
        intel_thread = threading.Thread(target=self._intel5300_worker, daemon=True)
        intel_thread.start()

        return True

    def _intel5300_worker(self):
        """Worker thread for Intel 5300 CSI collection."""
        try:
            # Start CSI logging
            process = subprocess.Popen(
                ['sudo', 'log_to_file', '/tmp/csi.dat'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            while self.is_collecting:
                # Read CSI data from file
                csi_data = self._parse_intel5300_format('/tmp/csi.dat')
                if csi_data is not None and not self.data_queue.full():
                    self.data_queue.put(csi_data)

                time.sleep(1.0 / self.sampling_rate)

            process.terminate()

        except Exception as e:
            print(f"[ERROR] Intel 5300 collection failed: {e}")
            self.is_collecting = False

    def _parse_intel5300_format(self, filepath: str) -> Optional[np.ndarray]:
        """Parse Intel 5300 CSI data format."""
        # This is a placeholder - actual parsing depends on CSI tool output format
        # Real implementation would parse binary CSI data
        return None

    # ==================== ESP32 MODE ====================
    def _start_esp32(self) -> bool:
        """
        ESP32 with CSI firmware.
        Requires: ESP32-CSI-Tool or similar
        """
        print("[CSI Collector] Starting ESP32 mode")
        print("  Requirements:")
        print("  - ESP32 with CSI firmware")
        print("  - USB serial connection")
        print("  - ESP32-CSI-Tool: https://github.com/StevenMHernandez/ESP32-CSI-Tool")

        try:
            import serial

            # Try to open serial connection
            # Adjust COM port as needed
            self.serial_port = serial.Serial('COM3', 115200, timeout=1)
            print(f"[CSI Collector] Connected to ESP32 on COM3")

            self.is_collecting = True

            # Start collection thread
            esp_thread = threading.Thread(target=self._esp32_worker, daemon=True)
            esp_thread.start()

            return True

        except Exception as e:
            print(f"[ERROR] Failed to connect to ESP32: {e}")
            print("  Make sure ESP32 is connected and COM port is correct")
            return False

    def _esp32_worker(self):
        """Worker thread for ESP32 CSI collection."""
        try:
            while self.is_collecting:
                # Read CSI data from serial
                line = self.serial_port.readline().decode('utf-8').strip()

                if line.startswith('CSI_DATA'):
                    csi_data = self._parse_esp32_format(line)
                    if csi_data is not None and not self.data_queue.full():
                        self.data_queue.put(csi_data)

        except Exception as e:
            print(f"[ERROR] ESP32 collection failed: {e}")
            self.is_collecting = False

    def _parse_esp32_format(self, line: str) -> Optional[np.ndarray]:
        """Parse ESP32 CSI data format."""
        # Placeholder - actual parsing depends on ESP32 firmware output
        return None

    # ==================== ATHEROS MODE ====================
    def _start_atheros(self) -> bool:
        """
        Atheros WiFi cards with Atheros CSI Tool.
        Requires: https://github.com/xieyaxiongfly/Atheros-CSI-Tool
        """
        print("[CSI Collector] Starting Atheros mode")
        print("  Requirements:")
        print("  - Atheros WiFi card (AR9xxx series)")
        print("  - Atheros CSI Tool")

        print("[ERROR] Atheros mode not fully implemented yet.")
        print("  Please use simulation mode or contribute Atheros support!")

        return False


if __name__ == "__main__":
    # Test the collector
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    collector = CSICollector(config)

    if collector.start_collection():
        print("Collecting CSI data for 5 seconds...")
        time.sleep(5)

        # Get some samples
        for i in range(10):
            data = collector.get_csi_data()
            if data is not None:
                print(f"Sample {i+1}: Shape {data.shape}, "
                      f"Amplitude range [{data[:,:,0].min():.1f}, {data[:,:,0].max():.1f}]")

        collector.stop_collection()
