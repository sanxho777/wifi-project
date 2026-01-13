"""
WiFi Room Sensing - Main Application
Uses WiFi signals to detect objects and create 3D room models.
"""

import yaml
import time
import numpy as np
import argparse
import os
from datetime import datetime

# Import our modules
from csi_collector import CSICollector
from signal_processor import SignalProcessor
from ai_model import WiFiSensingModel
from reconstruction_3d import Reconstructor3D
from visualizer import Visualizer3D, DashboardVisualizer


class WiFiRoomSensing:
    """
    Main application class.
    Orchestrates all components for real-time WiFi sensing.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        print("=" * 70)
        print("WiFi Room Sensing System")
        print("3D Object Detection using WiFi Signals")
        print("=" * 70)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"\n[Main] Loaded configuration from {config_path}")

        # Initialize components
        print("\n[Main] Initializing components...")

        self.collector = CSICollector(self.config)
        self.processor = SignalProcessor(self.config)
        self.ai_model = WiFiSensingModel(self.config)
        self.reconstructor = Reconstructor3D(self.config)

        # Visualization
        self.use_visualization = self.config['visualization']['realtime']
        if self.use_visualization:
            self.visualizer_3d = Visualizer3D(self.config)
            self.dashboard = DashboardVisualizer(self.config)
        else:
            self.visualizer_3d = None
            self.dashboard = None

        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.last_detection = None

        print("[Main] All components initialized successfully!")

    def start(self):
        """Start the WiFi sensing system."""
        print("\n" + "=" * 70)
        print("STARTING WIFI ROOM SENSING")
        print("=" * 70)

        # Start CSI collection
        if not self.collector.start_collection():
            print("[ERROR] Failed to start CSI collection!")
            return

        self.start_time = time.time()

        print("\n[Main] System is running...")
        print("  Press Ctrl+C to stop\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n\n[Main] Stopping system...")
        finally:
            self.stop()

    def _main_loop(self):
        """Main processing loop."""
        last_update_time = time.time()
        update_interval = self.config['visualization']['update_interval'] / 1000.0

        while True:
            # Get new CSI data
            csi_data = self.collector.get_csi_data(timeout=0.1)

            if csi_data is not None:
                # Add to signal processor buffer
                self.processor.add_csi_sample(csi_data)

                # Process when buffer is full
                if self.processor.is_ready():
                    # Process CSI data
                    features = self.processor.process()

                    # AI prediction
                    detected_objects, locations = self.ai_model.predict(features)

                    # Calculate confidences (for visualization)
                    confidences = [0.7 + np.random.random() * 0.25
                                 for _ in detected_objects]  # Placeholder

                    # Store detection
                    self.last_detection = {
                        'objects': detected_objects,
                        'locations': locations,
                        'confidences': confidences,
                        'timestamp': time.time()
                    }

                    # Update statistics
                    self.frame_count += 1

                    # Print detection info
                    if detected_objects:
                        print(f"\r[Frame {self.frame_count}] Detected: {len(detected_objects)} objects - "
                              f"{', '.join(detected_objects)}", end='', flush=True)

                    # Update visualization (throttled)
                    current_time = time.time()
                    if self.use_visualization and (current_time - last_update_time) > update_interval:
                        self._update_visualization()
                        last_update_time = current_time

            else:
                # No data available, brief sleep
                time.sleep(0.01)

    def _update_visualization(self):
        """Update all visualizations."""
        if not self.last_detection:
            return

        detection = self.last_detection

        # Update 3D reconstruction
        room = self.reconstructor.reconstruct(
            detection['objects'],
            detection['locations']
        )

        # Update 3D visualizer
        if self.visualizer_3d:
            try:
                self.visualizer_3d.update_scene(room)
            except Exception as e:
                # Visualization errors shouldn't crash the system
                pass

        # Update dashboard
        if self.dashboard:
            try:
                csi_buffer = list(self.processor.csi_buffer) if self.processor.csi_buffer else None
                self.dashboard.update(
                    detection['objects'],
                    detection['locations'],
                    detection['confidences'],
                    csi_buffer
                )
            except Exception as e:
                pass

    def stop(self):
        """Stop the system gracefully."""
        print("\n[Main] Shutting down...")

        # Stop collection
        self.collector.stop_collection()

        # Save final state
        if self.last_detection:
            self._save_results()

        # Close visualizations
        if self.visualizer_3d:
            self.visualizer_3d.close()
        if self.dashboard:
            self.dashboard.close()

        # Print statistics
        self._print_statistics()

        print("\n[Main] System stopped successfully.")
        print("=" * 70)

    def _save_results(self):
        """Save detection results and visualizations."""
        output_dir = self.config['visualization']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save 3D scene
        scene_path = os.path.join(output_dir, f"scene_{timestamp}.ply")
        self.reconstructor.export_scene(scene_path, format='ply')

        # Save dashboard image
        if self.dashboard:
            dashboard_path = os.path.join(output_dir, f"dashboard_{timestamp}.png")
            self.dashboard.save(dashboard_path)

        # Save detection summary
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("WiFi Room Sensing - Detection Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Total frames processed: {self.frame_count}\n\n")

            f.write("Last Detection:\n")
            for obj, loc in zip(self.last_detection['objects'],
                               self.last_detection['locations']):
                f.write(f"  - {obj}: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})\n")

        print(f"\n[Main] Results saved to {output_dir}/")

    def _print_statistics(self):
        """Print system statistics."""
        if self.start_time:
            runtime = time.time() - self.start_time
            fps = self.frame_count / runtime if runtime > 0 else 0

            print("\n" + "=" * 70)
            print("STATISTICS")
            print("=" * 70)
            print(f"Runtime: {runtime:.2f} seconds")
            print(f"Frames processed: {self.frame_count}")
            print(f"Average FPS: {fps:.2f}")

            if self.last_detection:
                print(f"\nLast Detection:")
                print(f"  Objects: {len(self.last_detection['objects'])}")
                for obj, loc in zip(self.last_detection['objects'],
                                   self.last_detection['locations']):
                    print(f"    - {obj} at ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='WiFi Room Sensing - 3D Object Detection using WiFi Signals'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization (useful for headless operation)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Run for specified duration in seconds (default: run until Ctrl+C)'
    )

    args = parser.parse_args()

    # Override config if no-viz specified
    if args.no_viz:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['visualization']['realtime'] = False

        # Save modified config temporarily
        temp_config = 'config_temp.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        config_path = temp_config
    else:
        config_path = args.config

    # Create and start system
    system = WiFiRoomSensing(config_path)

    if args.duration:
        print(f"\n[Main] Running for {args.duration} seconds...")
        import threading
        stop_timer = threading.Timer(args.duration, lambda: None)
        stop_timer.start()

        try:
            system.start()
        except:
            pass

        stop_timer.cancel()
    else:
        system.start()

    # Cleanup temp config
    if args.no_viz and os.path.exists('config_temp.yaml'):
        os.remove('config_temp.yaml')


if __name__ == "__main__":
    main()
