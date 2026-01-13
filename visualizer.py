"""
Visualization Module
Displays the 3D reconstructed scene and real-time CSI data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import open3d as o3d
from typing import List, Optional
import time
from reconstruction_3d import Room3D, Reconstructor3D


class Visualizer3D:
    """
    3D scene visualizer using Open3D.
    Shows the room and detected objects in 3D.
    """

    def __init__(self, config: dict):
        self.config = config
        self.room_dimensions = config['room']['dimensions']

        # Open3D visualizer
        self.vis = None
        self.initialized = False

        print(f"[Visualizer3D] Initialized")

    def initialize(self):
        """Initialize the Open3D visualizer window."""
        if self.initialized:
            return

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="WiFi Room Sensing - 3D View",
            width=1280,
            height=720
        )

        # Set up camera
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.5)

        self.initialized = True
        print("[Visualizer3D] Window created")

    def update_scene(self, room: Room3D):
        """
        Update the 3D scene with new data.

        Args:
            room: Room3D object containing all geometries
        """
        if not self.initialized:
            self.initialize()

        # Clear existing geometries
        self.vis.clear_geometries()

        # Add all geometries
        geometries = room.get_all_geometries()
        for geom in geometries:
            self.vis.add_geometry(geom, reset_bounding_box=False)

        # Update view
        self.vis.poll_events()
        self.vis.update_renderer()

    def render_frame(self):
        """Render a single frame."""
        if self.vis:
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        """Close the visualization window."""
        if self.vis:
            self.vis.destroy_window()
            self.initialized = False

    def capture_screenshot(self, filepath: str):
        """Save a screenshot of the current view."""
        if self.vis:
            self.vis.capture_screen_image(filepath)
            print(f"[Visualizer3D] Screenshot saved to {filepath}")


class CSIVisualizer:
    """
    Real-time CSI data visualizer using matplotlib.
    Shows amplitude and phase heatmaps.
    """

    def __init__(self, config: dict):
        self.config = config
        self.num_antennas = config['hardware']['num_antennas']
        self.num_subcarriers = config['hardware']['num_subcarriers']

        # Create figure
        self.fig = None
        self.axes = None
        self.initialized = False

        print(f"[CSI Visualizer] Initialized")

    def initialize(self):
        """Initialize matplotlib figure."""
        if self.initialized:
            return

        self.fig, self.axes = plt.subplots(2, self.num_antennas, figsize=(15, 8))
        self.fig.suptitle('WiFi CSI Real-Time Data', fontsize=16)

        # Set up subplots
        for ant in range(self.num_antennas):
            # Amplitude plot
            self.axes[0, ant].set_title(f'Antenna {ant+1} - Amplitude')
            self.axes[0, ant].set_xlabel('Subcarrier')
            self.axes[0, ant].set_ylabel('Time')

            # Phase plot
            self.axes[1, ant].set_title(f'Antenna {ant+1} - Phase')
            self.axes[1, ant].set_xlabel('Subcarrier')
            self.axes[1, ant].set_ylabel('Time')

        plt.tight_layout()
        self.initialized = True

    def update_plot(self, csi_buffer: List[np.ndarray]):
        """
        Update CSI plots with new data.

        Args:
            csi_buffer: List of recent CSI samples
        """
        if not self.initialized:
            self.initialize()

        if len(csi_buffer) == 0:
            return

        # Convert buffer to array (time, antennas, subcarriers, 2)
        csi_data = np.array(csi_buffer)

        # Plot for each antenna
        for ant in range(self.num_antennas):
            # Amplitude heatmap
            amplitude = csi_data[:, ant, :, 0]  # (time, subcarriers)
            self.axes[0, ant].clear()
            im1 = self.axes[0, ant].imshow(
                amplitude,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest'
            )
            self.axes[0, ant].set_title(f'Antenna {ant+1} - Amplitude')
            self.axes[0, ant].set_xlabel('Subcarrier')
            self.axes[0, ant].set_ylabel('Time')

            # Phase heatmap
            phase = csi_data[:, ant, :, 1]  # (time, subcarriers)
            self.axes[1, ant].clear()
            im2 = self.axes[1, ant].imshow(
                phase,
                aspect='auto',
                cmap='twilight',
                interpolation='nearest',
                vmin=-np.pi,
                vmax=np.pi
            )
            self.axes[1, ant].set_title(f'Antenna {ant+1} - Phase')
            self.axes[1, ant].set_xlabel('Subcarrier')
            self.axes[1, ant].set_ylabel('Time')

        plt.pause(0.001)

    def save_plot(self, filepath: str):
        """Save current plot to file."""
        if self.fig:
            self.fig.savefig(filepath, dpi=150)
            print(f"[CSI Visualizer] Plot saved to {filepath}")

    def close(self):
        """Close matplotlib windows."""
        plt.close('all')


class DashboardVisualizer:
    """
    Comprehensive dashboard showing all system information.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fig = None
        self.axes = None
        self.initialized = False

        print(f"[Dashboard] Initialized")

    def initialize(self):
        """Initialize dashboard figure."""
        if self.initialized:
            return

        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Create subplots
        self.ax_detections = self.fig.add_subplot(gs[0, :])  # Top: Detection timeline
        self.ax_confidence = self.fig.add_subplot(gs[1, 0])   # Middle-left: Confidence
        self.ax_locations = self.fig.add_subplot(gs[1, 1:])   # Middle-right: 2D location map
        self.ax_stats = self.fig.add_subplot(gs[2, 0])        # Bottom-left: Statistics
        self.ax_csi = self.fig.add_subplot(gs[2, 1:])         # Bottom-right: CSI preview

        self.fig.suptitle('WiFi Room Sensing Dashboard', fontsize=18, fontweight='bold')

        self.initialized = True

    def update(self, detected_objects: List[str],
              locations: List[tuple],
              confidences: List[float],
              csi_buffer: Optional[List[np.ndarray]] = None):
        """
        Update dashboard with new data.

        Args:
            detected_objects: List of detected object names
            locations: List of 3D positions
            confidences: List of confidence scores
            csi_buffer: Recent CSI samples (optional)
        """
        if not self.initialized:
            self.initialize()

        # Clear all axes
        for ax in [self.ax_detections, self.ax_confidence, self.ax_locations,
                  self.ax_stats, self.ax_csi]:
            ax.clear()

        # 1. Detection timeline (bar chart)
        if detected_objects:
            self.ax_detections.barh(detected_objects, confidences, color='skyblue')
            self.ax_detections.set_xlabel('Confidence')
            self.ax_detections.set_title('Detected Objects')
            self.ax_detections.set_xlim(0, 1)
        else:
            self.ax_detections.text(0.5, 0.5, 'No objects detected',
                                   ha='center', va='center', fontsize=12)
            self.ax_detections.set_title('Detected Objects')

        # 2. Confidence distribution (pie chart)
        if detected_objects:
            self.ax_confidence.pie(confidences, labels=detected_objects,
                                  autopct='%1.1f%%', startangle=90)
            self.ax_confidence.set_title('Detection Distribution')
        else:
            self.ax_confidence.text(0.5, 0.5, 'No data',
                                   ha='center', va='center')
            self.ax_confidence.set_title('Detection Distribution')

        # 3. 2D location map (top-down view)
        room_dims = self.config['room']['dimensions']
        self.ax_locations.set_xlim(0, room_dims[0])
        self.ax_locations.set_ylim(0, room_dims[1])
        self.ax_locations.set_aspect('equal')
        self.ax_locations.set_xlabel('X (meters)')
        self.ax_locations.set_ylabel('Y (meters)')
        self.ax_locations.set_title('Room Map (Top View)')
        self.ax_locations.grid(True, alpha=0.3)

        # Draw room boundary
        self.ax_locations.plot([0, room_dims[0], room_dims[0], 0, 0],
                              [0, 0, room_dims[1], room_dims[1], 0],
                              'k-', linewidth=2)

        # Plot object locations
        if locations:
            colors = plt.cm.tab10(np.linspace(0, 1, len(locations)))
            for i, (obj, loc, color) in enumerate(zip(detected_objects, locations, colors)):
                x, y, z = loc
                self.ax_locations.scatter(x, y, s=200, c=[color], alpha=0.7,
                                        edgecolors='black', linewidth=2)
                self.ax_locations.annotate(obj, (x, y), xytext=(5, 5),
                                          textcoords='offset points', fontsize=9)

        # 4. Statistics
        stats_text = f"System Statistics\n\n"
        stats_text += f"Objects Detected: {len(detected_objects)}\n"
        if confidences:
            stats_text += f"Avg Confidence: {np.mean(confidences):.2f}\n"
            stats_text += f"Max Confidence: {np.max(confidences):.2f}\n"
            stats_text += f"Min Confidence: {np.min(confidences):.2f}\n"
        stats_text += f"\nRoom Size: {room_dims[0]}m × {room_dims[1]}m × {room_dims[2]}m"

        self.ax_stats.text(0.1, 0.5, stats_text, fontsize=11,
                          verticalalignment='center', family='monospace')
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Statistics')

        # 5. CSI preview (if available)
        if csi_buffer and len(csi_buffer) > 0:
            csi_data = np.array(csi_buffer)
            # Show average amplitude across all antennas
            avg_amplitude = np.mean(csi_data[:, :, :, 0], axis=1)  # (time, subcarriers)
            im = self.ax_csi.imshow(avg_amplitude, aspect='auto', cmap='viridis')
            self.ax_csi.set_xlabel('Subcarrier')
            self.ax_csi.set_ylabel('Time')
            self.ax_csi.set_title('CSI Amplitude Preview')
            plt.colorbar(im, ax=self.ax_csi, label='Amplitude (dB)')
        else:
            self.ax_csi.text(0.5, 0.5, 'CSI data not available',
                           ha='center', va='center')
            self.ax_csi.set_title('CSI Amplitude Preview')

        plt.pause(0.001)

    def save(self, filepath: str):
        """Save dashboard to file."""
        if self.fig:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"[Dashboard] Saved to {filepath}")

    def close(self):
        """Close dashboard."""
        plt.close(self.fig)


if __name__ == "__main__":
    # Test visualizers
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Testing visualizers...")

    # Test dashboard
    dashboard = DashboardVisualizer(config)

    test_objects = ['person', 'chair', 'table']
    test_locations = [(2.5, 2.0, 0.8), (1.5, 1.5, 0.5), (3.0, 2.5, 0.4)]
    test_confidences = [0.85, 0.72, 0.91]

    dashboard.update(test_objects, test_locations, test_confidences)

    print("Dashboard displayed. Close window to continue...")
    plt.show()

    print("\nVisualizer test PASSED!")
