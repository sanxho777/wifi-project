"""
3D Reconstruction Module
Creates 3D models from detected objects and their locations.
"""

import numpy as np
from typing import List, Tuple, Dict
import open3d as o3d


class Object3D:
    """Represents a 3D object in the room."""

    def __init__(self, name: str, position: Tuple[float, float, float],
                 size: Tuple[float, float, float] = None):
        self.name = name
        self.position = position  # (x, y, z)
        self.size = size or self._get_default_size(name)
        self.mesh = None
        self.point_cloud = None

    def _get_default_size(self, name: str) -> Tuple[float, float, float]:
        """Get default size for common objects (width, depth, height)."""
        default_sizes = {
            'person': (0.5, 0.3, 1.7),
            'chair': (0.5, 0.5, 1.0),
            'table': (1.5, 0.8, 0.75),
            'sofa': (2.0, 0.9, 0.85),
            'bed': (2.0, 1.5, 0.6),
            'tv': (1.2, 0.1, 0.7),
            'plant': (0.3, 0.3, 0.8),
            'box': (0.5, 0.5, 0.5),
            'door': (0.9, 0.1, 2.1),
            'window': (1.2, 0.1, 1.5),
        }
        return default_sizes.get(name, (0.5, 0.5, 0.5))

    def create_mesh(self, color=None) -> o3d.geometry.TriangleMesh:
        """Create a 3D mesh for this object."""
        if color is None:
            color = self._get_default_color()

        # Create box mesh
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=self.size[0],
            height=self.size[2],
            depth=self.size[1]
        )

        # Position the mesh
        mesh.translate(self.position)

        # Center the mesh on its position
        mesh.translate([-self.size[0]/2, -self.size[2]/2, -self.size[1]/2])

        # Color
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()

        self.mesh = mesh
        return mesh

    def _get_default_color(self) -> List[float]:
        """Get default color for object type (RGB, 0-1 range)."""
        colors = {
            'person': [0.8, 0.3, 0.3],    # Red
            'chair': [0.6, 0.4, 0.2],     # Brown
            'table': [0.5, 0.35, 0.25],   # Dark brown
            'sofa': [0.3, 0.5, 0.7],      # Blue
            'bed': [0.7, 0.7, 0.9],       # Light blue
            'tv': [0.2, 0.2, 0.2],        # Dark gray
            'plant': [0.2, 0.7, 0.2],     # Green
            'box': [0.8, 0.7, 0.5],       # Beige
            'door': [0.4, 0.3, 0.2],      # Wood
            'window': [0.7, 0.85, 0.95],  # Light blue (glass)
        }
        return colors.get(self.name, [0.5, 0.5, 0.5])


class Room3D:
    """Represents the 3D room environment."""

    def __init__(self, dimensions: Tuple[float, float, float]):
        self.dimensions = dimensions  # (length, width, height)
        self.objects = []
        self.room_mesh = None
        self.floor_mesh = None

    def create_room_structure(self):
        """Create 3D meshes for room walls and floor."""
        length, width, height = self.dimensions

        # Create floor
        floor = o3d.geometry.TriangleMesh.create_box(
            width=length,
            height=0.01,
            depth=width
        )
        floor.paint_uniform_color([0.8, 0.8, 0.8])  # Gray floor
        floor.compute_vertex_normals()
        self.floor_mesh = floor

        # Create walls (wireframe)
        walls = []

        # Create line set for room boundaries
        points = [
            [0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0],  # Floor
            [0, 0, height], [length, 0, height], [length, width, height], [0, width, height]  # Ceiling
        ]

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Floor edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Ceiling edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.3, 0.3, 0.3])  # Dark gray

        self.room_mesh = line_set

    def add_object(self, obj: Object3D):
        """Add an object to the room."""
        self.objects.append(obj)

    def get_all_geometries(self) -> List:
        """Get all 3D geometries for visualization."""
        geometries = []

        # Add room structure
        if self.floor_mesh:
            geometries.append(self.floor_mesh)
        if self.room_mesh:
            geometries.append(self.room_mesh)

        # Add objects
        for obj in self.objects:
            if obj.mesh is None:
                obj.create_mesh()
            geometries.append(obj.mesh)

        return geometries


class Reconstructor3D:
    """
    Main 3D reconstruction engine.
    Converts detected objects into a 3D scene.
    """

    def __init__(self, config: dict):
        self.config = config
        self.room_dimensions = config['room']['dimensions']
        self.voxel_size = config['reconstruction']['voxel_size']
        self.method = config['reconstruction']['method']

        # Create room
        self.room = Room3D(self.room_dimensions)
        self.room.create_room_structure()

        print(f"[3D Reconstructor] Initialized")
        print(f"  Room dimensions: {self.room_dimensions}")
        print(f"  Reconstruction method: {self.method}")

    def reconstruct(self, detected_objects: List[str],
                   locations: List[Tuple[float, float, float]]) -> Room3D:
        """
        Reconstruct 3D scene from detections.

        Args:
            detected_objects: List of object names
            locations: List of 3D positions (x, y, z)

        Returns:
            Room3D object with all geometries
        """
        # Clear existing objects
        self.room.objects = []

        # Create 3D objects
        for obj_name, position in zip(detected_objects, locations):
            obj_3d = Object3D(obj_name, position)
            self.room.add_object(obj_3d)

        print(f"[3D Reconstructor] Reconstructed {len(detected_objects)} objects")

        return self.room

    def create_point_cloud(self, detected_objects: List[str],
                          locations: List[Tuple[float, float, float]]) -> o3d.geometry.PointCloud:
        """
        Create a point cloud representation.

        Args:
            detected_objects: List of object names
            locations: List of 3D positions

        Returns:
            Open3D point cloud
        """
        points = []
        colors = []

        for obj_name, position in zip(detected_objects, locations):
            # Generate points around object location
            num_points = 100

            obj_3d = Object3D(obj_name, position)
            color = obj_3d._get_default_color()

            # Generate random points around object center
            for _ in range(num_points):
                offset = np.random.randn(3) * 0.1  # Small random offset
                point = np.array(position) + offset
                points.append(point)
                colors.append(color)

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd

    def create_voxel_grid(self, detected_objects: List[str],
                         locations: List[Tuple[float, float, float]]) -> o3d.geometry.VoxelGrid:
        """
        Create a voxel grid representation.

        Args:
            detected_objects: List of object names
            locations: List of 3D positions

        Returns:
            Open3D voxel grid
        """
        # First create point cloud
        pcd = self.create_point_cloud(detected_objects, locations)

        # Convert to voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=self.voxel_size
        )

        return voxel_grid

    def export_scene(self, filepath: str, format: str = 'ply'):
        """
        Export 3D scene to file.

        Args:
            filepath: Output file path
            format: File format ('ply', 'obj', 'stl')
        """
        geometries = self.room.get_all_geometries()

        if not geometries:
            print("[3D Reconstructor] No geometries to export")
            return

        # Combine all meshes
        combined_mesh = o3d.geometry.TriangleMesh()

        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                combined_mesh += geom

        # Save
        if format == 'ply':
            o3d.io.write_triangle_mesh(filepath, combined_mesh)
        elif format == 'obj':
            o3d.io.write_triangle_mesh(filepath, combined_mesh)
        elif format == 'stl':
            o3d.io.write_triangle_mesh(filepath, combined_mesh)

        print(f"[3D Reconstructor] Exported scene to {filepath}")

    def get_scene_summary(self) -> Dict:
        """Get summary of the reconstructed scene."""
        return {
            'room_dimensions': self.room_dimensions,
            'num_objects': len(self.room.objects),
            'objects': [(obj.name, obj.position) for obj in self.room.objects]
        }


if __name__ == "__main__":
    # Test the reconstructor
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Testing 3D reconstructor...")

    reconstructor = Reconstructor3D(config)

    # Test with some dummy detections
    test_objects = ['person', 'chair', 'table']
    test_locations = [(2.5, 2.0, 0.8), (1.5, 1.5, 0.5), (3.0, 2.5, 0.4)]

    room = reconstructor.reconstruct(test_objects, test_locations)

    print(f"\nScene summary:")
    summary = reconstructor.get_scene_summary()
    print(f"  Room: {summary['room_dimensions']}")
    print(f"  Objects: {summary['num_objects']}")
    for name, pos in summary['objects']:
        print(f"    - {name} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    print("\n3D reconstructor test PASSED!")
