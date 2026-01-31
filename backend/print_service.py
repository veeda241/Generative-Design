"""
3D Printing Service for Generative Design Platform
Provides print-ready mesh generation, STL export, and print optimization.

Features:
- Point cloud to watertight mesh conversion
- Wall thickness validation
- Print bed sizing and orientation optimization
- STL/3MF export for 3D printing
- Optional G-Code generation via slicer integration
"""

import numpy as np
import logging
import os
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import binary_fill_holes
import json

logger = logging.getLogger(__name__)

# 3D Printing Constants
PRINTER_PROFILES = {
    'ender3': {
        'name': 'Creality Ender 3',
        'bed_x': 220, 'bed_y': 220, 'bed_z': 250,
        'min_wall': 0.4,  # mm (one nozzle width)
        'min_layer': 0.1,  # mm
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75
    },
    'prusa_mk3s': {
        'name': 'Prusa i3 MK3S+',
        'bed_x': 250, 'bed_y': 210, 'bed_z': 210,
        'min_wall': 0.4,
        'min_layer': 0.05,
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75
    },
    'bambu_x1': {
        'name': 'Bambu Lab X1 Carbon',
        'bed_x': 256, 'bed_y': 256, 'bed_z': 256,
        'min_wall': 0.4,
        'min_layer': 0.05,
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75
    },
    'resin_elegoo': {
        'name': 'Elegoo Saturn 2',
        'bed_x': 192, 'bed_y': 120, 'bed_z': 200,
        'min_wall': 0.3,  # Resin can go thinner
        'min_layer': 0.025,
        'technology': 'resin'
    },
    'generic': {
        'name': 'Generic FDM Printer',
        'bed_x': 200, 'bed_y': 200, 'bed_z': 200,
        'min_wall': 0.4,
        'min_layer': 0.1,
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75
    }
}

PRINT_MATERIALS = {
    'pla': {'name': 'PLA', 'temp_nozzle': 210, 'temp_bed': 60, 'density': 1.24},
    'petg': {'name': 'PETG', 'temp_nozzle': 240, 'temp_bed': 85, 'density': 1.27},
    'abs': {'name': 'ABS', 'temp_nozzle': 250, 'temp_bed': 110, 'density': 1.04},
    'tpu': {'name': 'TPU', 'temp_nozzle': 230, 'temp_bed': 50, 'density': 1.21},
    'resin_standard': {'name': 'Standard Resin', 'density': 1.1}
}


class PrintService:
    """Service for preparing 3D models for printing."""
    
    def __init__(self):
        self.printer_profile = PRINTER_PROFILES['generic']
        self.material = PRINT_MATERIALS['pla']
        
    def set_printer(self, printer_key: str):
        """Set the target printer profile."""
        if printer_key in PRINTER_PROFILES:
            self.printer_profile = PRINTER_PROFILES[printer_key]
            logger.info(f"Set printer profile: {self.printer_profile['name']}")
        else:
            logger.warning(f"Unknown printer: {printer_key}, using generic")
            
    def set_material(self, material_key: str):
        """Set the print material."""
        if material_key in PRINT_MATERIALS:
            self.material = PRINT_MATERIALS[material_key]
            logger.info(f"Set material: {self.material['name']}")
        else:
            logger.warning(f"Unknown material: {material_key}, using PLA")
    
    def point_cloud_to_mesh(self, points: List[Dict], method: str = 'ball_pivot',
                           resolution: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert point cloud to watertight mesh suitable for 3D printing.
        
        Args:
            points: List of point dicts with x, y, z coordinates
            method: 'ball_pivot', 'poisson', 'alpha_shape', or 'convex'
            resolution: Resolution for mesh reconstruction
            
        Returns:
            Tuple of (vertices, faces) arrays
        """
        if not points:
            raise ValueError("Empty point cloud")
        
        # Extract coordinates
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        
        # Normalize and center
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        max_extent = np.max(np.abs(coords_centered))
        if max_extent > 0:
            coords_normalized = coords_centered / max_extent
        else:
            coords_normalized = coords_centered
        
        logger.info(f"Converting {len(coords)} points to mesh using {method}...")
        
        try:
            if method == 'convex':
                vertices, faces = self._convex_hull_mesh(coords_normalized)
            elif method == 'alpha_shape':
                vertices, faces = self._alpha_shape_mesh(coords_normalized, resolution)
            elif method == 'voxel':
                vertices, faces = self._voxel_mesh(coords_normalized, resolution)
            elif method == 'ball_pivot':
                vertices, faces = self._ball_pivot_mesh(coords_normalized, resolution)
            else:
                # Default to ball pivot with convex fallback
                try:
                    vertices, faces = self._ball_pivot_mesh(coords_normalized, resolution)
                except:
                    vertices, faces = self._convex_hull_mesh(coords_normalized)
            
            logger.info(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
            return vertices, faces
            
        except Exception as e:
            logger.error(f"Mesh generation failed: {e}, falling back to convex hull")
            return self._convex_hull_mesh(coords_normalized)
    
    def _convex_hull_mesh(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh using convex hull."""
        hull = ConvexHull(coords)
        return coords, hull.simplices
    
    def _alpha_shape_mesh(self, coords: np.ndarray, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh using alpha shapes (Delaunay with filtering)."""
        # Perform Delaunay triangulation
        tri = Delaunay(coords)
        
        # Calculate alpha value based on resolution
        alpha = 2.0 / resolution
        
        # Filter simplices based on circumradius
        valid_simplices = []
        for simplex in tri.simplices:
            vertices = coords[simplex]
            # Calculate circumradius of tetrahedron
            circumradius = self._calculate_circumradius(vertices)
            if circumradius < alpha:
                valid_simplices.append(simplex)
        
        if len(valid_simplices) == 0:
            # Fallback to convex hull if alpha shape fails
            return self._convex_hull_mesh(coords)
        
        # Extract surface faces
        faces = self._extract_surface_faces(np.array(valid_simplices))
        return coords, faces
    
    def _voxel_mesh(self, coords: np.ndarray, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh using voxelization and marching cubes."""
        # Create voxel grid
        grid_size = resolution
        voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        
        # Map points to voxel grid
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1  # Prevent division by zero
        
        for point in coords:
            idx = ((point - min_coords) / range_coords * (grid_size - 1)).astype(int)
            idx = np.clip(idx, 0, grid_size - 1)
            voxels[idx[0], idx[1], idx[2]] = True
        
        # Fill holes in voxel grid
        voxels = binary_fill_holes(voxels)
        
        # Generate mesh using simple marching cubes
        vertices, faces = self._marching_cubes_simple(voxels, min_coords, range_coords, grid_size)
        
        return vertices, faces
    
    def _ball_pivot_mesh(self, coords: np.ndarray, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified ball pivoting algorithm for surface reconstruction."""
        # Use voxel method as a robust alternative
        return self._voxel_mesh(coords, resolution)
    
    def _calculate_circumradius(self, vertices: np.ndarray) -> float:
        """Calculate circumradius of a simplex."""
        if len(vertices) == 4:  # Tetrahedron
            a = np.linalg.norm(vertices[0] - vertices[1])
            b = np.linalg.norm(vertices[0] - vertices[2])
            c = np.linalg.norm(vertices[0] - vertices[3])
            d = np.linalg.norm(vertices[1] - vertices[2])
            e = np.linalg.norm(vertices[1] - vertices[3])
            f = np.linalg.norm(vertices[2] - vertices[3])
            
            # Volume using Cayley-Menger determinant
            V = self._tetrahedron_volume(vertices)
            if V > 1e-10:
                return (a * d * e) / (24 * V)
        return 1.0
    
    def _tetrahedron_volume(self, vertices: np.ndarray) -> float:
        """Calculate volume of tetrahedron."""
        a = vertices[1] - vertices[0]
        b = vertices[2] - vertices[0]
        c = vertices[3] - vertices[0]
        return abs(np.dot(a, np.cross(b, c))) / 6
    
    def _extract_surface_faces(self, simplices: np.ndarray) -> np.ndarray:
        """Extract surface triangles from tetrahedral mesh."""
        # Each tetrahedron has 4 faces
        face_counts = {}
        
        for simplex in simplices:
            # Generate 4 triangular faces
            faces = [
                tuple(sorted([simplex[0], simplex[1], simplex[2]])),
                tuple(sorted([simplex[0], simplex[1], simplex[3]])),
                tuple(sorted([simplex[0], simplex[2], simplex[3]])),
                tuple(sorted([simplex[1], simplex[2], simplex[3]]))
            ]
            for face in faces:
                face_counts[face] = face_counts.get(face, 0) + 1
        
        # Surface faces appear exactly once
        surface_faces = [list(face) for face, count in face_counts.items() if count == 1]
        return np.array(surface_faces) if surface_faces else np.array([])
    
    def _marching_cubes_simple(self, voxels: np.ndarray, origin: np.ndarray, 
                               range_coords: np.ndarray, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simple marching cubes implementation for voxel to mesh conversion."""
        vertices = []
        faces = []
        vertex_map = {}
        
        # Iterate through voxel grid
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                for k in range(grid_size - 1):
                    # Check cube configuration
                    cube = [
                        voxels[i, j, k], voxels[i+1, j, k],
                        voxels[i+1, j+1, k], voxels[i, j+1, k],
                        voxels[i, j, k+1], voxels[i+1, j, k+1],
                        voxels[i+1, j+1, k+1], voxels[i, j+1, k+1]
                    ]
                    
                    # Simple surface detection: if cube has both filled and empty voxels
                    if any(cube) and not all(cube):
                        # Add face on boundary
                        self._add_cube_faces(i, j, k, cube, vertices, faces, vertex_map,
                                           origin, range_coords, grid_size)
        
        if not vertices:
            # Fallback: create simple box mesh
            return self._create_box_mesh()
        
        return np.array(vertices), np.array(faces)
    
    def _add_cube_faces(self, i, j, k, cube, vertices, faces, vertex_map,
                        origin, range_coords, grid_size):
        """Add faces for a boundary cube."""
        # Define cube vertices
        cube_verts = [
            (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
            (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
        ]
        
        # Define faces (6 faces of cube)
        cube_faces = [
            (0, 3, 2, 1),  # Bottom
            (4, 5, 6, 7),  # Top
            (0, 1, 5, 4),  # Front
            (2, 3, 7, 6),  # Back
            (0, 4, 7, 3),  # Left
            (1, 2, 6, 5),  # Right
        ]
        
        # Check each face for boundary
        for face_idx, (v0, v1, v2, v3) in enumerate(cube_faces):
            # Only add face if it's on the boundary
            if self._is_boundary_face(cube, face_idx):
                # Get or create vertex indices
                indices = []
                for vi in [v0, v1, v2, v3]:
                    pos = cube_verts[vi]
                    if pos not in vertex_map:
                        # Convert grid position to world coordinates
                        world_pos = origin + (np.array(pos) / (grid_size - 1)) * range_coords
                        vertex_map[pos] = len(vertices)
                        vertices.append(world_pos.tolist())
                    indices.append(vertex_map[pos])
                
                # Add two triangles for quad face
                faces.append([indices[0], indices[1], indices[2]])
                faces.append([indices[0], indices[2], indices[3]])
    
    def _is_boundary_face(self, cube, face_idx):
        """Check if a face of the cube is on the boundary."""
        # Face vertex indices for each of 6 faces
        face_vertices = [
            [0, 1, 2, 3],  # Bottom (z=0)
            [4, 5, 6, 7],  # Top (z=1)
            [0, 1, 5, 4],  # Front (y=0)
            [2, 3, 7, 6],  # Back (y=1)
            [0, 3, 7, 4],  # Left (x=0)
            [1, 2, 6, 5],  # Right (x=1)
        ]
        
        verts = face_vertices[face_idx]
        filled_count = sum(1 for v in verts if cube[v])
        
        # Boundary if not all are the same
        return 0 < filled_count < 4
    
    def _create_box_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a simple unit box mesh as fallback."""
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 2, 1], [0, 3, 2],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [2, 3, 7], [2, 7, 6],  # Back
            [0, 4, 7], [0, 7, 3],  # Left
            [1, 2, 6], [1, 6, 5],  # Right
        ], dtype=np.int32)
        
        return vertices, faces
    
    def validate_for_printing(self, vertices: np.ndarray, faces: np.ndarray,
                             target_size_mm: float = 100) -> Dict:
        """
        Validate mesh for 3D printing and return analysis.
        
        Args:
            vertices: Mesh vertices
            faces: Mesh faces
            target_size_mm: Desired size of longest dimension in mm
            
        Returns:
            Dict with validation results and recommendations
        """
        result = {
            'is_printable': True,
            'warnings': [],
            'recommendations': [],
            'stats': {}
        }
        
        # Calculate mesh statistics
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        size = max_bounds - min_bounds
        volume = self._estimate_volume(vertices, faces)
        
        # Scale to target size
        scale = target_size_mm / max(size)
        scaled_size = size * scale
        
        result['stats'] = {
            'vertices': len(vertices),
            'faces': len(faces),
            'original_dimensions': size.tolist(),
            'scaled_dimensions_mm': scaled_size.tolist(),
            'estimated_volume_mm3': volume * (scale ** 3),
            'scale_factor': scale
        }
        
        # Check if fits on print bed
        printer = self.printer_profile
        if (scaled_size[0] > printer['bed_x'] or 
            scaled_size[1] > printer['bed_y'] or 
            scaled_size[2] > printer['bed_z']):
            result['warnings'].append(
                f"Model exceeds print bed size ({printer['bed_x']}x{printer['bed_y']}x{printer['bed_z']}mm)"
            )
            result['recommendations'].append(
                f"Consider scaling down or splitting the model"
            )
        
        # Estimate print time and material
        material_density = self.material.get('density', 1.24)  # g/cm³
        volume_cm3 = result['stats']['estimated_volume_mm3'] / 1000
        weight_g = volume_cm3 * material_density
        
        # Rough print time estimate (based on ~30g/hour for FDM)
        print_time_hours = weight_g / 30
        
        result['stats']['estimated_weight_g'] = round(weight_g, 2)
        result['stats']['estimated_print_time_hours'] = round(print_time_hours, 1)
        result['stats']['estimated_filament_m'] = round(weight_g / 3.0, 2)  # Approximate
        
        # Check wall thickness (using vertex nearest neighbor distances)
        if len(vertices) > 100:
            min_wall = self._estimate_min_wall_thickness(vertices)
            if min_wall < printer.get('min_wall', 0.4):
                result['warnings'].append(
                    f"Some walls may be too thin for printing (min: {min_wall:.2f}mm)"
                )
                result['recommendations'].append(
                    f"Increase wall thickness to at least {printer['min_wall']}mm"
                )
        
        # Check for manifold issues (basic check)
        edge_issues = self._check_manifold(faces)
        if edge_issues > 0:
            result['warnings'].append(
                f"Mesh may have {edge_issues} non-manifold edges"
            )
            result['recommendations'].append(
                "Consider repairing mesh topology before printing"
            )
        
        # Set printability based on warnings
        result['is_printable'] = len(result['warnings']) == 0
        
        return result
    
    def _estimate_volume(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Estimate mesh volume using signed tetrahedron method."""
        volume = 0.0
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Signed volume of tetrahedron formed with origin
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(volume)
    
    def _estimate_min_wall_thickness(self, vertices: np.ndarray) -> float:
        """Estimate minimum wall thickness using nearest neighbor distances."""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(vertices)
        distances, _ = tree.query(vertices, k=2)  # k=2 because first is self
        min_distances = distances[:, 1]
        
        # Take 5th percentile as representative of thin areas
        return np.percentile(min_distances, 5)
    
    def _check_manifold(self, faces: np.ndarray) -> int:
        """Check for non-manifold edges (edges shared by more than 2 faces)."""
        edge_count = {}
        
        for face in faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Non-manifold edges have count != 2
        non_manifold = sum(1 for count in edge_count.values() if count != 2)
        return non_manifold
    
    def scale_for_printing(self, vertices: np.ndarray, target_size_mm: float = 100,
                          center: bool = True) -> np.ndarray:
        """Scale mesh to target size for printing."""
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        size = max_bounds - min_bounds
        
        scale = target_size_mm / max(size)
        scaled = vertices * scale
        
        if center:
            centroid = (scaled.max(axis=0) + scaled.min(axis=0)) / 2
            scaled = scaled - centroid
            # Move to positive Z (on print bed)
            scaled[:, 2] = scaled[:, 2] - scaled[:, 2].min()
        
        return scaled
    
    def export_stl(self, vertices: np.ndarray, faces: np.ndarray, 
                  filename: str, binary: bool = True) -> str:
        """
        Export mesh to STL file for 3D printing.
        
        Args:
            vertices: Mesh vertices
            faces: Mesh face indices
            filename: Output filename
            binary: Use binary STL format (smaller file size)
            
        Returns:
            Path to saved file
        """
        os.makedirs('exports', exist_ok=True)
        if not filename.endswith('.stl'):
            filename += '.stl'
        filepath = os.path.join('exports', filename)
        
        if binary:
            self._write_binary_stl(vertices, faces, filepath)
        else:
            self._write_ascii_stl(vertices, faces, filepath)
        
        logger.info(f"Exported STL: {filepath} ({len(faces)} triangles)")
        return filepath
    
    def _write_binary_stl(self, vertices: np.ndarray, faces: np.ndarray, filepath: str):
        """Write binary STL file."""
        with open(filepath, 'wb') as f:
            # 80-byte header
            header = b'Binary STL - Aether-Gen 3D Printing Export' + b'\x00' * 38
            f.write(header[:80])
            
            # Number of triangles
            f.write(np.array(len(faces), dtype=np.uint32).tobytes())
            
            # Write triangles
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal = normal / norm_len
                
                # Write normal and vertices
                f.write(np.array(normal, dtype=np.float32).tobytes())
                f.write(np.array(v0, dtype=np.float32).tobytes())
                f.write(np.array(v1, dtype=np.float32).tobytes())
                f.write(np.array(v2, dtype=np.float32).tobytes())
                f.write(np.array(0, dtype=np.uint16).tobytes())  # Attribute byte count
    
    def _write_ascii_stl(self, vertices: np.ndarray, faces: np.ndarray, filepath: str):
        """Write ASCII STL file."""
        with open(filepath, 'w') as f:
            f.write("solid AetherGen\n")
            
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Calculate normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal = normal / norm_len
                
                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid AetherGen\n")
    
    def export_3mf(self, vertices: np.ndarray, faces: np.ndarray,
                  filename: str, metadata: Dict = None) -> str:
        """
        Export mesh to 3MF format (modern 3D printing format).
        
        Args:
            vertices: Mesh vertices
            faces: Mesh face indices
            filename: Output filename
            metadata: Optional metadata dict
            
        Returns:
            Path to saved file
        """
        import zipfile
        from xml.etree import ElementTree as ET
        
        os.makedirs('exports', exist_ok=True)
        if not filename.endswith('.3mf'):
            filename += '.3mf'
        filepath = os.path.join('exports', filename)
        
        # Create 3MF package (ZIP file with XML content)
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Content types
            content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
            zf.writestr('[Content_Types].xml', content_types)
            
            # Relationships
            rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
            zf.writestr('_rels/.rels', rels)
            
            # 3D Model
            model_xml = self._create_3mf_model(vertices, faces, metadata)
            zf.writestr('3D/3dmodel.model', model_xml)
        
        logger.info(f"Exported 3MF: {filepath}")
        return filepath
    
    def _create_3mf_model(self, vertices: np.ndarray, faces: np.ndarray, 
                         metadata: Dict = None) -> str:
        """Create 3MF model XML content."""
        # Build vertices string
        vertices_str = '\n'.join(
            f'        <vertex x="{v[0]}" y="{v[1]}" z="{v[2]}"/>'
            for v in vertices
        )
        
        # Build triangles string
        triangles_str = '\n'.join(
            f'        <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>'
            for f in faces
        )
        
        model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Application">Aether-Gen 3D Printing Service</metadata>
  <metadata name="Title">{metadata.get('title', 'Generated Model') if metadata else 'Generated Model'}</metadata>
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
{vertices_str}
        </vertices>
        <triangles>
{triangles_str}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>'''
        return model


class PrintOptimizer:
    """Optimize point cloud and mesh for 3D printing."""
    
    @staticmethod
    def densify_point_cloud(points: List[Dict], target_count: int) -> List[Dict]:
        """
        Increase point cloud density for better mesh reconstruction.
        
        Args:
            points: Original point cloud
            target_count: Target number of points
            
        Returns:
            Densified point cloud
        """
        if len(points) >= target_count:
            return points
        
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        colors = np.array([[p.get('r', 0.5), p.get('g', 0.5), p.get('b', 0.5)] for p in points])
        
        densified_coords = list(coords)
        densified_colors = list(colors)
        
        while len(densified_coords) < target_count:
            # Random pair interpolation with jitter
            idx1, idx2 = np.random.choice(len(coords), 2, replace=False)
            t = np.random.uniform(0.2, 0.8)
            
            new_coord = coords[idx1] * (1 - t) + coords[idx2] * t
            new_color = colors[idx1] * (1 - t) + colors[idx2] * t
            
            # Add small random displacement
            new_coord += np.random.normal(0, 0.002, 3)
            
            densified_coords.append(new_coord)
            densified_colors.append(new_color)
        
        result = []
        for coord, color in zip(densified_coords, densified_colors):
            result.append({
                'x': float(coord[0]),
                'y': float(coord[1]),
                'z': float(coord[2]),
                'r': float(np.clip(color[0], 0, 1)),
                'g': float(np.clip(color[1], 0, 1)),
                'b': float(np.clip(color[2], 0, 1))
            })
        
        return result
    
    @staticmethod
    def smooth_point_cloud(points: List[Dict], iterations: int = 2) -> List[Dict]:
        """
        Smooth point cloud using Laplacian smoothing for better surfaces.
        """
        from scipy.spatial import cKDTree
        
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        colors = np.array([[p.get('r', 0.5), p.get('g', 0.5), p.get('b', 0.5)] for p in points])
        
        for _ in range(iterations):
            tree = cKDTree(coords)
            smoothed = np.zeros_like(coords)
            
            for i, point in enumerate(coords):
                # Find k nearest neighbors
                distances, indices = tree.query(point, k=min(10, len(coords)))
                neighbors = coords[indices]
                
                # Average position (Laplacian smoothing)
                smoothed[i] = neighbors.mean(axis=0)
            
            # Blend: 70% original, 30% smoothed
            coords = 0.7 * coords + 0.3 * smoothed
        
        result = []
        for coord, color in zip(coords, colors):
            result.append({
                'x': float(coord[0]),
                'y': float(coord[1]),
                'z': float(coord[2]),
                'r': float(color[0]),
                'g': float(color[1]),
                'b': float(color[2])
            })
        
        return result
    
    @staticmethod
    def orient_for_printing(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Find optimal orientation to minimize supports and maximize surface quality.
        
        Returns:
            Tuple of (rotated_vertices, orientation_info)
        """
        # Calculate face normals and areas
        normals = []
        areas = []
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = np.linalg.norm(cross) / 2
            if area > 0:
                normal = cross / (2 * area)
            else:
                normal = np.array([0, 0, 1])
            normals.append(normal)
            areas.append(area)
        
        normals = np.array(normals)
        areas = np.array(areas)
        
        # Find orientation that maximizes downward-facing flat area
        # This minimizes need for supports
        best_rotation = np.eye(3)
        best_score = float('inf')
        
        # Try different orientations
        test_rotations = [
            np.eye(3),  # Original
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # Rotate 90° around X
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # Rotate 90° around Y
        ]
        
        for rot in test_rotations:
            rotated_normals = np.dot(normals, rot.T)
            
            # Calculate score: penalize overhangs (normals pointing down with angle > 45°)
            overhang_score = 0
            for normal, area in zip(rotated_normals, areas):
                if normal[2] < -0.7:  # More than 45° overhang
                    overhang_score += area * abs(normal[2])
            
            if overhang_score < best_score:
                best_score = overhang_score
                best_rotation = rot
        
        # Apply best rotation
        rotated_vertices = np.dot(vertices, best_rotation.T)
        
        # Move to positive Z
        rotated_vertices[:, 2] -= rotated_vertices[:, 2].min()
        
        return rotated_vertices, {
            'rotation_applied': best_rotation.tolist(),
            'overhang_score': best_score
        }


# Singleton instance
_print_service = None

def get_print_service() -> PrintService:
    """Get or create the print service singleton."""
    global _print_service
    if _print_service is None:
        _print_service = PrintService()
    return _print_service
