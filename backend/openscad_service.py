"""
OpenSCAD-based 3D Model Generation Service
Integrates CADAM-style AI-powered CAD generation using Gemini
Generates OpenSCAD code from text prompts and converts to point clouds
"""

import os
import subprocess
import tempfile
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenSCAD system prompt (adapted from CADAM)
OPENSCAD_SYSTEM_PROMPT = """You are an expert CAD engineer that creates precise OpenSCAD models.
Generate OpenSCAD code for 3D models based on user descriptions.

CRITICAL RULES:
1. Return ONLY raw OpenSCAD code - no markdown, no explanations
2. All models must be manifold (watertight, 3D-printable)
3. Always use parameterized dimensions with variables at the top
4. Use $fn = 64 or higher for smooth curves
5. Connect all parts properly using union(), difference(), intersection()
6. Include helpful comments for parameters
7. Make the model centered at origin when possible

STYLE GUIDELINES:
- Use descriptive variable names
- Group related parameters together
- Use modules for repeated elements
- Ensure proper wall thickness for 3D printing (min 2mm)

Example for "a coffee mug":
// Mug Parameters
cup_height = 100;
cup_radius = 40;
handle_radius = 25;
handle_thickness = 8;
wall_thickness = 3;
$fn = 64;

difference() {
    union() {
        // Main cup body
        cylinder(h=cup_height, r=cup_radius);
        
        // Handle
        translate([cup_radius - 5, 0, cup_height/2])
        rotate([90, 0, 0])
        rotate_extrude(angle=180)
        translate([handle_radius, 0, 0])
        circle(r=handle_thickness/2);
    }
    
    // Hollow interior
    translate([0, 0, wall_thickness])
    cylinder(h=cup_height, r=cup_radius - wall_thickness);
}

Now generate OpenSCAD code for the following:"""


class OpenSCADService:
    """Service for generating 3D models using AI-powered OpenSCAD code generation."""
    
    def __init__(self, openscad_path: Optional[str] = None):
        """
        Initialize the OpenSCAD service.
        
        Args:
            openscad_path: Path to OpenSCAD executable. Auto-detected if None.
        """
        self.openscad_path = openscad_path or self._find_openscad()
        self.gemini_model = None
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key and api_key != 'your_actual_api_key_here':
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                    logger.info("✓ Gemini model initialized for OpenSCAD generation")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}")
        
        if self.openscad_path:
            logger.info(f"✓ OpenSCAD found at: {self.openscad_path}")
        else:
            logger.warning("OpenSCAD not found - will use fallback point cloud generation")
    
    def _find_openscad(self) -> Optional[str]:
        """Auto-detect OpenSCAD installation."""
        # Common installation paths
        paths = [
            r"C:\Program Files\OpenSCAD\openscad.exe",
            r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
            "/usr/bin/openscad",
            "/usr/local/bin/openscad",
            "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(
                ["where" if os.name == 'nt' else "which", "openscad"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        return None
    
    def generate_openscad_code(self, prompt: str) -> str:
        """
        Generate OpenSCAD code from a text prompt using Gemini.
        
        Args:
            prompt: Description of the 3D model to generate
            
        Returns:
            OpenSCAD code string
        """
        if not self.gemini_model:
            logger.warning("Gemini not available, using fallback template")
            return self._fallback_openscad_code(prompt)
        
        try:
            full_prompt = f"{OPENSCAD_SYSTEM_PROMPT}\n\n{prompt}"
            response = self.gemini_model.generate_content(full_prompt)
            code = response.text.strip()
            
            # Clean up any markdown formatting
            code = self._clean_openscad_code(code)
            
            logger.info(f"Generated OpenSCAD code ({len(code)} chars)")
            return code
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return self._fallback_openscad_code(prompt)
    
    def _clean_openscad_code(self, code: str) -> str:
        """Remove markdown formatting from generated code."""
        # Remove markdown code blocks
        code = re.sub(r'^```(?:openscad)?\s*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        code = code.strip()
        return code
    
    def _fallback_openscad_code(self, prompt: str) -> str:
        """Generate fallback OpenSCAD code for common objects."""
        prompt_lower = prompt.lower()
        
        # Basic shape templates
        if any(word in prompt_lower for word in ['cube', 'box', 'block']):
            return """
// Box Parameters
width = 50;
depth = 50;
height = 30;
corner_radius = 3;
$fn = 32;

minkowski() {
    cube([width - corner_radius*2, depth - corner_radius*2, height - corner_radius], center=true);
    sphere(r=corner_radius);
}
"""
        elif any(word in prompt_lower for word in ['sphere', 'ball']):
            return """
// Sphere Parameters
radius = 30;
$fn = 64;

sphere(r=radius);
"""
        elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pipe']):
            return """
// Cylinder Parameters
height = 60;
outer_radius = 25;
inner_radius = 20;
$fn = 64;

difference() {
    cylinder(h=height, r=outer_radius, center=true);
    cylinder(h=height + 1, r=inner_radius, center=true);
}
"""
        elif any(word in prompt_lower for word in ['gear', 'cog']):
            return """
// Gear Parameters
num_teeth = 20;
tooth_height = 8;
tooth_width = 5;
gear_radius = 40;
gear_thickness = 10;
hole_radius = 8;
$fn = 64;

difference() {
    union() {
        // Base cylinder
        cylinder(h=gear_thickness, r=gear_radius, center=true);
        
        // Teeth
        for (i = [0:num_teeth-1]) {
            rotate([0, 0, i * 360/num_teeth])
            translate([gear_radius, 0, 0])
            cube([tooth_height*2, tooth_width, gear_thickness], center=true);
        }
    }
    // Center hole
    cylinder(h=gear_thickness + 1, r=hole_radius, center=true);
}
"""
        elif any(word in prompt_lower for word in ['mug', 'cup']):
            return """
// Mug Parameters
cup_height = 90;
cup_radius = 35;
handle_size = 20;
wall_thickness = 3;
$fn = 64;

difference() {
    union() {
        // Cup body
        cylinder(h=cup_height, r=cup_radius);
        
        // Handle
        translate([cup_radius - 3, 0, cup_height * 0.6])
        rotate([90, 0, 0])
        rotate_extrude(angle=180)
        translate([handle_size, 0, 0])
        circle(r=5);
    }
    
    // Hollow interior
    translate([0, 0, wall_thickness])
    cylinder(h=cup_height, r=cup_radius - wall_thickness);
}
"""
        else:
            # Generic object
            return f"""
// Generated 3D Object: {prompt}
// Parameters
size = 50;
detail = 32;
$fn = detail;

// Main body - customize based on requirements
difference() {{
    // Outer shape
    minkowski() {{
        cube([size, size, size * 0.6], center=true);
        sphere(r=3);
    }}
    
    // Interior cavity
    translate([0, 0, 3])
    minkowski() {{
        cube([size - 6, size - 6, size * 0.6], center=true);
        sphere(r=2);
    }}
}}
"""
    
    def render_to_stl(self, scad_code: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Render OpenSCAD code to STL file.
        
        Args:
            scad_code: OpenSCAD code to render
            output_path: Output STL file path. Auto-generated if None.
            
        Returns:
            Path to generated STL file, or None if failed
        """
        if not self.openscad_path:
            logger.warning("OpenSCAD not available for rendering")
            return None
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False) as scad_file:
                scad_file.write(scad_code)
                scad_path = scad_file.name
            
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.stl')
            
            # Run OpenSCAD
            result = subprocess.run(
                [self.openscad_path, '-o', output_path, scad_path],
                capture_output=True, text=True, timeout=120
            )
            
            # Clean up temp scad file
            os.unlink(scad_path)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"✓ STL rendered: {output_path}")
                return output_path
            else:
                logger.error(f"OpenSCAD error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("OpenSCAD rendering timed out")
            return None
        except Exception as e:
            logger.error(f"STL rendering failed: {e}")
            return None
    
    def stl_to_point_cloud(self, stl_path: str, num_points: int = 8192) -> List[Dict]:
        """
        Convert STL file to point cloud by sampling surface points.
        
        Args:
            stl_path: Path to STL file
            num_points: Number of points to sample
            
        Returns:
            List of point dicts with pos and color
        """
        try:
            from stl import mesh as stl_mesh
            
            # Load STL
            model = stl_mesh.Mesh.from_file(stl_path)
            triangles = model.vectors  # Shape: (n_triangles, 3, 3)
            
            # Calculate triangle areas for weighted sampling
            v0 = triangles[:, 0, :]
            v1 = triangles[:, 1, :]
            v2 = triangles[:, 2, :]
            
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            total_area = np.sum(areas)
            
            if total_area == 0:
                raise ValueError("STL has zero surface area")
            
            probabilities = areas / total_area
            
            # Sample triangles based on area
            triangle_indices = np.random.choice(
                len(triangles), size=num_points, p=probabilities
            )
            
            # Generate random barycentric coordinates
            r1 = np.random.random(num_points)
            r2 = np.random.random(num_points)
            
            # Ensure points are inside triangle
            sqrt_r1 = np.sqrt(r1)
            u = 1 - sqrt_r1
            v = sqrt_r1 * (1 - r2)
            w = sqrt_r1 * r2
            
            # Sample points on triangles
            sampled_v0 = triangles[triangle_indices, 0, :]
            sampled_v1 = triangles[triangle_indices, 1, :]
            sampled_v2 = triangles[triangle_indices, 2, :]
            
            points = (u[:, np.newaxis] * sampled_v0 + 
                     v[:, np.newaxis] * sampled_v1 + 
                     w[:, np.newaxis] * sampled_v2)
            
            # Normalize to unit sphere
            centroid = np.mean(points, axis=0)
            points_centered = points - centroid
            max_dist = np.max(np.linalg.norm(points_centered, axis=1))
            if max_dist > 0:
                points_normalized = points_centered / max_dist
            else:
                points_normalized = points_centered
            
            # Calculate normals for coloring
            normals = cross[triangle_indices]
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
            
            # Color based on normal direction (creates nice shading effect)
            colors = (normals + 1) / 2  # Map from [-1,1] to [0,1]
            
            # Build point list
            point_list = []
            for i in range(len(points_normalized)):
                point_list.append({
                    'pos': points_normalized[i].tolist(),
                    'color': colors[i].tolist()
                })
            
            logger.info(f"✓ Generated {len(point_list)} points from STL")
            return point_list
            
        except ImportError:
            logger.warning("numpy-stl not installed, using fallback")
            return self._generate_fallback_points(num_points)
        except Exception as e:
            logger.error(f"STL to point cloud failed: {e}")
            return self._generate_fallback_points(num_points)
    
    def generate_point_cloud(self, prompt: str, num_points: int = 8192) -> Tuple[List[Dict], str]:
        """
        Generate a point cloud from a text prompt.
        
        Args:
            prompt: Text description of the 3D model
            num_points: Number of points to generate
            
        Returns:
            Tuple of (point_list, openscad_code)
        """
        # Generate OpenSCAD code
        scad_code = self.generate_openscad_code(prompt)
        
        # Try to render with OpenSCAD
        if self.openscad_path:
            stl_path = self.render_to_stl(scad_code)
            if stl_path:
                points = self.stl_to_point_cloud(stl_path, num_points)
                # Clean up temp STL
                try:
                    os.unlink(stl_path)
                except:
                    pass
                return points, scad_code
        
        # Fallback: generate parametric point cloud from code analysis
        points = self._generate_from_code_analysis(scad_code, num_points)
        return points, scad_code
    
    def _generate_from_code_analysis(self, scad_code: str, num_points: int) -> List[Dict]:
        """Generate point cloud by analyzing OpenSCAD code structure."""
        points = []
        
        # Extract dimensions from code
        dimensions = self._extract_dimensions(scad_code)
        
        # Detect shape types
        shapes = self._detect_shapes(scad_code)
        
        points_per_shape = max(100, num_points // max(1, len(shapes)))
        
        for shape, params in shapes:
            shape_points = self._generate_shape_points(shape, params, dimensions, points_per_shape)
            points.extend(shape_points)
        
        # If no shapes detected, generate a default object
        if not points:
            points = self._generate_fallback_points(num_points)
        
        # Normalize and center
        if points:
            points = self._normalize_points(points)
        
        # Trim or pad to exact count
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = [points[i] for i in indices]
        elif len(points) < num_points:
            # Duplicate points to reach target
            while len(points) < num_points:
                idx = np.random.randint(0, len(points))
                # Add slight noise to duplicated point
                new_point = {
                    'pos': [p + np.random.normal(0, 0.01) for p in points[idx]['pos']],
                    'color': points[idx]['color']
                }
                points.append(new_point)
        
        return points
    
    def _extract_dimensions(self, code: str) -> Dict[str, float]:
        """Extract dimension variables from OpenSCAD code."""
        dims = {}
        # Match variable assignments like: width = 50;
        pattern = r'(\w+)\s*=\s*([\d.]+)\s*;'
        for match in re.finditer(pattern, code):
            dims[match.group(1)] = float(match.group(2))
        return dims
    
    def _detect_shapes(self, code: str) -> List[Tuple[str, Dict]]:
        """Detect shapes from OpenSCAD code."""
        shapes = []
        
        # Cylinder
        for match in re.finditer(r'cylinder\s*\(\s*h\s*=\s*([\d.]+)\s*,\s*r\s*=\s*([\d.]+)', code):
            shapes.append(('cylinder', {'h': float(match.group(1)), 'r': float(match.group(2))}))
        
        # Cube
        for match in re.finditer(r'cube\s*\(\s*\[([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', code):
            shapes.append(('cube', {'x': float(match.group(1)), 'y': float(match.group(2)), 'z': float(match.group(3))}))
        
        # Sphere
        for match in re.finditer(r'sphere\s*\(\s*r\s*=\s*([\d.]+)', code):
            shapes.append(('sphere', {'r': float(match.group(1))}))
        
        return shapes if shapes else [('default', {})]
    
    def _generate_shape_points(self, shape: str, params: Dict, dims: Dict, num_points: int) -> List[Dict]:
        """Generate points for a specific shape."""
        points = []
        
        if shape == 'cylinder':
            h = params.get('h', dims.get('height', 50))
            r = params.get('r', dims.get('radius', 25))
            
            for _ in range(num_points):
                # Random point on cylinder surface
                if np.random.random() < 0.8:  # Curved surface
                    theta = np.random.uniform(0, 2 * np.pi)
                    z = np.random.uniform(0, h)
                    x, y = r * np.cos(theta), r * np.sin(theta)
                else:  # Top/bottom caps
                    theta = np.random.uniform(0, 2 * np.pi)
                    rad = np.random.uniform(0, r)
                    x, y = rad * np.cos(theta), rad * np.sin(theta)
                    z = 0 if np.random.random() < 0.5 else h
                
                points.append({
                    'pos': [x / 50, z / 50 - h/100, y / 50],
                    'color': [0.6, 0.7, 0.9]
                })
        
        elif shape == 'cube':
            sx = params.get('x', dims.get('width', 50))
            sy = params.get('y', dims.get('depth', 50))
            sz = params.get('z', dims.get('height', 50))
            
            for _ in range(num_points):
                # Random point on cube surface
                face = np.random.randint(6)
                u, v = np.random.uniform(-0.5, 0.5, 2)
                
                if face == 0: x, y, z = u * sx, v * sy, sz / 2
                elif face == 1: x, y, z = u * sx, v * sy, -sz / 2
                elif face == 2: x, y, z = sx / 2, u * sy, v * sz
                elif face == 3: x, y, z = -sx / 2, u * sy, v * sz
                elif face == 4: x, y, z = u * sx, sy / 2, v * sz
                else: x, y, z = u * sx, -sy / 2, v * sz
                
                points.append({
                    'pos': [x / 50, z / 50, y / 50],
                    'color': [0.8, 0.6, 0.4]
                })
        
        elif shape == 'sphere':
            r = params.get('r', dims.get('radius', 30))
            
            for _ in range(num_points):
                # Random point on sphere surface
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                
                # Color based on position
                color = [(x/r + 1)/2, (y/r + 1)/2, (z/r + 1)/2]
                
                points.append({
                    'pos': [x / 50, z / 50, y / 50],
                    'color': color
                })
        
        else:  # default - generate a generic object
            for _ in range(num_points):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(0.3, 1.0)
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi) * 0.6  # Flatten slightly
                
                points.append({
                    'pos': [x, z, y],
                    'color': [0.5, 0.6, 0.8]
                })
        
        return points
    
    def _generate_fallback_points(self, num_points: int) -> List[Dict]:
        """Generate a default object point cloud."""
        points = []
        for _ in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0.5, 1.0)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            points.append({
                'pos': [x, z, y],
                'color': [0.6, 0.7, 0.8]
            })
        return points
    
    def _normalize_points(self, points: List[Dict]) -> List[Dict]:
        """Normalize points to unit sphere."""
        coords = np.array([p['pos'] for p in points])
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
        
        if max_dist > 0:
            coords_normalized = coords_centered / max_dist
        else:
            coords_normalized = coords_centered
        
        for i, p in enumerate(points):
            p['pos'] = coords_normalized[i].tolist()
        
        return points


# Test
if __name__ == "__main__":
    service = OpenSCADService()
    points, code = service.generate_point_cloud("a detailed gear with 24 teeth")
    print(f"Generated {len(points)} points")
    print(f"OpenSCAD code:\n{code[:500]}...")
