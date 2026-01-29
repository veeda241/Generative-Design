from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import uvicorn
import uuid
import asyncio
import random
import math
from engine import EngineeringEngine, check_flow_compliance, estimate_power_kw, validate_topology
from exporter import DXFExporter
from bim_handler import bim_handler
from dotenv import load_dotenv
from point_e_service import PointEService
from openscad_service import OpenSCADService
from multimodel_service import get_multimodel_service, MultiModelService

load_dotenv()


def generate_smart_synthetic_points(prompt: str, num_points: int = 8192) -> List[Dict]:
    """
    Generate synthetic 3D point cloud based on prompt keywords.
    Creates recognizable shapes based on common objects mentioned in the prompt.
    """
    prompt_lower = prompt.lower()
    points = []
    
    # Detect shape type from prompt
    if any(word in prompt_lower for word in ['cube', 'box', 'crate', 'block']):
        points = generate_cube_points(num_points)
        color = (0.4, 0.6, 0.9)  # Blue
    elif any(word in prompt_lower for word in ['sphere', 'ball', 'globe', 'orb']):
        points = generate_sphere_points(num_points)
        color = (0.9, 0.4, 0.4)  # Red
    elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pipe', 'column', 'pillar']):
        points = generate_cylinder_points(num_points)
        color = (0.4, 0.9, 0.4)  # Green
    elif any(word in prompt_lower for word in ['cone', 'pyramid', 'rocket', 'arrow']):
        points = generate_cone_points(num_points)
        color = (0.9, 0.6, 0.2)  # Orange
    elif any(word in prompt_lower for word in ['torus', 'donut', 'ring', 'wheel', 'tire']):
        points = generate_torus_points(num_points)
        color = (0.8, 0.4, 0.8)  # Purple
    elif any(word in prompt_lower for word in ['car', 'vehicle', 'truck', 'auto']):
        points = generate_car_points(num_points)
        color = (0.2, 0.5, 0.9)  # Blue
    elif any(word in prompt_lower for word in ['house', 'building', 'home']):
        points = generate_house_points(num_points)
        color = (0.8, 0.6, 0.4)  # Brown
    elif any(word in prompt_lower for word in ['chair', 'seat', 'stool']):
        points = generate_chair_points(num_points)
        color = (0.6, 0.4, 0.2)  # Wood brown
    elif any(word in prompt_lower for word in ['table', 'desk']):
        points = generate_table_points(num_points)
        color = (0.5, 0.3, 0.1)  # Dark wood
    elif any(word in prompt_lower for word in ['gear', 'cog', 'sprocket']):
        points = generate_gear_points(num_points)
        color = (0.7, 0.7, 0.7)  # Metal gray
    elif any(word in prompt_lower for word in ['mug', 'cup', 'glass']):
        points = generate_mug_points(num_points)
        color = (0.9, 0.9, 0.9)  # White
    elif any(word in prompt_lower for word in ['star', 'starfish']):
        points = generate_star_points(num_points)
        color = (0.9, 0.8, 0.2)  # Gold
    elif any(word in prompt_lower for word in ['airplane', 'plane', 'aircraft', 'jet']):
        points = generate_airplane_points(num_points)
        color = (0.8, 0.8, 0.8)  # Silver
    else:
        # Default: composite shape
        points = generate_composite_shape(prompt_lower, num_points)
        color = (0.5, 0.7, 0.9)  # Light blue
    
    # Apply color with slight variations
    colored_points = []
    for p in points:
        variation = random.uniform(0.9, 1.1)
        colored_points.append({
            "x": p[0],
            "y": p[1],
            "z": p[2],
            "r": min(1.0, color[0] * variation),
            "g": min(1.0, color[1] * variation),
            "b": min(1.0, color[2] * variation)
        })
    
    return colored_points


def generate_cube_points(n: int) -> List[tuple]:
    """Generate points on the surface of a cube."""
    points = []
    side = 2.0
    per_face = n // 6
    
    for face in range(6):
        for _ in range(per_face):
            u = random.uniform(-side/2, side/2)
            v = random.uniform(-side/2, side/2)
            if face == 0: points.append((u, v, side/2))     # Front
            elif face == 1: points.append((u, v, -side/2))  # Back
            elif face == 2: points.append((u, side/2, v))   # Top
            elif face == 3: points.append((u, -side/2, v))  # Bottom
            elif face == 4: points.append((side/2, u, v))   # Right
            elif face == 5: points.append((-side/2, u, v))  # Left
    return points


def generate_sphere_points(n: int) -> List[tuple]:
    """Generate points on the surface of a sphere."""
    points = []
    radius = 1.0
    for _ in range(n):
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(random.uniform(-1, 1))
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        points.append((x, y, z))
    return points


def generate_cylinder_points(n: int) -> List[tuple]:
    """Generate points on the surface of a cylinder."""
    points = []
    radius = 1.0
    height = 2.0
    n_body = int(n * 0.7)
    n_caps = (n - n_body) // 2
    
    # Body
    for _ in range(n_body):
        theta = random.uniform(0, 2 * math.pi)
        h = random.uniform(-height/2, height/2)
        points.append((radius * math.cos(theta), h, radius * math.sin(theta)))
    
    # Top and bottom caps
    for _ in range(n_caps):
        r = radius * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * math.pi)
        points.append((r * math.cos(theta), height/2, r * math.sin(theta)))
        points.append((r * math.cos(theta), -height/2, r * math.sin(theta)))
    
    return points


def generate_cone_points(n: int) -> List[tuple]:
    """Generate points on the surface of a cone."""
    points = []
    radius = 1.0
    height = 2.0
    
    n_surface = int(n * 0.8)
    n_base = n - n_surface
    
    # Cone surface
    for _ in range(n_surface):
        h = random.uniform(0, height)
        r = radius * (1 - h/height)
        theta = random.uniform(0, 2 * math.pi)
        points.append((r * math.cos(theta), h - height/2, r * math.sin(theta)))
    
    # Base
    for _ in range(n_base):
        r = radius * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * math.pi)
        points.append((r * math.cos(theta), -height/2, r * math.sin(theta)))
    
    return points


def generate_torus_points(n: int) -> List[tuple]:
    """Generate points on the surface of a torus."""
    points = []
    R = 1.0  # Major radius
    r = 0.3  # Minor radius
    
    for _ in range(n):
        u = random.uniform(0, 2 * math.pi)
        v = random.uniform(0, 2 * math.pi)
        x = (R + r * math.cos(v)) * math.cos(u)
        y = r * math.sin(v)
        z = (R + r * math.cos(v)) * math.sin(u)
        points.append((x, y, z))
    
    return points


def generate_car_points(n: int) -> List[tuple]:
    """Generate points for a simplified car shape."""
    points = []
    
    # Body (box)
    n_body = int(n * 0.5)
    for _ in range(n_body):
        x = random.uniform(-1.5, 1.5)
        y = random.uniform(0.2, 0.7)
        z = random.uniform(-0.6, 0.6)
        # Surface only
        if random.random() < 0.3:
            y = 0.2 if random.random() < 0.5 else 0.7
        elif random.random() < 0.5:
            x = -1.5 if random.random() < 0.5 else 1.5
        else:
            z = -0.6 if random.random() < 0.5 else 0.6
        points.append((x, y, z))
    
    # Cabin (smaller box on top)
    n_cabin = int(n * 0.25)
    for _ in range(n_cabin):
        x = random.uniform(-0.5, 0.8)
        y = random.uniform(0.7, 1.1)
        z = random.uniform(-0.5, 0.5)
        points.append((x, y, z))
    
    # Wheels (4 cylinders)
    n_wheels = n - n_body - n_cabin
    wheel_positions = [(-1.0, 0.2, 0.7), (1.0, 0.2, 0.7), (-1.0, 0.2, -0.7), (1.0, 0.2, -0.7)]
    for _ in range(n_wheels):
        wp = random.choice(wheel_positions)
        theta = random.uniform(0, 2 * math.pi)
        r = 0.2
        offset = random.uniform(-0.05, 0.05)
        points.append((wp[0] + offset, wp[1] + r * math.sin(theta), wp[2] + r * math.cos(theta)))
    
    return points


def generate_house_points(n: int) -> List[tuple]:
    """Generate points for a house shape."""
    points = []
    
    # Main body (cube)
    n_body = int(n * 0.6)
    for _ in range(n_body):
        x = random.uniform(-1, 1)
        y = random.uniform(0, 1)
        z = random.uniform(-1, 1)
        # Surface
        face = random.randint(0, 5)
        if face == 0: x = -1
        elif face == 1: x = 1
        elif face == 2: y = 0
        elif face == 3: y = 1
        elif face == 4: z = -1
        else: z = 1
        points.append((x, y, z))
    
    # Roof (pyramid)
    n_roof = n - n_body
    for _ in range(n_roof):
        t = random.uniform(0, 1)
        x = random.uniform(-1, 1) * (1 - t)
        z = random.uniform(-1, 1) * (1 - t)
        y = 1 + t * 0.7
        points.append((x, y, z))
    
    return points


def generate_chair_points(n: int) -> List[tuple]:
    """Generate points for a chair shape."""
    points = []
    
    # Seat
    n_seat = int(n * 0.3)
    for _ in range(n_seat):
        x = random.uniform(-0.5, 0.5)
        y = 0.5 + random.uniform(-0.02, 0.02)
        z = random.uniform(-0.5, 0.5)
        points.append((x, y, z))
    
    # Back
    n_back = int(n * 0.25)
    for _ in range(n_back):
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(0.5, 1.2)
        z = -0.5 + random.uniform(-0.02, 0.02)
        points.append((x, y, z))
    
    # Legs (4 cylinders)
    n_legs = n - n_seat - n_back
    leg_positions = [(-0.4, 0, -0.4), (0.4, 0, -0.4), (-0.4, 0, 0.4), (0.4, 0, 0.4)]
    for _ in range(n_legs):
        lp = random.choice(leg_positions)
        theta = random.uniform(0, 2 * math.pi)
        r = 0.03
        y = random.uniform(0, 0.5)
        points.append((lp[0] + r * math.cos(theta), y, lp[2] + r * math.sin(theta)))
    
    return points


def generate_table_points(n: int) -> List[tuple]:
    """Generate points for a table shape."""
    points = []
    
    # Top
    n_top = int(n * 0.4)
    for _ in range(n_top):
        x = random.uniform(-1.2, 1.2)
        y = 0.8 + random.uniform(-0.02, 0.02)
        z = random.uniform(-0.7, 0.7)
        points.append((x, y, z))
    
    # Legs (4 cylinders)
    n_legs = n - n_top
    leg_positions = [(-1.0, 0, -0.5), (1.0, 0, -0.5), (-1.0, 0, 0.5), (1.0, 0, 0.5)]
    for _ in range(n_legs):
        lp = random.choice(leg_positions)
        theta = random.uniform(0, 2 * math.pi)
        r = 0.05
        y = random.uniform(0, 0.8)
        points.append((lp[0] + r * math.cos(theta), y, lp[2] + r * math.sin(theta)))
    
    return points


def generate_gear_points(n: int) -> List[tuple]:
    """Generate points for a gear shape."""
    points = []
    teeth = 12
    outer_r = 1.0
    inner_r = 0.7
    thickness = 0.2
    
    for _ in range(n):
        theta = random.uniform(0, 2 * math.pi)
        # Alternate between teeth and valleys
        tooth_angle = (2 * math.pi) / teeth
        pos_in_tooth = (theta % tooth_angle) / tooth_angle
        
        if pos_in_tooth < 0.5:
            r = outer_r
        else:
            r = inner_r
        
        r += random.uniform(-0.02, 0.02)
        z = random.uniform(-thickness/2, thickness/2)
        
        points.append((r * math.cos(theta), z, r * math.sin(theta)))
    
    return points


def generate_mug_points(n: int) -> List[tuple]:
    """Generate points for a mug shape."""
    points = []
    
    # Body (cylinder)
    n_body = int(n * 0.7)
    for _ in range(n_body):
        theta = random.uniform(0, 2 * math.pi)
        y = random.uniform(0, 1)
        r = 0.4
        points.append((r * math.cos(theta), y, r * math.sin(theta)))
    
    # Handle (partial torus)
    n_handle = int(n * 0.2)
    for _ in range(n_handle):
        u = random.uniform(-math.pi/2, math.pi/2)
        v = random.uniform(0, 2 * math.pi)
        R = 0.25
        r = 0.05
        x = 0.4 + (R + r * math.cos(v)) * math.cos(u)
        y = 0.5 + R * math.sin(u)
        z = r * math.sin(v)
        points.append((x, y, z))
    
    # Bottom
    n_bottom = n - n_body - n_handle
    for _ in range(n_bottom):
        r = 0.4 * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 2 * math.pi)
        points.append((r * math.cos(theta), 0, r * math.sin(theta)))
    
    return points


def generate_star_points(n: int) -> List[tuple]:
    """Generate points for a 5-pointed star shape."""
    points = []
    outer_r = 1.0
    inner_r = 0.4
    thickness = 0.1
    
    for _ in range(n):
        theta = random.uniform(0, 2 * math.pi)
        # 5 points
        star_angle = (2 * math.pi) / 5
        pos_in_point = (theta % star_angle) / star_angle
        
        if pos_in_point < 0.5:
            r = inner_r + (outer_r - inner_r) * (2 * pos_in_point)
        else:
            r = outer_r - (outer_r - inner_r) * (2 * (pos_in_point - 0.5))
        
        z = random.uniform(-thickness/2, thickness/2)
        points.append((r * math.cos(theta), z, r * math.sin(theta)))
    
    return points


def generate_airplane_points(n: int) -> List[tuple]:
    """Generate points for an airplane shape."""
    points = []
    
    # Fuselage (elongated cylinder)
    n_body = int(n * 0.4)
    for _ in range(n_body):
        x = random.uniform(-1.5, 1.5)
        theta = random.uniform(0, 2 * math.pi)
        r = 0.2 * (1 - 0.3 * abs(x) / 1.5)  # Taper at ends
        points.append((x, r * math.sin(theta), r * math.cos(theta)))
    
    # Wings
    n_wings = int(n * 0.35)
    for _ in range(n_wings):
        x = random.uniform(-0.3, 0.3)
        z = random.uniform(-1.5, 1.5)
        y = random.uniform(-0.02, 0.02)
        points.append((x, y, z))
    
    # Tail (vertical)
    n_tail = int(n * 0.15)
    for _ in range(n_tail):
        x = random.uniform(-1.5, -1.2)
        y = random.uniform(0, 0.5)
        z = random.uniform(-0.02, 0.02)
        points.append((x, y, z))
    
    # Tail (horizontal)
    n_htail = n - n_body - n_wings - n_tail
    for _ in range(n_htail):
        x = random.uniform(-1.5, -1.2)
        y = random.uniform(-0.02, 0.02)
        z = random.uniform(-0.5, 0.5)
        points.append((x, y, z))
    
    return points


def generate_composite_shape(prompt: str, n: int) -> List[tuple]:
    """Generate a composite 3D shape for unrecognized prompts."""
    points = []
    
    # Create a meaningful composite shape
    # Base: cylinder
    n_base = int(n * 0.4)
    for _ in range(n_base):
        theta = random.uniform(0, 2 * math.pi)
        h = random.uniform(-0.5, 0.5)
        r = 0.8
        points.append((r * math.cos(theta), h, r * math.sin(theta)))
    
    # Top: hemisphere
    n_top = int(n * 0.3)
    for _ in range(n_top):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi/2)
        r = 0.8
        x = r * math.sin(phi) * math.cos(theta)
        y = 0.5 + r * math.cos(phi)
        z = r * math.sin(phi) * math.sin(theta)
        points.append((x, y, z))
    
    # Details: small spheres around
    n_detail = n - n_base - n_top
    for _ in range(n_detail):
        angle = random.uniform(0, 2 * math.pi)
        pos_r = 1.2
        center = (pos_r * math.cos(angle), 0, pos_r * math.sin(angle))
        
        # Small sphere at this position
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(random.uniform(-1, 1))
        r = 0.15
        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)
        points.append((x, y, z))
    
    return points

app = FastAPI(title="Aether-Gen API", version="2.0.0 - Multi-Model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = EngineeringEngine()
point_e_service = None # Lazy load only if requested to save VRAM initially
openscad_service = None # Lazy load OpenSCAD/CADAM service
multimodel_service = None # Lazy load multi-model service

class DesignRequest(BaseModel):
    prompt: str

class PointCloudRequest(BaseModel):
    prompt: str
    quality: str = 'high'  # 'fast', 'normal', 'high', or 'ultra'
    method: str = 'multimodel'  # 'multimodel', 'auto', 'hybrid', 'ensemble', 'point_e', 'openscad', 'synthetic'

class ExportPointCloudRequest(BaseModel):
    points: List[Dict]
    format: str = 'ply'  # 'ply', 'obj', or 'json'
    filename: str = 'point_cloud'

class ConsultRequest(BaseModel):
    design: Dict
    message: str

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AETHER-GEN Multi-Model Backend is running", "version": "2.0.0"}

@app.post("/generate")
async def generate_design(request: DesignRequest):
    try:
        # Generate raw layout
        layout = await engine.generate_layout(request.prompt)
        
        # Enrich with Engineering AECCI (Automated Engineering Compliance & Cost Intelligence)
        components = layout.get('components', [])
        cost_sum = 0
        
        # 1. Topology Validation
        layout['compliance_logs'] = validate_topology(components)
        
        # 2. Component Enrichment
        for comp in components:
            ctype = comp.get('type')
            props = comp.setdefault('properties', {})
            
            # Costing Heuristics
            if ctype == 'pump':
                cost_sum += 12000
                flow_val = props.get('flow_rate', '30 MLD')
                val = float(str(flow_val).split()[0])
                props['power_estimate_kw'] = estimate_power_kw(val * 1000 / 24)
                props['is_safe'] = True
            elif ctype == 'tank':
                cost_sum += 45000
            elif ctype == 'pipe':
                cost_sum += 500 * (props.get('diameter', 4) / 2)
                dia = props.get('diameter', 12)
                is_safe, msg = check_flow_compliance(dia, 50.0)
                props['is_safe'] = is_safe
                props['compliance_msg'] = msg
            elif ctype == 'valve':
                cost_sum += 2500
            
        layout['cost_estimate'] = int(cost_sum * 1.1) # 10% contingency
        layout['safety_compliance'] = all(c.get('properties', {}).get('is_safe', True) for c in components)
        
        # 3. Generate CAD Assets
        dxf_filename = f"design_{uuid.uuid4().hex[:8]}.dxf"
        dxf_path = DXFExporter.generate_dxf(layout, dxf_filename)
        layout['dxf_url'] = f"/exports/{dxf_filename}"
        
        return layout
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-points")
async def generate_points(request: PointCloudRequest):
    """
    Generate 3D point cloud from text prompt using Multi-Model system.
    
    Methods:
    - 'multimodel' (default): Smart combination of Point-E + OpenSCAD
    - 'hybrid': Generate different parts with best-suited method
    - 'ensemble': Generate with both and blend results
    - 'point_e': AI-based generation (good for organic shapes)
    - 'openscad': Parametric CAD (good for mechanical parts)
    - 'synthetic': Fast procedural generation
    """
    import random
    import math
    
    try:
        method = request.method
        
        # Determine number of points based on quality
        num_points = {
            'fast': 2048,
            'normal': 4096,
            'high': 8192,
            'ultra': 16384
        }.get(request.quality, 8192)
        
        print(f"🎨 Multi-Model Generation: prompt='{request.prompt}', method={method}, quality={request.quality}")
        
        # Use Multi-Model system for best quality
        if method in ['multimodel', 'auto', 'hybrid', 'ensemble']:
            global multimodel_service
            if multimodel_service is None:
                print("🔧 Initializing Multi-Model Service...")
                multimodel_service = get_multimodel_service()
            
            # Map method names
            strategy_map = {
                'multimodel': 'auto',
                'auto': 'auto',
                'hybrid': 'hybrid',
                'ensemble': 'ensemble'
            }
            strategy = strategy_map.get(method, 'auto')
            
            try:
                points, metadata = await multimodel_service.generate_point_cloud(
                    request.prompt, 
                    num_points=num_points,
                    strategy=strategy
                )
                
                if points and len(points) > 100:
                    print(f"✅ Multi-Model generated {len(points)} points: {metadata.get('methods_used', [])}")
                    return {
                        "points": points,
                        "count": len(points),
                        "quality": request.quality,
                        "source": "multimodel",
                        "methods_used": metadata.get('methods_used', []),
                        "strategy": metadata.get('strategy', strategy),
                        "analysis": metadata.get('analysis', {}),
                        "message": metadata.get('message', f"Generated {len(points)} points using Multi-Model system")
                    }
            except Exception as e:
                print(f"⚠️ Multi-Model generation failed: {e}, falling back to Point-E...")
        
        # Fallback: Try Point-E directly
        if method in ['point_e', 'multimodel', 'auto']:
            use_point_e = os.getenv('USE_POINT_E', 'true').lower() == 'true'
            if use_point_e:
                global point_e_service
                if point_e_service is None:
                    print(f"🔧 Initializing Point-E Service (quality={request.quality})...")
                    point_e_service = PointEService(quality=request.quality)
                
                try:
                    print(f"🎯 Calling Point-E generate_point_cloud...")
                    points = point_e_service.generate_point_cloud(request.prompt)
                    
                    if points and len(points) > 100:
                        # Convert Point-E format to match frontend format
                        formatted_points = []
                        for p in points:
                            formatted_points.append({
                                "x": p["pos"][0],
                                "y": p["pos"][1],
                                "z": p["pos"][2],
                                "r": p["color"][0],
                                "g": p["color"][1],
                                "b": p["color"][2]
                            })
                        
                        print(f"✅ Point-E generated {len(formatted_points)} points")
                        return {
                            "points": formatted_points, 
                            "count": len(formatted_points), 
                            "quality": request.quality, 
                            "source": "point-e",
                            "device": str(point_e_service.device),
                            "message": f"Generated using Point-E on {point_e_service.device}"
                        }
                except Exception as e:
                    import traceback
                    print(f"⚠️ Point-E generation failed: {e}")
                    traceback.print_exc()
                    if method == 'point_e':
                        raise HTTPException(status_code=500, detail=str(e))
        
        # Try OpenSCAD/CADAM if specifically requested
        if method == 'openscad':
            global openscad_service
            if openscad_service is None:
                print(f"🔧 Initializing OpenSCAD/CADAM Service...")
                openscad_service = OpenSCADService()
            
            try:
                points, scad_code = openscad_service.generate_point_cloud(request.prompt, num_points)
                
                if points and len(points) > 100:
                    # Convert to frontend format
                    formatted_points = []
                    for p in points:
                        formatted_points.append({
                            "x": p["pos"][0],
                            "y": p["pos"][1],
                            "z": p["pos"][2],
                            "r": p["color"][0],
                            "g": p["color"][1],
                            "b": p["color"][2]
                        })
                    
                    return {
                        "points": formatted_points,
                        "count": len(formatted_points),
                        "quality": request.quality,
                        "source": "openscad-cadam",
                        "openscad_code": scad_code[:2000] if scad_code else None,
                        "message": f"Generated using OpenSCAD/CADAM AI ({len(formatted_points)} points)"
                    }
            except Exception as e:
                print(f"⚠️ OpenSCAD generation failed: {e}")
                if method == 'openscad':
                    raise
        
        # Final Fallback: Enhanced Synthetic Generation
        print(f"🔄 Using enhanced synthetic generation for '{request.prompt}'")
        points = generate_smart_synthetic_points(request.prompt, num_points)
        
        return {
            "points": points, 
            "count": len(points), 
            "quality": request.quality, 
            "source": "synthetic-enhanced", 
            "message": f"Generated synthetic 3D shape ({len(points)} points)"
        }
            
    except Exception as e:
        print(f"❌ ERROR in point cloud generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-point-cloud")
async def export_point_cloud(request: ExportPointCloudRequest):
    """Export point cloud to various formats (PLY, OBJ, JSON)."""
    try:
        if not request.points:
            raise ValueError("No points provided")
        
        # Sanitize filename
        safe_filename = "".join(c if c.isalnum() or c in '-_' else '_' for c in request.filename)
        
        if request.format == 'ply':
            filepath = DXFExporter.export_point_cloud_to_ply(request.points, f"{safe_filename}.ply")
            download_name = f"{safe_filename}.ply"
        elif request.format == 'obj':
            filepath = DXFExporter.export_point_cloud_to_obj(request.points, f"{safe_filename}.obj")
            download_name = f"{safe_filename}.obj"
        elif request.format == 'json':
            import json
            os.makedirs('exports', exist_ok=True)
            filepath = os.path.join('exports', f"{safe_filename}.json")
            with open(filepath, 'w') as f:
                json.dump({"points": request.points, "count": len(request.points)}, f)
            download_name = f"{safe_filename}.json"
        else:
            raise ValueError(f"Unsupported format: {request.format}")
        
        return {
            "success": True,
            "filename": download_name,
            "url": f"/exports/{os.path.basename(filepath)}",
            "format": request.format
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consult")
async def consult_design(request: ConsultRequest):
    try:
        response = await engine.consult_design(request.design, request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# IFC/BIM Endpoints
@app.post("/upload-ifc")
async def upload_ifc_file(file: UploadFile = File(...)):
    """
    Upload and parse IFC file for BIM visualization
    Supports: IFC 2x3, IFC 4.0, IFC 4.1
    """
    try:
        # Save uploaded file
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        file_path = os.path.join(uploads_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Load and parse IFC
        ifc_data = bim_handler.load_ifc(file_path)
        
        if "error" in ifc_data:
            raise HTTPException(status_code=400, detail=ifc_data["error"])
        
        # Convert to Three.js format if available
        threejs_path = bim_handler.convert_to_threejs(file_path)
        
        return {
            "success": True,
            "file_path": file_path,
            "threejs_model": threejs_path,
            "data": ifc_data,
            "message": f"Successfully loaded {file.filename}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IFC upload error: {str(e)}")


@app.get("/bim/element-properties/{element_id}")
async def get_bim_element_properties(element_id: int, file_path: str = ""):
    """
    Get detailed properties of a specific BIM element
    """
    try:
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path required")
        
        properties = bim_handler.get_element_properties(file_path, element_id)
        return {"element_id": element_id, "properties": properties}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bim/convert")
async def convert_ifc_to_format(file_path: str, target_format: str = "threejs"):
    """
    Convert IFC to different formats
    Supported: threejs, gltf, obj
    """
    try:
        if target_format == "threejs":
            result = bim_handler.convert_to_threejs(file_path)
            return {"success": True, "output_path": result, "format": "threejs"}
        else:
            return {"success": False, "error": f"Format {target_format} not yet supported"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Server configuration
if __name__ == "__main__":
    import os
    from fastapi.staticfiles import StaticFiles
    
    # Ensure exports directory exists
    os.makedirs('exports', exist_ok=True)
    app.mount("/exports", StaticFiles(directory="exports"), name="exports")
    
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
