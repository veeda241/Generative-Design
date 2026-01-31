from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
# Import Tripo AI service (replaces Point-E)
from tripo_service import TripoService, get_tripo_service
from openscad_service import OpenSCADService
from multimodel_service import get_multimodel_service, MultiModelService
from print_service import get_print_service, PrintService, PrintOptimizer, PRINTER_PROFILES, PRINT_MATERIALS

load_dotenv()

# Global Tripo service
tripo_service: Optional[TripoService] = None


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
    elif any(word in prompt_lower for word in ['train', 'locomotive', 'railway', 'railroad']):
        points = generate_train_points(num_points)
        color = (0.3, 0.3, 0.35)  # Dark metal
    elif any(word in prompt_lower for word in ['car', 'vehicle', 'truck', 'auto']):
        points = generate_car_points(num_points)
        color = (0.2, 0.5, 0.9)  # Blue
    elif any(word in prompt_lower for word in ['boat', 'ship', 'yacht', 'vessel']):
        points = generate_boat_points(num_points)
        color = (0.9, 0.9, 0.95)  # White
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


def generate_train_points(n: int) -> List[tuple]:
    """Generate points for a detailed train/locomotive shape - surface only for clear mesh."""
    points = []
    
    # Helper function to generate box surface points
    def box_surface(cx, cy, cz, sx, sy, sz, count):
        pts = []
        faces = [
            ('x', sx/2), ('x', -sx/2),
            ('y', sy/2), ('y', -sy/2),
            ('z', sz/2), ('z', -sz/2)
        ]
        per_face = count // 6
        for axis, offset in faces:
            for _ in range(per_face):
                if axis == 'x':
                    pts.append((cx + offset, cy + random.uniform(-sy/2, sy/2), cz + random.uniform(-sz/2, sz/2)))
                elif axis == 'y':
                    pts.append((cx + random.uniform(-sx/2, sx/2), cy + offset, cz + random.uniform(-sz/2, sz/2)))
                else:
                    pts.append((cx + random.uniform(-sx/2, sx/2), cy + random.uniform(-sy/2, sy/2), cz + offset))
        return pts
    
    # Helper function to generate cylinder surface points
    def cylinder_surface(cx, cy, cz, radius, height, count, axis='y'):
        pts = []
        n_side = int(count * 0.8)
        n_cap = (count - n_side) // 2
        
        for _ in range(n_side):
            theta = random.uniform(0, 2 * math.pi)
            h = random.uniform(-height/2, height/2)
            if axis == 'y':
                pts.append((cx + radius * math.cos(theta), cy + h, cz + radius * math.sin(theta)))
            elif axis == 'x':
                pts.append((cx + h, cy + radius * math.cos(theta), cz + radius * math.sin(theta)))
            else:
                pts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta), cz + h))
        
        for _ in range(n_cap):
            theta = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, radius)
            if axis == 'y':
                pts.append((cx + r * math.cos(theta), cy + height/2, cz + r * math.sin(theta)))
                pts.append((cx + r * math.cos(theta), cy - height/2, cz + r * math.sin(theta)))
        return pts
    
    # Main locomotive body
    points.extend(box_surface(0, 0.6, 0, 2.0, 0.6, 0.8, int(n * 0.3)))
    
    # Locomotive nose (front wedge using triangular surface)
    n_nose = int(n * 0.1)
    for _ in range(n_nose):
        x = random.uniform(1.0, 1.4)
        progress = (x - 1.0) / 0.4
        max_y = 0.9 - 0.4 * progress
        max_z = 0.4 - 0.25 * progress
        # Surface only - on the sloped face
        y = 0.3 + random.uniform(0, max_y - 0.3)
        z = random.uniform(-max_z, max_z)
        points.append((x, y, z))
    
    # Cabin (control room) - box on top
    points.extend(box_surface(-0.4, 1.1, 0, 0.8, 0.4, 0.7, int(n * 0.12)))
    
    # Windows on cabin (indentations represented as darker points)
    n_windows = int(n * 0.03)
    for _ in range(n_windows):
        x = random.uniform(-0.7, -0.1)
        y = random.uniform(1.0, 1.2)
        z = random.choice([-0.36, 0.36])
        points.append((x, y, z))
    
    # 6 Wheels (detailed cylinders)
    wheel_positions = [(-0.7, -0.45), (-0.7, 0.45), (0.0, -0.45), (0.0, 0.45), (0.7, -0.45), (0.7, 0.45)]
    n_per_wheel = int(n * 0.06)
    for wx, wz in wheel_positions:
        points.extend(cylinder_surface(wx, 0.2, wz, 0.2, 0.08, n_per_wheel, axis='z'))
    
    # Smokestack (cylinder)
    points.extend(cylinder_surface(0.6, 1.1, 0, 0.1, 0.4, int(n * 0.05), axis='y'))
    
    # Cowcatcher (front plow)
    n_cow = int(n * 0.05)
    for _ in range(n_cow):
        x = random.uniform(1.3, 1.5)
        y = random.uniform(0.1, 0.35)
        z = random.uniform(-0.3 * (1.5 - x) / 0.2, 0.3 * (1.5 - x) / 0.2)
        points.append((x, y, z))
    
    # Undercarriage/chassis
    points.extend(box_surface(0, 0.25, 0, 2.0, 0.1, 0.9, int(n * 0.08)))
    
    # Headlight
    n_light = int(n * 0.02)
    for _ in range(n_light):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi/2)
        r = 0.08
        x = 1.4 + r * math.cos(phi)
        y = 0.75 + r * math.sin(phi) * math.sin(theta)
        z = r * math.sin(phi) * math.cos(theta)
        points.append((x, y, z))
    
    # Fill remaining points with details
    remaining = n - len(points)
    for _ in range(max(0, remaining)):
        # Add rivets/details on main body surface
        x = random.uniform(-0.95, 0.95)
        y = random.choice([0.3, 0.9])  # Top or bottom edge
        z = random.uniform(-0.38, 0.38)
        points.append((x, y, z))
    
    return points


def generate_boat_points(n: int) -> List[tuple]:
    """Generate points for a boat/ship shape."""
    points = []
    
    # Hull (elongated with curved bottom)
    n_hull = int(n * 0.5)
    for _ in range(n_hull):
        x = random.uniform(-1.5, 1.5)
        progress = abs(x) / 1.5
        width = 0.6 * (1 - progress * 0.4)  # Narrower at ends
        z = random.uniform(-width, width)
        
        # Curved hull bottom
        depth = 0.4 * (1 - (z / width) ** 2) if width > 0 else 0
        y = random.uniform(-depth, 0.1)
        points.append((x, y, z))
    
    # Deck
    n_deck = int(n * 0.2)
    for _ in range(n_deck):
        x = random.uniform(-1.3, 1.3)
        progress = abs(x) / 1.3
        width = 0.55 * (1 - progress * 0.3)
        z = random.uniform(-width, width)
        y = random.uniform(0.1, 0.15)
        points.append((x, y, z))
    
    # Cabin/superstructure
    n_cabin = int(n * 0.15)
    for _ in range(n_cabin):
        x = random.uniform(-0.5, 0.3)
        y = random.uniform(0.15, 0.6)
        z = random.uniform(-0.3, 0.3)
        if random.random() < 0.3:
            y = random.choice([0.15, 0.6])
        points.append((x, y, z))
    
    # Mast
    n_mast = int(n * 0.08)
    for _ in range(n_mast):
        theta = random.uniform(0, 2 * math.pi)
        r = 0.03
        y = random.uniform(0.3, 1.2)
        x = 0.0 + r * math.cos(theta)
        z = r * math.sin(theta)
        points.append((x, y, z))
    
    # Bow details
    n_bow = n - n_hull - n_deck - n_cabin - n_mast
    for _ in range(n_bow):
        x = random.uniform(1.3, 1.6)
        progress = (x - 1.3) / 0.3
        y = random.uniform(-0.1 * (1 - progress), 0.2 * (1 - progress))
        z = random.uniform(-0.1 * (1 - progress), 0.1 * (1 - progress))
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

app = FastAPI(title="Aether-Gen API", version="2.5.0 - Multi-Model + 3D Printing")

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
print_service = None  # Lazy load 3D printing service

class DesignRequest(BaseModel):
    prompt: str

class PointCloudRequest(BaseModel):
    prompt: str
    quality: str = 'high'  # 'fast', 'normal', 'high', or 'ultra'
    method: str = 'multimodel'  # 'multimodel', 'auto', 'hybrid', 'ensemble', 'point_e', 'openscad', 'synthetic'

class ExportPointCloudRequest(BaseModel):
    points: List[Dict]
    format: str = 'ply'  # 'ply', 'obj', 'json', 'stl', or '3mf'
    filename: str = 'point_cloud'

class ConsultRequest(BaseModel):
    design: Dict
    message: str

# 3D Printing Request Models
class PrintableModelRequest(BaseModel):
    points: List[Dict]
    target_size_mm: float = 100.0
    mesh_method: str = 'voxel'  # 'voxel', 'alpha_shape', 'convex', 'ball_pivot'
    mesh_resolution: int = 64
    printer: str = 'generic'  # Printer profile key
    material: str = 'pla'  # Material key
    optimize_orientation: bool = True
    densify: bool = True
    smooth: bool = True

class ExportPrintableRequest(BaseModel):
    points: List[Dict]
    format: str = 'stl'  # 'stl' or '3mf'
    filename: str = 'print_model'
    target_size_mm: float = 100.0
    mesh_method: str = 'voxel'
    mesh_resolution: int = 64
    binary_stl: bool = True

class PrintValidationRequest(BaseModel):
    points: List[Dict]
    target_size_mm: float = 100.0
    printer: str = 'generic'
    material: str = 'pla'

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AETHER-GEN Tripo AI + 3D Generation Backend is running", "version": "3.0.0"}


# ============== TRIPO AI ENDPOINTS ==============

class TripoTextRequest(BaseModel):
    prompt: str
    model_version: str = "v2.0-20240919"
    negative_prompt: Optional[str] = None
    face_limit: int = 50000
    texture: bool = True
    pbr: bool = True
    output_format: str = "glb"


class TripoImageRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    model_version: str = "v2.0-20240919"
    face_limit: int = 50000
    texture: bool = True
    pbr: bool = True
    output_format: str = "glb"


@app.post("/tripo/text-to-3d")
async def tripo_text_to_3d(request: TripoTextRequest):
    """
    Generate a high-quality 3D model from text using Tripo AI.
    Requires TRIPO_API_KEY environment variable.
    """
    global tripo_service
    
    if tripo_service is None:
        tripo_service = get_tripo_service()
    
    if not tripo_service.api_key:
        raise HTTPException(
            status_code=400, 
            detail="TRIPO_API_KEY not configured. Please set your API key."
        )
    
    try:
        print(f"🚀 Tripo AI: Generating 3D from text: '{request.prompt}'")
        
        result = await tripo_service.text_to_3d(
            prompt=request.prompt,
            model_version=request.model_version,
            negative_prompt=request.negative_prompt,
            face_limit=request.face_limit,
            texture=request.texture,
            pbr=request.pbr,
            output_format=request.output_format
        )
        
        if result.get("status") == "success":
            local_path = result.get("local_path")
            if local_path:
                result["download_url"] = f"/exports/{os.path.basename(local_path)}"
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Tripo text-to-3d error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tripo/image-to-3d")
async def tripo_image_to_3d(request: TripoImageRequest):
    """
    Generate a high-quality 3D model from an image using Tripo AI.
    Requires TRIPO_API_KEY environment variable.
    """
    global tripo_service
    
    if tripo_service is None:
        tripo_service = get_tripo_service()
    
    if not tripo_service.api_key:
        raise HTTPException(
            status_code=400, 
            detail="TRIPO_API_KEY not configured. Please set your API key."
        )
    
    if not request.image_url and not request.image_base64:
        raise HTTPException(status_code=400, detail="Either image_url or image_base64 is required")
    
    try:
        print(f"🚀 Tripo AI: Generating 3D from image...")
        
        result = await tripo_service.image_to_3d(
            image_url=request.image_url,
            image_base64=request.image_base64,
            model_version=request.model_version,
            face_limit=request.face_limit,
            texture=request.texture,
            pbr=request.pbr,
            output_format=request.output_format
        )
        
        if result.get("status") == "success":
            local_path = result.get("local_path")
            if local_path:
                result["download_url"] = f"/exports/{os.path.basename(local_path)}"
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Tripo image-to-3d error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tripo/balance")
async def tripo_balance():
    """Get Tripo AI API balance/credits."""
    global tripo_service
    
    if tripo_service is None:
        tripo_service = get_tripo_service()
    
    if not tripo_service.api_key:
        raise HTTPException(status_code=400, detail="TRIPO_API_KEY not configured")
    
    balance = await tripo_service.get_balance()
    return balance


# ============== END TRIPO AI ENDPOINTS ==============

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


class MeshRequest(BaseModel):
    prompt: str
    method: str = 'auto'  # 'auto', 'tripo', 'openscad', 'synthetic'
    mesh_resolution: int = 64
    use_texture: bool = True
    use_pbr: bool = True


@app.post("/generate-mesh")
async def generate_mesh(request: MeshRequest):
    """
    Generate a 3D mesh from a text prompt using Tripo AI.
    Falls back to synthetic generation if Tripo is unavailable.
    """
    global print_service, tripo_service
    
    try:
        print(f"🎨 Generating mesh for prompt: '{request.prompt}'")
        print(f"📦 Method: {request.method}")
        
        # Initialize Tripo service if needed
        if tripo_service is None:
            tripo_service = get_tripo_service()
        
        mesh_path = None
        mesh_filename = None
        method_used = request.method
        
        # Try Tripo AI first (unless synthetic is explicitly requested)
        if request.method in ['auto', 'tripo'] and tripo_service.api_key:
            print("🚀 Using Tripo AI for 3D generation...")
            
            try:
                result = await tripo_service.text_to_3d(
                    prompt=request.prompt,
                    texture=request.use_texture,
                    pbr=request.use_pbr,
                    face_limit=50000
                )
                
                if result.get("status") == "success":
                    # Get the GLB file path
                    glb_path = result.get("local_path")
                    
                    if glb_path and os.path.exists(glb_path):
                        # Convert GLB to STL for frontend compatibility
                        mesh_id = str(uuid.uuid4())[:8]
                        mesh_filename = f"tripo_{mesh_id}.stl"
                        mesh_path = os.path.join("exports", mesh_filename)
                        os.makedirs("exports", exist_ok=True)
                        
                        if tripo_service.convert_to_stl(glb_path, mesh_path):
                            method_used = "tripo"
                            print(f"✅ Tripo AI mesh generated and converted to STL")
                        else:
                            # Keep GLB if STL conversion fails
                            mesh_filename = os.path.basename(glb_path)
                            mesh_path = glb_path
                            method_used = "tripo"
                            print(f"⚠️ STL conversion failed, using GLB format")
                    else:
                        print(f"⚠️ Tripo succeeded but no local file. Falling back to synthetic.")
                        mesh_path = None
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"⚠️ Tripo AI generation failed: {error_msg}. Falling back to synthetic.")
                    mesh_path = None
                    
            except Exception as tripo_error:
                print(f"⚠️ Tripo AI error: {tripo_error}. Falling back to synthetic.")
                mesh_path = None
        
        # Fallback to synthetic generation if Tripo is not available or failed
        if mesh_path is None or not os.path.exists(mesh_path):
            print("🔧 Using synthetic point cloud generation...")
            method_used = "synthetic"
            
            # Step 1: Generate point cloud
            point_request = PointCloudRequest(
                prompt=request.prompt,
                quality='high',
                method='synthetic'
            )
            
            # Call generate_points internally
            point_result = await generate_points(point_request)
            points = point_result.get('points', [])
            
            if not points or len(points) < 100:
                raise HTTPException(status_code=500, detail="Failed to generate sufficient points")
            
            print(f"✅ Generated {len(points)} points, converting to mesh...")
            
            # Step 2: Convert point cloud to mesh using print_service
            if print_service is None:
                print("🔧 Initializing 3D Print Service...")
                print_service = get_print_service()
            
            # Convert points to the format expected by print_service (x, y, z format)
            point_array = []
            for p in points:
                if 'x' in p:
                    point_array.append({'x': p['x'], 'y': p['y'], 'z': p['z']})
                elif 'pos' in p:
                    point_array.append({'x': p['pos'][0], 'y': p['pos'][1], 'z': p['pos'][2]})
                else:
                    continue  # Skip invalid points
            
            if len(point_array) < 100:
                raise HTTPException(status_code=500, detail="Not enough valid points for mesh generation")
            
            # Generate mesh using ball_pivot for better surface preservation
            vertices, faces = print_service.point_cloud_to_mesh(
                point_array,
                method='ball_pivot',
                resolution=request.mesh_resolution
            )
            
            if len(vertices) == 0 or len(faces) == 0:
                raise HTTPException(status_code=500, detail="Failed to generate mesh from points")
            
            print(f"✅ Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
            
            # Step 3: Export mesh to binary STL file
            import struct
            import numpy as np
            
            mesh_id = str(uuid.uuid4())[:8]
            mesh_filename = f"mesh_{mesh_id}.stl"
            mesh_path = os.path.join("exports", mesh_filename)
            os.makedirs("exports", exist_ok=True)
            
            vertices_arr = np.array(vertices)
            faces_arr = np.array(faces)
            
            with open(mesh_path, 'wb') as f:
                header = f"Generated mesh for: {request.prompt[:60]}"
                f.write(header.encode('utf-8').ljust(80, b'\0'))
                
                num_triangles = len(faces_arr)
                f.write(struct.pack('<I', num_triangles))
                
                for face in faces_arr:
                    v0 = vertices_arr[face[0]]
                    v1 = vertices_arr[face[1]]
                    v2 = vertices_arr[face[2]]
                    
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 0:
                        normal = normal / norm_len
                    else:
                        normal = np.array([0, 0, 1])
                    
                    f.write(struct.pack('<fff', *normal))
                    f.write(struct.pack('<fff', *v0))
                    f.write(struct.pack('<fff', *v1))
                    f.write(struct.pack('<fff', *v2))
                    f.write(struct.pack('<H', 0))
        
        print(f"✅ Mesh saved to {mesh_path}")
        
        return {
            "success": True,
            "message": f"Generated 3D mesh from '{request.prompt}'",
            "method": method_used,
            "url": f"/exports/{mesh_filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Mesh generation failed: {e}")
        import traceback
        traceback.print_exc()
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


# ============= 3D PRINTING ENDPOINTS =============

@app.get("/printing/printers")
async def get_printer_profiles():
    """Get available 3D printer profiles."""
    return {
        "printers": {k: v for k, v in PRINTER_PROFILES.items()},
        "default": "generic"
    }

@app.get("/printing/materials")
async def get_print_materials():
    """Get available print materials."""
    return {
        "materials": {k: v for k, v in PRINT_MATERIALS.items()},
        "default": "pla"
    }

@app.post("/printing/validate")
async def validate_for_printing(request: PrintValidationRequest):
    """
    Validate a point cloud for 3D printing.
    Returns printability analysis, estimated print time, and material usage.
    """
    global print_service
    
    try:
        if print_service is None:
            print_service = get_print_service()
        
        # Set printer and material
        print_service.set_printer(request.printer)
        print_service.set_material(request.material)
        
        # Convert to mesh
        print("🔧 Converting point cloud to mesh for validation...")
        vertices, faces = print_service.point_cloud_to_mesh(
            request.points, 
            method='voxel', 
            resolution=48  # Lower resolution for quick validation
        )
        
        # Scale to target size
        vertices = print_service.scale_for_printing(vertices, request.target_size_mm)
        
        # Validate
        validation = print_service.validate_for_printing(vertices, faces, request.target_size_mm)
        
        return {
            "success": True,
            "validation": validation,
            "printer": request.printer,
            "material": request.material
        }
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/printing/convert-to-mesh")
async def convert_to_printable_mesh(request: PrintableModelRequest):
    """
    Convert point cloud to print-ready mesh.
    Applies densification, smoothing, and orientation optimization.
    """
    global print_service
    
    try:
        if print_service is None:
            print_service = get_print_service()
        
        print_service.set_printer(request.printer)
        print_service.set_material(request.material)
        
        points = request.points
        
        # Pre-processing
        if request.densify:
            print("📈 Densifying point cloud...")
            target_count = max(len(points) * 2, 16384)
            points = PrintOptimizer.densify_point_cloud(points, target_count)
        
        if request.smooth:
            print("✨ Smoothing point cloud...")
            points = PrintOptimizer.smooth_point_cloud(points, iterations=2)
        
        # Convert to mesh
        print(f"🔧 Converting to mesh (method={request.mesh_method}, resolution={request.mesh_resolution})...")
        vertices, faces = print_service.point_cloud_to_mesh(
            points,
            method=request.mesh_method,
            resolution=request.mesh_resolution
        )
        
        # Scale for printing
        vertices = print_service.scale_for_printing(vertices, request.target_size_mm)
        
        # Optimize orientation
        orientation_info = None
        if request.optimize_orientation:
            print("🔄 Optimizing print orientation...")
            vertices, orientation_info = PrintOptimizer.orient_for_printing(vertices, faces)
        
        # Validate
        validation = print_service.validate_for_printing(vertices, faces, request.target_size_mm)
        
        return {
            "success": True,
            "mesh": {
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
                "vertex_count": len(vertices),
                "face_count": len(faces)
            },
            "validation": validation,
            "orientation": orientation_info,
            "settings": {
                "target_size_mm": request.target_size_mm,
                "mesh_method": request.mesh_method,
                "mesh_resolution": request.mesh_resolution,
                "printer": request.printer,
                "material": request.material
            }
        }
        
    except Exception as e:
        print(f"❌ Mesh conversion error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/printing/export")
async def export_for_printing(request: ExportPrintableRequest):
    """
    Export point cloud as print-ready STL or 3MF file.
    """
    global print_service
    
    try:
        if print_service is None:
            print_service = get_print_service()
        
        # Densify and smooth
        points = PrintOptimizer.densify_point_cloud(request.points, 16384)
        points = PrintOptimizer.smooth_point_cloud(points, iterations=2)
        
        # Convert to mesh
        print(f"🔧 Converting to mesh for export...")
        vertices, faces = print_service.point_cloud_to_mesh(
            points,
            method=request.mesh_method,
            resolution=request.mesh_resolution
        )
        
        # Scale for printing
        vertices = print_service.scale_for_printing(vertices, request.target_size_mm)
        
        # Optimize orientation
        vertices, _ = PrintOptimizer.orient_for_printing(vertices, faces)
        
        # Sanitize filename
        safe_filename = "".join(c if c.isalnum() or c in '-_' else '_' for c in request.filename)
        
        # Export
        if request.format.lower() == 'stl':
            filepath = print_service.export_stl(
                vertices, faces, safe_filename, 
                binary=request.binary_stl
            )
        elif request.format.lower() == '3mf':
            filepath = print_service.export_3mf(
                vertices, faces, safe_filename,
                metadata={'title': request.filename}
            )
        else:
            raise ValueError(f"Unsupported format: {request.format}. Use 'stl' or '3mf'")
        
        # Validate
        validation = print_service.validate_for_printing(vertices, faces, request.target_size_mm)
        
        return {
            "success": True,
            "filename": os.path.basename(filepath),
            "url": f"/exports/{os.path.basename(filepath)}",
            "format": request.format,
            "mesh_stats": {
                "vertices": len(vertices),
                "faces": len(faces)
            },
            "validation": validation
        }
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/printing/generate-and-export")
async def generate_and_export_printable(prompt: str, quality: str = 'high', 
                                        printer: str = 'generic', 
                                        target_size_mm: float = 100.0,
                                        export_format: str = 'stl'):
    """
    One-step endpoint: Generate 3D model from text and export as print-ready file.
    
    Args:
        prompt: Text description of the object
        quality: Generation quality ('fast', 'normal', 'high', 'ultra')
        printer: Target printer profile
        target_size_mm: Size of longest dimension in mm
        export_format: 'stl' or '3mf'
    """
    global multimodel_service, print_service
    
    try:
        # Initialize services
        if multimodel_service is None:
            multimodel_service = get_multimodel_service()
        if print_service is None:
            print_service = get_print_service()
        
        print_service.set_printer(printer)
        
        # Generate point cloud
        num_points = {'fast': 4096, 'normal': 8192, 'high': 16384, 'ultra': 32768}.get(quality, 16384)
        print(f"🎨 Generating 3D model: '{prompt}' (quality={quality}, points={num_points})")
        
        points, metadata = await multimodel_service.generate_point_cloud(prompt, num_points, strategy='auto')
        
        if not points or len(points) < 100:
            raise ValueError("Failed to generate sufficient points")
        
        print(f"✅ Generated {len(points)} points")
        
        # Prepare for printing
        points = PrintOptimizer.densify_point_cloud(points, max(len(points), 16384))
        points = PrintOptimizer.smooth_point_cloud(points, iterations=2)
        
        # Convert to mesh
        vertices, faces = print_service.point_cloud_to_mesh(points, method='voxel', resolution=64)
        vertices = print_service.scale_for_printing(vertices, target_size_mm)
        vertices, orientation_info = PrintOptimizer.orient_for_printing(vertices, faces)
        
        # Export
        safe_filename = "".join(c if c.isalnum() or c in '-_' else '_' for c in prompt[:30])
        
        if export_format.lower() == 'stl':
            filepath = print_service.export_stl(vertices, faces, safe_filename, binary=True)
        else:
            filepath = print_service.export_3mf(vertices, faces, safe_filename, 
                                                metadata={'title': prompt})
        
        validation = print_service.validate_for_printing(vertices, faces, target_size_mm)
        
        return {
            "success": True,
            "prompt": prompt,
            "filename": os.path.basename(filepath),
            "url": f"/exports/{os.path.basename(filepath)}",
            "format": export_format,
            "generation": {
                "methods_used": metadata.get('methods_used', []),
                "points_generated": len(points)
            },
            "mesh_stats": {
                "vertices": len(vertices),
                "faces": len(faces)
            },
            "validation": validation,
            "orientation": orientation_info
        }
        
    except Exception as e:
        print(f"❌ Generate and export error: {e}")
        import traceback
        traceback.print_exc()
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
