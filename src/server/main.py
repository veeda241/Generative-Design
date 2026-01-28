from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import uvicorn
import uuid
import asyncio
from engine import EngineeringEngine, check_flow_compliance, estimate_power_kw, validate_topology
from exporter import DXFExporter
from bim_handler import bim_handler
from dotenv import load_dotenv
from point_e_service import PointEService

load_dotenv()

app = FastAPI(title="Aether-Gen API", version="1.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = EngineeringEngine()
point_e_service = None # Lazy load only if requested to save VRAM initially

class DesignRequest(BaseModel):
    prompt: str

class PointCloudRequest(BaseModel):
    prompt: str
    quality: str = 'normal'  # 'fast', 'normal', or 'high'

class ExportPointCloudRequest(BaseModel):
    points: List[Dict]
    format: str = 'ply'  # 'ply', 'obj', or 'json'
    filename: str = 'point_cloud'

class ConsultRequest(BaseModel):
    design: Dict
    message: str

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AETHER-GEN Backend is running", "version": "1.5.0"}

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
    """Generate 3D point cloud from text prompt. Uses Point-E if enabled, otherwise synthetic fallback."""
    try:
        if os.getenv('USE_POINT_E', 'false').lower() == 'true':
            global point_e_service
            if point_e_service is None:
                print(f"DEBUG: Initializing Point-E Service (quality={request.quality})...")
                point_e_service = PointEService(quality=request.quality)
            
            points = point_e_service.generate_point_cloud(request.prompt)
            
            # Convert Point-E format to match synthetic output format for frontend consistency
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
                "source": "point-e",
                "device": str(point_e_service.device),
                "message": f"Generated using Point-E on {point_e_service.device}"
            }
        
        # Fallback to synthetic generation
        design = await engine.generate_layout(request.prompt)
        components = design.get('components', [])
        
        # Create synthetic point cloud from components
        points = []
        import random
        import math
        
        for comp in components:
            pos = comp.get('position', [0, 0, 0])
            comp_type = comp.get('type', 'unknown')
            
            # Generate points around each component
            num_points = 200 if request.quality == 'fast' else (500 if request.quality == 'normal' else 1000)
            
            for _ in range(num_points):
                # Random offset around component
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, 5)
                height_offset = random.uniform(-2, 2)
                
                point = {
                    "x": pos[0] + radius * math.cos(angle),
                    "y": pos[1] + height_offset,
                    "z": pos[2] + radius * math.sin(angle),
                    "r": 0.6 if comp_type == 'pump' else (0.2 if comp_type == 'tank' else 0.4),
                    "g": 0.7 if comp_type == 'pump' else (0.4 if comp_type == 'tank' else 0.6),
                    "b": 1.0 if comp_type == 'pump' else (0.8 if comp_type == 'tank' else 0.7),
                }
                points.append(point)
        
        return {"points": points, "count": len(points), "quality": request.quality, "source": "synthetic", "message": "Generated using synthetic point cloud (GPU not available)"}
            
    except Exception as e:
        print(f"ERROR in point cloud generation: {str(e)}")
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

@app.post("/upload-ifc")
async def upload_ifc_file(file: UploadFile = File(...)):
    try:
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        file_path = os.path.join(uploads_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        ifc_data = bim_handler.load_ifc(file_path)
        
        if "error" in ifc_data:
            raise HTTPException(status_code=400, detail=ifc_data["error"])
        
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
    try:
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path required")
        
        properties = bim_handler.get_element_properties(file_path, element_id)
        return {"element_id": element_id, "properties": properties}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bim/convert")
async def convert_ifc_to_format(file_path: str, target_format: str = "threejs"):
    try:
        if target_format == "threejs":
            result = bim_handler.convert_to_threejs(file_path)
            return {"success": True, "output_path": result, "format": "threejs"}
        else:
            return {"success": False, "error": f"Format {target_format} not yet supported"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    from fastapi.staticfiles import StaticFiles
    
    os.makedirs('exports', exist_ok=True)
    app.mount("/exports", StaticFiles(directory="exports"), name="exports")
    
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
