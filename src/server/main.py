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
        
        # Enrich with Engineering AECCI
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
            
        layout['cost_estimate'] = int(cost_sum * 1.1)
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
    """Generate 3D point cloud using synthetic fallback (Point-E removed)."""
    try:
        # Fallback to synthetic generation
        design = await engine.generate_layout(request.prompt)
        components = design.get('components', [])
        
        points = []
        import random
        import math
        
        for comp in components:
            pos = comp.get('position', [0, 0, 0])
            comp_type = comp.get('type', 'unknown')
            num_points = 200 if request.quality == 'fast' else (500 if request.quality == 'normal' else 1000)
            
            for _ in range(num_points):
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
        
        return {"points": points, "count": len(points), "quality": request.quality, "source": "synthetic", "message": "Generated using synthetic point cloud (Point-E removed)"}
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-point-cloud")
async def export_point_cloud(request: ExportPointCloudRequest):
    """Export point cloud to various formats."""
    try:
        if not request.points:
            raise ValueError("No points provided")
        
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
            raise ValueError(f"Unsupported format: {request.format} (STL/OBJ Mesh require /generate-mesh endpoint)")
        
        return {
            "success": True,
            "filename": download_name,
            "url": f"/exports/{os.path.basename(filepath)}",
            "format": request.format
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== MULTI-MODEL MESH GENERATION ==========

class MeshGenerationRequest(BaseModel):
    prompt: str
    method: str = 'triposr'  # 'auto', 'triposr', or 'shap-e'

multimodel_service = None

@app.post("/generate-mesh")
async def generate_mesh(request: MeshGenerationRequest):
    """Generate a high-quality mesh using TripoSR or Shap-E."""
    global multimodel_service
    try:
        if multimodel_service is None:
            from multimodel_service import MultiModelService
            multimodel_service = MultiModelService()
        
        import concurrent.futures
        def generate():
            return multimodel_service.generate_best_mesh(request.prompt, request.method)
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            verts, faces, method_used = await loop.run_in_executor(pool, generate)
        
        safe_filename = "".join(c if c.isalnum() or c in '-_' else '_' for c in request.prompt[:30])
        filename = f"mesh_{safe_filename}_{uuid.uuid4().hex[:8]}.stl"
        filepath = DXFExporter.export_mesh_to_stl(verts, faces, filename)
        
        return {
            "success": True,
            "method": method_used,
            "vertices": len(verts),
            "faces": len(faces),
            "url": f"/exports/{os.path.basename(filepath)}",
            "filename": filename,
            "message": f"Generated using {method_used} with {len(verts)} vertices and {len(faces)} faces"
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/poisson-mesh")
async def poisson_mesh_from_points(request: ExportPointCloudRequest):
    """Apply Poisson surface reconstruction to an existing point cloud."""
    global multimodel_service
    try:
        if multimodel_service is None:
            from multimodel_service import MultiModelService
            multimodel_service = MultiModelService()
        
        formatted_points = []
        for p in request.points:
            formatted_points.append({
                "pos": [p.get("x", 0), p.get("y", 0), p.get("z", 0)],
                "color": [p.get("r", 0.5), p.get("g", 0.6), p.get("b", 0.8)]
            })
        
        import concurrent.futures
        def reconstruct():
            return multimodel_service.point_cloud_to_mesh_poisson(formatted_points)
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            verts, faces = await loop.run_in_executor(pool, reconstruct)
        
        safe_filename = "".join(c if c.isalnum() or c in '-_' else '_' for c in request.filename)
        filename = f"{safe_filename}_poisson.stl"
        filepath = DXFExporter.export_mesh_to_stl(verts, faces, filename)
        
        return {
            "success": True,
            "method": "poisson",
            "vertices": len(verts),
            "faces": len(faces),
            "url": f"/exports/{os.path.basename(filepath)}",
            "filename": filename
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
    from fastapi.staticfiles import StaticFiles
    os.makedirs('exports', exist_ok=True)
    app.mount("/exports", StaticFiles(directory="exports"), name="exports")
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
