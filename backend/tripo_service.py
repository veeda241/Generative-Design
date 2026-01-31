"""
Tripo AI 3D Generation Service
High-quality text-to-3D and image-to-3D generation using Tripo AI API.
"""

import os
import asyncio
import aiohttp
import logging
import time
import base64
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TripoService:
    """Service for generating 3D models using Tripo AI API."""
    
    BASE_URL = "https://api.tripo3d.ai/v2/openapi"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tripo service with API key."""
        self.api_key = api_key or os.getenv("TRIPO_API_KEY")
        if not self.api_key:
            logger.warning("TRIPO_API_KEY not set. Tripo AI features will be limited.")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        # Output directory for generated models
        self.output_dir = Path(__file__).parent / "exports"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Tripo AI Service initialized")
    
    async def text_to_3d(
        self,
        prompt: str,
        model_version: str = "v2.0-20240919",
        negative_prompt: Optional[str] = None,
        face_limit: int = 50000,
        texture: bool = True,
        pbr: bool = True,
        output_format: str = "glb"
    ) -> Dict[str, Any]:
        """
        Generate a 3D model from text prompt.
        
        Args:
            prompt: Text description of the 3D model
            model_version: Tripo model version to use
            negative_prompt: Things to avoid in generation
            face_limit: Maximum number of faces in the mesh
            texture: Whether to generate textures
            pbr: Whether to generate PBR materials
            output_format: Output format (glb, fbx, usdz, obj, stl)
        
        Returns:
            Dict with task_id, status, and model_url when complete
        """
        if not self.api_key:
            return {"error": "TRIPO_API_KEY not configured", "status": "failed"}
        
        try:
            # Step 1: Create the generation task
            task_data = {
                "type": "text_to_model",
                "prompt": prompt,
                "model_version": model_version,
                "face_limit": face_limit,
                "texture": texture,
                "pbr": pbr
            }
            
            if negative_prompt:
                task_data["negative_prompt"] = negative_prompt
            
            async with aiohttp.ClientSession() as session:
                # Create task
                async with session.post(
                    f"{self.BASE_URL}/task",
                    json=task_data,
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Tripo API error: {error_text}")
                        return {"error": f"API error: {response.status}", "status": "failed"}
                    
                    result = await response.json()
                    
                    if result.get("code") != 0:
                        return {"error": result.get("message", "Unknown error"), "status": "failed"}
                    
                    task_id = result["data"]["task_id"]
                    logger.info(f"Created Tripo task: {task_id}")
                
                # Step 2: Poll for completion
                model_data = await self._poll_task(session, task_id)
                
                if model_data.get("status") == "success":
                    # Step 3: Download the model
                    model_url = model_data.get("model_url")
                    if model_url:
                        file_path = await self._download_model(
                            session, model_url, task_id, output_format
                        )
                        model_data["local_path"] = str(file_path)
                
                return model_data
                
        except Exception as e:
            logger.error(f"Error in text_to_3d: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def image_to_3d(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        model_version: str = "v2.0-20240919",
        face_limit: int = 50000,
        texture: bool = True,
        pbr: bool = True,
        output_format: str = "glb"
    ) -> Dict[str, Any]:
        """
        Generate a 3D model from an image.
        
        Args:
            image_path: Local path to image file
            image_url: URL of the image
            image_base64: Base64 encoded image
            model_version: Tripo model version
            face_limit: Maximum faces in mesh
            texture: Generate textures
            pbr: Generate PBR materials
            output_format: Output format
        
        Returns:
            Dict with task_id, status, and model_url when complete
        """
        if not self.api_key:
            return {"error": "TRIPO_API_KEY not configured", "status": "failed"}
        
        try:
            # Prepare image data
            if image_path:
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                    # Detect mime type
                    ext = Path(image_path).suffix.lower()
                    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
                    mime_type = mime_map.get(ext, "image/png")
                    image_token = f"data:{mime_type};base64,{image_data}"
            elif image_base64:
                image_token = image_base64 if image_base64.startswith("data:") else f"data:image/png;base64,{image_base64}"
            elif image_url:
                image_token = image_url
            else:
                return {"error": "No image provided", "status": "failed"}
            
            task_data = {
                "type": "image_to_model",
                "file": {"type": "png" if "png" in image_token else "jpg", "url": image_token} if image_url else None,
                "model_version": model_version,
                "face_limit": face_limit,
                "texture": texture,
                "pbr": pbr
            }
            
            # For base64, use different format
            if not image_url:
                task_data["file"] = {
                    "type": "png",
                    "file_token": image_token
                }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/task",
                    json=task_data,
                    headers=self.headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Tripo API error: {error_text}")
                        return {"error": f"API error: {response.status}", "status": "failed"}
                    
                    result = await response.json()
                    
                    if result.get("code") != 0:
                        return {"error": result.get("message", "Unknown error"), "status": "failed"}
                    
                    task_id = result["data"]["task_id"]
                    logger.info(f"Created Tripo image task: {task_id}")
                
                # Poll for completion
                model_data = await self._poll_task(session, task_id)
                
                if model_data.get("status") == "success":
                    model_url = model_data.get("model_url")
                    if model_url:
                        file_path = await self._download_model(
                            session, model_url, task_id, output_format
                        )
                        model_data["local_path"] = str(file_path)
                
                return model_data
                
        except Exception as e:
            logger.error(f"Error in image_to_3d: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _poll_task(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        max_wait: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """Poll task status until completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            async with session.get(
                f"{self.BASE_URL}/task/{task_id}",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    await asyncio.sleep(poll_interval)
                    continue
                
                result = await response.json()
                
                if result.get("code") != 0:
                    return {"error": result.get("message"), "status": "failed", "task_id": task_id}
                
                data = result.get("data", {})
                status = data.get("status")
                progress = data.get("progress", 0)
                
                logger.info(f"Task {task_id}: {status} ({progress}%)")
                
                if status == "success":
                    output = data.get("output", {})
                    return {
                        "status": "success",
                        "task_id": task_id,
                        "model_url": output.get("model"),
                        "rendered_image": output.get("rendered_image"),
                        "pbr_model": output.get("pbr_model"),
                        "base_model": output.get("base_model")
                    }
                elif status == "failed":
                    return {
                        "status": "failed",
                        "task_id": task_id,
                        "error": data.get("message", "Task failed")
                    }
                elif status in ["queued", "running"]:
                    await asyncio.sleep(poll_interval)
                else:
                    await asyncio.sleep(poll_interval)
        
        return {"status": "timeout", "task_id": task_id, "error": "Task timed out"}
    
    async def _download_model(
        self,
        session: aiohttp.ClientSession,
        model_url: str,
        task_id: str,
        output_format: str
    ) -> Path:
        """Download the generated model."""
        file_path = self.output_dir / f"tripo_{task_id}.{output_format}"
        
        async with session.get(model_url) as response:
            if response.status == 200:
                content = await response.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                logger.info(f"Downloaded model to {file_path}")
            else:
                logger.error(f"Failed to download model: {response.status}")
        
        return file_path
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get current API balance/credits."""
        if not self.api_key:
            return {"error": "TRIPO_API_KEY not configured"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/user/balance",
                    headers=self.headers
                ) as response:
                    result = await response.json()
                    return result.get("data", {})
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {"error": str(e)}
    
    def convert_to_stl(self, glb_path: str, stl_path: str) -> bool:
        """Convert GLB to STL format using trimesh."""
        try:
            import trimesh
            
            # Load GLB
            mesh = trimesh.load(glb_path)
            
            # Handle scene vs single mesh
            if isinstance(mesh, trimesh.Scene):
                # Combine all meshes in scene
                meshes = []
                for name, geometry in mesh.geometry.items():
                    if isinstance(geometry, trimesh.Trimesh):
                        meshes.append(geometry)
                if meshes:
                    combined = trimesh.util.concatenate(meshes)
                else:
                    logger.error("No meshes found in GLB scene")
                    return False
            else:
                combined = mesh
            
            # Export as STL
            combined.export(stl_path, file_type='stl')
            logger.info(f"Converted {glb_path} to {stl_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting to STL: {e}")
            return False


# Singleton instance
_tripo_service: Optional[TripoService] = None

def get_tripo_service() -> TripoService:
    """Get or create the Tripo service singleton."""
    global _tripo_service
    if _tripo_service is None:
        _tripo_service = TripoService()
    return _tripo_service


# Convenience functions
async def generate_3d_from_text(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate 3D model from text prompt."""
    service = get_tripo_service()
    return await service.text_to_3d(prompt, **kwargs)


async def generate_3d_from_image(image_path: str = None, image_url: str = None, **kwargs) -> Dict[str, Any]:
    """Generate 3D model from image."""
    service = get_tripo_service()
    return await service.image_to_3d(image_path=image_path, image_url=image_url, **kwargs)
