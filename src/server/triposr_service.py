"""
TripoSR 3D Generation Service
=============================
Uses a Text-to-Image model (SDXL Turbo) + TripoSR (Image-to-3D)
for high-quality, sharp 3D meshes.
"""

import os
import torch
import numpy as np
import logging
from PIL import Image
from typing import Tuple, List, Optional
import tempfile
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripoSRService:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"TripoSRService initializing on {self.device}")
        
        self.t2i_pipe = None
        self.tsr_model = None
        self.tsr_loaded = False
        
    def _load_models(self):
        """Lazy load T2I and TripoSR models."""
        if self.tsr_loaded:
            return True
            
        try:
            from diffusers import AutoPipelineForText2Image
            import torch
            
            # 1. Load Text-to-Image (SDXL Turbo for speed)
            logger.info("Loading SDXL Turbo for Text-to-Image...")
            self.t2i_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16 if str(self.device) == 'cuda' else torch.float32,
                variant="fp16" if str(self.device) == 'cuda' else None
            ).to(self.device)
            
            # 2. Add TripoSR path to sys.path
            repo_path = os.path.join(os.path.dirname(__file__), "triposr_repo")
            if os.path.exists(repo_path):
                sys.path.append(repo_path)
                
            # 3. Load TripoSR Model
            logger.info("Loading TripoSR model...")
            # We'll try to import from the repo if cloned, otherwise we use a fallback or wait
            try:
                from tsr.system import TSR
                from tsr.utils import remove_background, resize_foreground
                
                self.tsr_model = TSR.from_pretrained(
                    "stabilityai/TripoSR",
                    config_name="config.yaml",
                    weight_name="model.ckpt"
                ).to(self.device)
                
                self.tsr_loaded = True
                logger.info("✓ TripoSR and T2I loaded successfully")
                return True
            except ImportError:
                logger.warning("TripoSR code not found in path. Please ensure triposr_repo is present.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load TripoSR pipeline: {e}")
            return False

    def generate_mesh(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a high-quality mesh using the Text -> Image -> TripoSR pipeline.
        """
        if not self._load_models():
            raise RuntimeError("TripoSR models could not be loaded.")
            
        # 1. Generate Image from Text
        logger.info(f"Generating image for prompt: {prompt}")
        # Add 'white background, 3d model, centered' to improve TripoSR results
        enhanced_prompt = f"{prompt}, 3d model, professional engineering design, sharp edges, centered, white background"
        
        image = self.t2i_pipe(
            prompt=enhanced_prompt,
            num_inference_steps=1, # Turbo only needs 1 step
            guidance_scale=0.0    # Turbo guide is usually 0
        ).images[0]
        
        # Save image for debugging (optional)
        # image.save("last_t2i_generation.png")
        
        # 2. Process with TripoSR
        logger.info("Converting image to 3D mesh with TripoSR...")
        
        # We need to use the TripoSR repo's processing logic
        from tsr.utils import remove_background, resize_foreground
        
        # Preprocess image
        image = np.array(image)
        # Background removal (optional but recommended for TripoSR)
        try:
            import rembg
            image = rembg.remove(image)
        except:
            pass
            
        image = Image.fromarray(image)
        image = resize_foreground(image, 0.85) # Rescale to fit TripoSR's expectations
        
        # Inference
        with torch.no_grad():
            scene_codes = self.tsr_model(image, device=str(self.device))
            
        # Extract mesh
        # TripoSR's extract_mesh returns a list of meshes (one per batch)
        # We take the first one and convert to numpy
        meshes = self.tsr_model.extract_mesh(scene_codes)
        mesh = meshes[0]
        
        # Extract vertices and faces
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        logger.info(f"✓ TripoSR Mesh generated: {len(verts)} vertices, {len(faces)} faces")
        
        return verts, faces
