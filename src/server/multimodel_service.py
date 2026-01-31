"""
Multi-Model 3D Generation Service
=================================
Combines multiple models and methods for best 3D generation results:
1. TripoSR: Text → Image → High-Quality Mesh
2. Shap-E: Text → Direct Mesh (Fallback)
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelService:
    """Multi-model 3D generation service for high-quality mesh output."""
    
    def __init__(self, device=None):
        import torch
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"MultiModelService initializing on {self.device}")
        
        self.shap_e_loaded = False
        self.shap_e_models = {}
        self.triposr_service = None
        
    def _load_shap_e(self):
        """Lazy load Shap-E models."""
        if not self.shap_e_loaded:
            try:
                from shap_e.diffusion.sample import sample_latents
                from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
                from shap_e.models.download import load_model, load_config
                from shap_e.util.notebooks import decode_latent_mesh
                
                self.shap_e_models['xm'] = load_model('transmitter', device=self.device)
                self.shap_e_models['model'] = load_model('text300M', device=self.device)
                self.shap_e_models['diffusion'] = diffusion_from_config(load_config('diffusion'))
                self.shap_e_models['sample_latents'] = sample_latents
                self.shap_e_models['decode_latent_mesh'] = decode_latent_mesh
                
                self.shap_e_loaded = True
            except Exception as e:
                logger.error(f"Shap-E load failed: {e}")
                return False
        return True

    def _load_triposr(self):
        """Lazy load TripoSR service."""
        if self.triposr_service is None:
            try:
                from triposr_service import TripoSRService
                self.triposr_service = TripoSRService(device=str(self.device))
            except Exception as e:
                logger.error(f"TripoSR load failed: {e}")
                return False
        return True
    
    def generate_mesh_shap_e(self, prompt: str, guidance_scale: float = 20.0, steps: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Direct Text-to-Mesh using Shap-E."""
        if not self._load_shap_e():
            raise RuntimeError("Shap-E load failed.")
            
        refined_prompt = f"{prompt}, highly detailed, sharp edges, professional 3D model, engineering CAD style"
        latents = self.shap_e_models['sample_latents'](
            batch_size=1,
            model=self.shap_e_models['model'],
            diffusion=self.shap_e_models['diffusion'],
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[refined_prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        mesh = self.shap_e_models['decode_latent_mesh'](self.shap_e_models['xm'], latents[0]).tri_mesh()
        return np.array(mesh.verts, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)

    def generate_mesh_triposr(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """High-Quality Text-to-Image-to-3D using TripoSR."""
        if not self._load_triposr():
            raise RuntimeError("TripoSR load failed.")
        return self.triposr_service.generate_mesh(prompt)

    def generate_best_mesh(self, prompt: str, method: str = 'auto') -> Tuple[np.ndarray, np.ndarray, str]:
        """Orchestrate the best generation method."""
        if method == 'auto' or method == 'triposr':
            try:
                # TripoSR is the current "best" for clarity
                verts, faces = self.generate_mesh_triposr(prompt)
                return verts, faces, 'triposr'
            except Exception as e:
                logger.warning(f"TripoSR failed: {e}. Falling back to Shap-E.")
                method = 'shap-e'
        
        if method == 'shap-e':
            verts, faces = self.generate_mesh_shap_e(prompt)
            return verts, faces, 'shap-e'
            
        raise ValueError(f"Unsupported method: {method}")
