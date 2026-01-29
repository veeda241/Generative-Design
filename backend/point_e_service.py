import torch
import os
from tqdm.auto import tqdm
import numpy as np
import logging
import gc

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from point_e.util.pc_to_mesh import marching_cubes_mesh
import time

CACHE_DIR = r'c:\hackathon\Gemini_CLI\Generative-Design\point_e_cache_fixed'
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointEService:
    def __init__(self, device=None, quality='high'):
        """
        Initialize Point-E service with fine-tuned settings for exact 3D modeling.
        
        Args:
            device: torch device ('cuda', 'cpu', or None for auto)
            quality: 'fast' (base40m only, 1024 pts), 
                     'normal' (base40m + upsample, 4096 pts), 
                     'high' (enhanced settings, 8192 pts) - DEFAULT
                     'ultra' (maximum quality, 16384 pts, higher guidance)
        """
        # Determine device
        env_device = os.getenv('POINT_E_DEVICE')
        if env_device:
            self.device = torch.device(env_device)
            logger.info(f"Using device from environment: {self.device}")
        elif device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.quality = quality
        self.base_model = None
        self.base_diffusion = None
        self.upsample_model = None
        self.upsample_diffusion = None
        self.sampler = None
        
        # Quality presets for exact 3D modeling
        # Note: base40M-textvec has a fixed context of 1024 points for base model
        # Upsampler can add more points (4x upsample)
        self.quality_presets = {
            'fast': {'base_points': 1024, 'total_points': 1024, 'guidance_scale': 3.0, 'use_upsample': False},
            'normal': {'base_points': 1024, 'total_points': 4096, 'guidance_scale': 5.0, 'use_upsample': True},
            'high': {'base_points': 1024, 'total_points': 4096, 'guidance_scale': 7.5, 'use_upsample': True},
            'ultra': {'base_points': 1024, 'total_points': 4096, 'guidance_scale': 10.0, 'use_upsample': True}
        }
        
        logger.info(f"Initializing Point-E on {self.device} with quality='{quality}' for exact 3D modeling...")
        self._load_models()
    
    def _load_models(self):
        """Load Point-E models with fine-tuned settings for exact 3D modeling."""
        try:
            preset = self.quality_presets.get(self.quality, self.quality_presets['high'])
            
            # Use base40M-textvec for text-to-3D generation
            self.base_name = 'base40M-textvec'
            
            logger.info(f"Loading {self.base_name} base model for exact 3D modeling...")
            self.base_model = model_from_config(MODEL_CONFIGS[self.base_name], self.device)
            self.base_model.eval()
            self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
            self.base_model.load_state_dict(load_checkpoint(self.base_name, self.device, cache_dir=CACHE_DIR))
            
            # Load upsample model for higher quality
            if preset['use_upsample']:
                logger.info("Loading upsample model for enhanced detail...")
                self.upsample_name = 'upsample'
                self.upsample_model = model_from_config(MODEL_CONFIGS[self.upsample_name], self.device)
                self.upsample_model.eval()
                self.upsample_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.upsample_name])
                self.upsample_model.load_state_dict(load_checkpoint(self.upsample_name, self.device, cache_dir=CACHE_DIR))
                
                # Calculate point distribution for upsampling
                upsample_points = preset['total_points'] - preset['base_points']
                
                self.sampler = PointCloudSampler(
                    device=self.device,
                    models=[self.base_model, self.upsample_model],
                    diffusions=[self.base_diffusion, self.upsample_diffusion],
                    num_points=[preset['base_points'], upsample_points],
                    aux_channels=['R', 'G', 'B'],
                    guidance_scale=[preset['guidance_scale'], preset['guidance_scale'] * 0.3],
                    use_karras=[True, True],  # Use Karras noise schedule
                    karras_steps=[64, 64],  # More diffusion steps for higher quality
                    sigma_min=[1e-3, 1e-3],  # Fine-tuned noise parameters
                    sigma_max=[120.0, 120.0],
                    s_churn=[3.0, 0.0],  # Add stochasticity for better details
                )
            else:
                self.sampler = PointCloudSampler(
                    device=self.device,
                    models=[self.base_model],
                    diffusions=[self.base_diffusion],
                    num_points=[preset['base_points']],
                    aux_channels=['R', 'G', 'B'],
                    guidance_scale=[preset['guidance_scale']],
                    use_karras=[True],
                    karras_steps=[64],
                    sigma_min=[1e-3],
                    sigma_max=[120.0],
                    s_churn=[3.0],
                )
            
            logger.info(f"✓ Point-E models loaded successfully (quality={self.quality}, points={preset['total_points']}, guidance={preset['guidance_scale']})")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    def generate_point_cloud(self, prompt: str, num_samples: int = 1, enhance_prompt: bool = True):
        """
        Generate high-quality point cloud from text prompt for exact 3D modeling.
        
        Args:
            prompt: Text description of object to generate
            num_samples: Number of samples to generate (default 1)
            enhance_prompt: Whether to enhance the prompt for better 3D generation
            
        Returns:
            List of dicts with 'pos' and 'color' for each point
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Enhance prompt for better 3D generation
        if enhance_prompt:
            enhanced_prompt = self._enhance_prompt_for_3d(prompt)
        else:
            enhanced_prompt = prompt
        
        logger.info(f"Generating {num_samples} high-quality point cloud(s) for: {enhanced_prompt}")
        start_time = time.time()
        
        try:
            # Generate samples with fine-tuned settings
            samples = None
            for x in tqdm(self.sampler.sample_batch_progressive(
                batch_size=num_samples,
                model_kwargs=dict(texts=[enhanced_prompt] * num_samples)
            )):
                samples = x

            if samples is None:
                raise RuntimeError("Failed to generate samples")

            # Process the first sample (primary output)
            pc = self.sampler.output_to_point_clouds(samples)[0]
            
            # Convert to serializable format for frontend
            points = self._point_cloud_to_list(pc)
            
            # Apply post-processing for cleaner geometry
            points = self._post_process_points(points)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Generated {len(points)} high-quality points in {elapsed:.2f}s")
            
            return points
            
        except Exception as e:
            logger.error(f"Point cloud generation failed: {str(e)}")
            raise
        finally:
            # Clean up memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    def _enhance_prompt_for_3d(self, prompt: str) -> str:
        """
        Enhance the prompt with 3D-specific descriptors for better generation.
        """
        # Add 3D quality descriptors if not already present
        quality_terms = ['3d model', 'detailed', 'high quality', 'precise geometry']
        prompt_lower = prompt.lower()
        
        enhancements = []
        if '3d' not in prompt_lower:
            enhancements.append('3D model of')
        if 'detailed' not in prompt_lower and 'detail' not in prompt_lower:
            enhancements.append('highly detailed')
        if 'quality' not in prompt_lower:
            enhancements.append('high quality')
        
        if enhancements:
            enhanced = f"{', '.join(enhancements)} {prompt}, precise geometry, clean surfaces"
        else:
            enhanced = f"{prompt}, precise geometry, clean surfaces"
        
        logger.info(f"Enhanced prompt: {enhanced}")
        return enhanced
    
    def _post_process_points(self, points: list) -> list:
        """
        Post-process point cloud for cleaner, more exact geometry.
        Applies outlier removal and point normalization.
        """
        if len(points) < 100:
            return points
        
        coords = np.array([p['pos'] for p in points])
        colors = np.array([p['color'] for p in points])
        
        # Calculate centroid and center the point cloud
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        
        # Remove outliers (points beyond 2.5 standard deviations)
        distances = np.linalg.norm(coords_centered, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 2.5 * std_dist
        
        valid_mask = distances < threshold
        coords_filtered = coords_centered[valid_mask]
        colors_filtered = colors[valid_mask]
        
        # Normalize to unit sphere for consistent scaling
        max_dist = np.max(np.linalg.norm(coords_filtered, axis=1))
        if max_dist > 0:
            coords_normalized = coords_filtered / max_dist
        else:
            coords_normalized = coords_filtered
        
        # Reconstruct points list
        processed_points = []
        for i in range(len(coords_normalized)):
            processed_points.append({
                'pos': coords_normalized[i].tolist(),
                'color': colors_filtered[i].tolist()
            })
        
        logger.info(f"Post-processing: {len(points)} -> {len(processed_points)} points (removed {len(points) - len(processed_points)} outliers)")
        return processed_points

    def _point_cloud_to_list(self, pc):
        """Convert Point-E point cloud to serializable list format."""
        coords = pc.coords.cpu().numpy() if hasattr(pc.coords, 'cpu') else np.array(pc.coords)
        colors = np.stack([
            pc.channels['R'].cpu().numpy() if hasattr(pc.channels['R'], 'cpu') else np.array(pc.channels['R']),
            pc.channels['G'].cpu().numpy() if hasattr(pc.channels['G'], 'cpu') else np.array(pc.channels['G']),
            pc.channels['B'].cpu().numpy() if hasattr(pc.channels['B'], 'cpu') else np.array(pc.channels['B']),
        ], axis=-1)
        
        points = []
        for i in range(len(coords)):
            points.append({
                "pos": coords[i].tolist(),
                "color": colors[i].tolist()
            })
        return points

    def get_point_cloud_mesh(self, point_cloud, level=0.5):
        """
        Convert point cloud to mesh using marching cubes.
        
        Args:
            point_cloud: Point cloud from generate_point_cloud
            level: Isosurface level for marching cubes
            
        Returns:
            vertices and faces of the mesh
        """
        try:
            # Reconstruct mesh from point cloud
            coords = np.array([p['pos'] for p in point_cloud])
            mesh = marching_cubes_mesh(self.device, coords, level=level)
            return mesh
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Test execution
    service = PointEService()
    points = service.generate_point_cloud("a red industrial pump")
    print(f"Sample point: {points[0]}")
