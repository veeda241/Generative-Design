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
    def __init__(self, device=None, quality='normal'):
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
        logger.info(f"Initializing Point-E on {self.device} with quality='{quality}'...")
        self._load_models()
    def _load_models(self):
        try:
            if self.quality == 'fast':
                self.base_name = 'base40M-textvec'
            elif self.quality == 'high':
                self.base_name = 'base40M-textvec'
            else:
                self.base_name = 'base40M-textvec'
            logger.info(f"Loading {self.base_name} base model...")
            self.base_model = model_from_config(MODEL_CONFIGS[self.base_name], self.device)
            self.base_model.eval()
            self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
            self.base_model.load_state_dict(load_checkpoint(self.base_name, self.device, cache_dir=CACHE_DIR))
            if self.quality != 'fast':
                logger.info("Loading upsample model...")
                self.upsample_name = 'upsample'
                self.upsample_model = model_from_config(MODEL_CONFIGS[self.upsample_name], self.device)
                self.upsample_model.eval()
                self.upsample_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.upsample_name])
                self.upsample_model.load_state_dict(load_checkpoint(self.upsample_name, self.device, cache_dir=CACHE_DIR))
                self.sampler = PointCloudSampler(
                    device=self.device,
                    models=[self.base_model, self.upsample_model],
                    diffusions=[self.base_diffusion, self.upsample_diffusion],
                    num_points=[1024, 4096 - 1024],
                    aux_channels=['R', 'G', 'B'],
                    guidance_scale=[3.0, 0.0],
                )
            else:
                self.sampler = PointCloudSampler(
                    device=self.device,
                    models=[self.base_model],
                    diffusions=[self.base_diffusion],
                    num_points=[1024],
                    aux_channels=['R', 'G', 'B'],
                    guidance_scale=[3.0],
                )
            logger.info("✓ Point-E models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    def generate_point_cloud(self, prompt: str, num_samples: int = 1):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        logger.info(f"Generating {num_samples} point cloud(s) for: {prompt}")
        start_time = time.time()
        try:
            samples = None
            for x in tqdm(self.sampler.sample_batch_progressive(
                batch_size=num_samples,
                model_kwargs=dict(texts=[prompt] * num_samples)
            )):
                samples = x
            if samples is None:
                raise RuntimeError("Failed to generate samples")
            pc = self.sampler.output_to_point_clouds(samples)[0]
            points = self._point_cloud_to_list(pc)
            elapsed = time.time() - start_time
            logger.info(f"✓ Generated {len(points)} points in {elapsed:.2f}s")
            return points
        except Exception as e:
            logger.error(f"Point cloud generation failed: {str(e)}")
            raise
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    def _point_cloud_to_list(self, pc):
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
        try:
            coords = np.array([p['pos'] for p in point_cloud])
            mesh = marching_cubes_mesh(self.device, coords, level=level)
            return mesh
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            raise
