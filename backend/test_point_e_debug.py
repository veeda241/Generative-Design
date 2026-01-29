"""Debug Point-E model loading."""
import os
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Try importing Point-E
try:
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.models.download import load_checkpoint
    print("\n✓ Point-E imports successful")
except Exception as e:
    print(f"\n✗ Point-E import failed: {e}")
    sys.exit(1)

# Check available models
print(f"\nAvailable models: {list(MODEL_CONFIGS.keys())}")
print(f"Available diffusion configs: {list(DIFFUSION_CONFIGS.keys())}")

# Try loading base40M-textvec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

CACHE_DIR = r'c:\hackathon\Gemini_CLI\Generative-Design\point_e_cache_fixed'
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    print("\nLoading base40M-textvec model...")
    base_model = model_from_config(MODEL_CONFIGS['base40M-textvec'], device)
    base_model.eval()
    print("✓ Model config loaded")
    
    print("Loading checkpoint...")
    state_dict = load_checkpoint('base40M-textvec', device, cache_dir=CACHE_DIR)
    base_model.load_state_dict(state_dict)
    print("✓ Checkpoint loaded successfully!")
    
    print("\nLoading diffusion...")
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['base40M-textvec'])
    print("✓ Diffusion loaded")
    
    # Create sampler
    print("\nCreating sampler...")
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0],
        use_karras=[True],
        karras_steps=[32],
        sigma_min=[1e-3],
        sigma_max=[120.0],
        s_churn=[3.0],  # Required parameter
    )
    print("✓ Sampler created")
    
    # Test generation
    print("\nTesting point cloud generation (this may take a moment)...")
    prompt = "a cube"
    for x in sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(texts=[prompt])
    ):
        samples = x
    
    coords = samples[:, :, :3].cpu().numpy()
    print(f"✓ Generated {coords.shape[1]} points!")
    print(f"  Coordinates range: x=[{coords[0,:,0].min():.2f}, {coords[0,:,0].max():.2f}]")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
