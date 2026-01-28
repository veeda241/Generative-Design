import torch
import sys
import os
from dotenv import load_dotenv

# Load env to get POINT_E_DEVICE
load_dotenv(dotenv_path='backend/.env')

def preload():
    print(f"Python executable: {sys.executable}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA is not available. Check your installation.")
        return False

    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Import service
    sys.path.append('backend')
    from point_e_service import PointEService
    
    device = os.getenv('POINT_E_DEVICE', 'cuda')
    print(f"Initializing Point-E Service on {device}...")
    
    try:
        # This will trigger the downloads
        service = PointEService(device=device, quality='fast')
        print("Point-E 'fast' models loaded successfully.")
        
        service = PointEService(device=device, quality='normal')
        print("Point-E 'normal' models (including upsample) loaded successfully.")
        
        print("ALL MODELS DOWNLOADED AND READY ON GPU.")
        return True
    except Exception as e:
        print(f"Error during preloading: {e}")
        return False

if __name__ == "__main__":
    preload()
