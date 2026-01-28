import torch
import os
import sys

def check():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
    try:
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        print("Point-E library found.")
    except Exception as e:
        print(f"Point-E library NOT found: {e}")

if __name__ == "__main__":
    check()
