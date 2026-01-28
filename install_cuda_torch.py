"""
Script to install PyTorch with CUDA 12.1 support.
Run this with: python install_cuda_torch.py
"""
import subprocess
import sys

def main():
    print("Installing PyTorch with CUDA 12.1 support...")
    print("This will take several minutes due to the large download size (~2.5GB)")
    
    # Use subprocess to run pip with the correct index URL
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--force-reinstall"
    ], check=False)
    
    if result.returncode == 0:
        print("\n✓ PyTorch with CUDA 12.1 installed successfully!")
        # Verify installation
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n✗ Installation failed with code {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
