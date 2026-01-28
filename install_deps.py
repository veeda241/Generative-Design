import subprocess
import sys

def install():
    try:
        print("Installing dependencies for CUDA 12.1...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "torch", "torchvision", "torchaudio", "notebook", "torchmetrics", "--upgrade"
        ])
        print("Successfully installed core dependencies.")
        
        print("Installing from requirements.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"
        ])
        print("Successfully installed all dependencies.")
    except Exception as e:
        print(f"Error during installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install()
