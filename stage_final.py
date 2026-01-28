import os
import shutil
import time

def stage():
    base_dir = r"c:\hackathon\Gemini_CLI\Generative-Design"
    src_dir = os.path.join(base_dir, 'backend', 'point_e_model_cache')
    dst_dir = os.path.join(base_dir, 'point_e_cache_fixed')
    
    print(f"Checking for source directory: {src_dir}")
    if not os.path.exists(src_dir):
        print("Source directory not found!")
        return

    print(f"Checking for destination directory: {dst_dir}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        print("Created destination directory.")

    models = ['base_40m.pt', 'ViT-L-14.pt']
    for model in models:
        src = os.path.join(src_dir, model)
        dst = os.path.join(dst_dir, model)
        if os.path.exists(src):
            print(f"Copying {model}...")
            try:
                shutil.copy2(src, dst)
                print(f"Successfully copied {model}")
            except Exception as e:
                print(f"Failed to copy {model}: {e}")
        else:
            print(f"Source file {model} not found.")

    print("FINISHED STAGING")

if __name__ == "__main__":
    stage()
