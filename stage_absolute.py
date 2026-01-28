import os
import shutil
import requests

def stage():
    base_dir = r"c:\hackathon\Gemini_CLI\Generative-Design"
    backend_dir = os.path.join(base_dir, 'backend')
    new_cache = os.path.join(backend_dir, 'point_e_cache_fixed')
    old_cache = os.path.join(backend_dir, 'point_e_model_cache')
    root_cache = os.path.join(base_dir, 'point_e_model_cache')
    
    print(f"Target directory: {new_cache}")
    os.makedirs(new_cache, exist_ok=True)
    
    # Files to try and find
    models = ['base_40m.pt', 'ViT-L-14.pt']
    
    for model in models:
        dst = os.path.join(new_cache, model)
        if os.path.exists(dst):
            print(f"{model} already in destination.")
            continue
            
        # Try finding in backend old cache
        src = os.path.join(old_cache, model)
        if os.path.exists(src):
             print(f"Found {model} in backend cache. Copying...")
             try:
                 shutil.copy2(src, dst)
                 continue
             except Exception as e:
                 print(f"Failed to copy from backend cache: {e}")

        # Try finding in root cache
        src = os.path.join(root_cache, model)
        if os.path.exists(src):
             print(f"Found {model} in root cache. Copying...")
             try:
                 shutil.copy2(src, dst)
                 continue
             except Exception as e:
                 print(f"Failed to copy from root cache: {e}")

    # Final check for upsample model
    upsample_url = "https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt"
    upsample_dst = os.path.join(new_cache, 'upsample_40m.pt')
    
    if not os.path.exists(upsample_dst):
        print(f"Downloading upsample model to {upsample_dst}...")
        try:
            r = requests.get(upsample_url, stream=True, timeout=60)
            r.raise_for_status()
            with open(upsample_dst, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download successful.")
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print("Upsample model ready.")

if __name__ == "__main__":
    stage()
