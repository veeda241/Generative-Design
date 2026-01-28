import os

def migrate():
    src_dir = r"c:\hackathon\Gemini_CLI\Generative-Design\backend\point_e_model_cache"
    dst_dir = r"c:\hackathon\Gemini_CLI\Generative-Design\point_e_cache_fixed"
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    
    models = ['base_40m.pt', 'ViT-L-14.pt']
    for model in models:
        src = os.path.join(src_dir, model)
        dst = os.path.join(dst_dir, model)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Migrating {model}...")
            try:
                with open(src, 'rb') as f_src:
                    with open(dst, 'wb') as f_dst:
                        f_dst.write(f_src.read())
                print(f"Successfully migrated {model}")
            except Exception as e:
                print(f"Failed to migrate {model}: {e}")
        else:
            print(f"Skipping {model} (not found or already in destination)")

if __name__ == "__main__":
    migrate()
