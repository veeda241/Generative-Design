"""Quick test of multimodel service"""
import sys
import traceback

try:
    print("Testing multimodel imports...")
    
    print("1. Testing torch...")
    import torch
    print(f"   torch: OK, CUDA: {torch.cuda.is_available()}")
    
    print("2. Testing open3d...")
    import open3d as o3d
    print(f"   open3d: OK, version: {o3d.__version__}")
    
    print("3. Testing shap_e...")
    from shap_e.models.download import load_model
    print("   shap_e: OK")
    
    print("4. Testing multimodel_service...")
    from multimodel_service import MultiModelService
    print("   multimodel_service: OK")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
