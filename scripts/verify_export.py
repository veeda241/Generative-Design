import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'server')))

from exporter import DXFExporter

def test_stl_export():
    print("Testing STL export...")
    # Create a simple cube mesh
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Front
        [1, 2, 6], [1, 6, 5], # Right
        [2, 3, 7], [2, 7, 6], # Back
        [3, 0, 4], [3, 4, 7]  # Left
    ], dtype=np.uint32)
    
    filename = "test_cube.stl"
    filepath = DXFExporter.export_mesh_to_stl(verts, faces, filename)
    
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ STL file created at: {filepath}")
        print(f"✓ File size: {size} bytes")
        if size > 84: # 80 bytes header + 4 bytes face count
            print("✓ File contains data.")
            return True
    else:
        print("✗ STL file was not created.")
        return False

def test_obj_mesh_export():
    print("Testing OBJ mesh export...")
    verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    faces = np.array([[0,1,2]], dtype=np.uint32)
    
    filename = "test_mesh.obj"
    filepath = DXFExporter.export_mesh_to_obj(verts, faces, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            if 'v ' in content and 'f ' in content:
                print(f"✓ OBJ mesh file created and verified at: {filepath}")
                return True
    print("✗ OBJ mesh verification failed.")
    return False

if __name__ == "__main__":
    s1 = test_stl_export()
    s2 = test_obj_mesh_export()
    if s1 and s2:
        print("\nALL BACKEND EXPORT TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
