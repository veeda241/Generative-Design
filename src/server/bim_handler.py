"""
IFC/BIM File Handler Service
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import ifcopenshell
    IFC_AVAILABLE = True
except ImportError:
    IFC_AVAILABLE = False
    logger.warning("ifcopenshell not installed. IFC support disabled.")

class IFCHandler:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.cache_dir = self.upload_dir / "ifc_cache"
        self.cache_dir.mkdir(exist_ok=True)
    def load_ifc(self, file_path: str) -> Dict[str, Any]:
        if not IFC_AVAILABLE:
            return {"error": "IFC support not available"}
        try:
            ifc_file = ifcopenshell.open(file_path)
            file_info = {"file": Path(file_path).name, "size": os.path.getsize(file_path), "format": ifc_file.schema}
            elements = self._extract_elements(ifc_file)
            return {"success": True, "file_info": file_info, "elements": elements, "statistics": {"total_elements": len(elements)}}
        except Exception as e:
            return {"error": str(e)}
    def _extract_elements(self, ifc_file) -> List[Dict[str, Any]]:
        elements = []
        try:
            products = ifc_file.by_type("IfcProduct")
            for product in products[:100]:
                elements.append({"id": product.id(), "name": product.Name or "Unnamed", "type": product.is_a()})
        except:
            pass
        return elements
    def convert_to_threejs(self, file_path: str) -> Optional[str]:
        if not IFC_AVAILABLE:
            return None
        try:
            ifc_file = ifcopenshell.open(file_path)
            geometries = []
            products = ifc_file.by_type("IfcProduct")
            for product in products[:50]:
                if hasattr(product, "Representation") and product.Representation:
                    geometries.append({"name": product.Name or "Unnamed", "type": product.is_a(), "id": product.id()})
            output_file = self.cache_dir / f"{Path(file_path).stem}.json"
            with open(output_file, 'w') as f:
                json.dump({"version": "1.0", "geometries": geometries, "count": len(geometries)}, f, indent=2)
            return str(output_file)
        except Exception as e:
            logger.error(f"IFC to Three.js conversion error: {str(e)}")
            return None
    def get_element_properties(self, file_path: str, element_id: int) -> Dict[str, Any]:
        if not IFC_AVAILABLE:
            return {}
        try:
            ifc_file = ifcopenshell.open(file_path)
            element = ifc_file.by_id(element_id)
            return {"name": element.Name or "Unnamed", "type": element.is_a(), "id": element.id()}
        except Exception as e:
            return {}

bim_handler = IFCHandler()
