"""
IFC/BIM File Handler Service
Handles IFC file uploads and processing using ifcopenshell
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
    """Handle IFC file processing and BIM data extraction"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.cache_dir = self.upload_dir / "ifc_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_ifc(self, file_path: str) -> Dict[str, Any]:
        """
        Load and parse IFC file
        
        Args:
            file_path: Path to IFC file
            
        Returns:
            Dictionary containing BIM data
        """
        if not IFC_AVAILABLE:
            logger.error("ifcopenshell not available")
            return {"error": "IFC support not available"}
        
        try:
            ifc_file = ifcopenshell.open(file_path)
            
            # Extract basic info
            file_info = {
                "file": Path(file_path).name,
                "size": os.path.getsize(file_path),
                "format": ifc_file.schema,
                "project": self._extract_project_info(ifc_file)
            }
            
            # Extract BIM elements
            elements = self._extract_elements(ifc_file)
            
            # Extract spatial hierarchy
            spatial_structure = self._extract_spatial_structure(ifc_file)
            
            # Extract properties
            properties = self._extract_properties(ifc_file)
            
            return {
                "success": True,
                "file_info": file_info,
                "elements": elements,
                "spatial_structure": spatial_structure,
                "properties": properties,
                "statistics": {
                    "total_elements": len(elements),
                    "unique_types": len(set(e.get("type") for e in elements)),
                    "walls": len([e for e in elements if e.get("type") == "IfcWall"]),
                    "doors": len([e for e in elements if e.get("type") == "IfcDoor"]),
                    "windows": len([e for e in elements if e.get("type") == "IfcWindow"]),
                    "furniture": len([e for e in elements if e.get("type") == "IfcFurniture"]),
                }
            }
        except Exception as e:
            logger.error(f"Error loading IFC file: {str(e)}")
            return {"error": str(e)}
    
    def _extract_project_info(self, ifc_file) -> Dict[str, str]:
        """Extract project information from IFC"""
        try:
            project = ifc_file.by_type("IfcProject")[0]
            return {
                "name": project.Name or "Unnamed Project",
                "description": project.Description or "No description",
                "schema": ifc_file.schema
            }
        except:
            return {"name": "Unknown", "schema": ifc_file.schema}
    
    def _extract_elements(self, ifc_file) -> List[Dict[str, Any]]:
        """Extract BIM elements"""
        elements = []
        
        # Get all spatial elements
        spatial_elements = ifc_file.by_type("IfcSpatialElement")
        
        for element in spatial_elements:
            elem_data = {
                "id": element.id(),
                "name": element.Name or "Unnamed",
                "type": element.is_a(),
                "description": element.Description or ""
            }
            elements.append(elem_data)
        
        # Get products (walls, doors, windows, etc.)
        products = ifc_file.by_type("IfcProduct")
        
        for product in products[:100]:  # Limit to first 100 for performance
            prod_data = {
                "id": product.id(),
                "name": product.Name or "Unnamed",
                "type": product.is_a(),
                "description": getattr(product, "Description", "") or ""
            }
            
            # Get placement if available
            if hasattr(product, "ObjectPlacement") and product.ObjectPlacement:
                try:
                    placement = product.ObjectPlacement
                    if hasattr(placement, "RelativePlacement"):
                        location = placement.RelativePlacement.Location
                        prod_data["location"] = [location.Coordinates[i] for i in range(3)]
                except:
                    pass
            
            elements.append(prod_data)
        
        return elements
    
    def _extract_spatial_structure(self, ifc_file) -> Dict[str, Any]:
        """Extract spatial hierarchy"""
        try:
            project = ifc_file.by_type("IfcProject")[0]
            sites = ifc_file.by_type("IfcSite")
            buildings = ifc_file.by_type("IfcBuilding")
            storeys = ifc_file.by_type("IfcBuildingStorey")
            
            return {
                "project": project.Name or "Project",
                "sites": len(sites),
                "buildings": len(buildings),
                "storeys": len(storeys),
                "site_names": [s.Name for s in sites],
                "building_names": [b.Name for b in buildings],
                "storey_names": [s.Name for s in storeys]
            }
        except:
            return {"project": "Unknown", "buildings": 0, "storeys": 0}
    
    def _extract_properties(self, ifc_file) -> Dict[str, List[str]]:
        """Extract properties and property sets"""
        properties = {}
        
        try:
            prop_sets = ifc_file.by_type("IfcPropertySet")
            for prop_set in prop_sets[:20]:  # Limit for performance
                props = []
                if hasattr(prop_set, "HasProperties"):
                    for prop in prop_set.HasProperties:
                        if hasattr(prop, "Name"):
                            props.append(prop.Name)
                if props:
                    properties[prop_set.Name or "Unknown"] = props
        except:
            pass
        
        return properties
    
    def convert_to_threejs(self, file_path: str) -> Optional[str]:
        """
        Convert IFC to Three.js compatible format (JSON)
        
        Args:
            file_path: Path to IFC file
            
        Returns:
            Path to converted file or None
        """
        if not IFC_AVAILABLE:
            return None
        
        try:
            ifc_file = ifcopenshell.open(file_path)
            
            # Create Three.js compatible geometry data
            geometries = []
            
            # Extract geometry from products
            products = ifc_file.by_type("IfcProduct")
            
            for product in products[:50]:  # Limit for performance
                if hasattr(product, "Representation") and product.Representation:
                    geom_data = {
                        "name": product.Name or "Unnamed",
                        "type": product.is_a(),
                        "id": product.id()
                    }
                    geometries.append(geom_data)
            
            # Save to JSON
            output_file = self.cache_dir / f"{Path(file_path).stem}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "version": "1.0",
                    "metadata": {
                        "type": "BIM",
                        "generator": "AETHER-GEN IFC Handler"
                    },
                    "geometries": geometries,
                    "count": len(geometries)
                }, f, indent=2)
            
            return str(output_file)
        
        except Exception as e:
            logger.error(f"IFC to Three.js conversion error: {str(e)}")
            return None
    
    def get_element_properties(self, file_path: str, element_id: int) -> Dict[str, Any]:
        """Get detailed properties of a specific element"""
        if not IFC_AVAILABLE:
            return {}
        
        try:
            ifc_file = ifcopenshell.open(file_path)
            element = ifc_file.by_id(element_id)
            
            properties = {
                "name": element.Name or "Unnamed",
                "type": element.is_a(),
                "id": element.id(),
                "description": getattr(element, "Description", "") or ""
            }
            
            # Add type information
            if hasattr(element, "ObjectType"):
                properties["object_type"] = element.ObjectType or "Unknown"
            
            # Add quantity information
            if hasattr(element, "Quantities"):
                quantities = {}
                for qty in element.Quantities or []:
                    if hasattr(qty, "Name"):
                        quantities[qty.Name] = str(qty)
                if quantities:
                    properties["quantities"] = quantities
            
            return properties
        
        except Exception as e:
            logger.error(f"Error getting element properties: {str(e)}")
            return {}


# Initialize handler
bim_handler = IFCHandler()
