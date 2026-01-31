"""
Multi-Model 3D Generation Service
Combines Point-E (AI) and OpenSCAD (Parametric CAD) for superior 3D model generation.

Strategy:
1. Analyze prompt to determine best approach
2. Generate with both methods when appropriate
3. Merge/ensemble results for higher quality
4. Use OpenSCAD for precise geometry, Point-E for organic details
"""

import numpy as np
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
import re
import os

logger = logging.getLogger(__name__)

# Keywords for routing decisions
MECHANICAL_KEYWORDS = [
    'gear', 'cog', 'screw', 'bolt', 'nut', 'bearing', 'shaft', 'axle',
    'bracket', 'mount', 'flange', 'housing', 'enclosure', 'box', 'case',
    'pipe', 'tube', 'cylinder', 'cone', 'cube', 'sphere', 'prism',
    'hexagon', 'octagon', 'polygon', 'thread', 'groove', 'slot',
    'plate', 'panel', 'frame', 'beam', 'rod', 'bar', 'channel',
    'wheel', 'pulley', 'sprocket', 'cam', 'lever', 'hinge', 'joint',
    'connector', 'adapter', 'coupling', 'fitting', 'valve', 'nozzle'
]

ORGANIC_KEYWORDS = [
    'animal', 'creature', 'human', 'face', 'head', 'body', 'hand', 'foot',
    'tree', 'plant', 'flower', 'leaf', 'branch', 'root', 'organic',
    'sculpture', 'statue', 'figurine', 'character', 'monster', 'dragon',
    'cloud', 'wave', 'terrain', 'mountain', 'rock', 'stone', 'crystal',
    'abstract', 'artistic', 'freeform', 'curved', 'flowing', 'natural'
]

HYBRID_KEYWORDS = [
    'car', 'vehicle', 'robot', 'machine', 'device', 'tool', 'furniture',
    'chair', 'table', 'desk', 'lamp', 'bottle', 'cup', 'mug', 'vase',
    'building', 'house', 'architecture', 'bridge', 'tower', 'airplane',
    'ship', 'boat', 'train', 'motorcycle', 'bicycle'
]


class MultiModelService:
    """
    Multi-model 3D generation service combining Point-E and OpenSCAD.
    """
    
    def __init__(self):
        self.point_e_service = None
        self.openscad_service = None
        self._initialized = False
        
    def _lazy_init_point_e(self):
        """Lazy initialize Point-E service."""
        if self.point_e_service is None:
            try:
                from point_e_service import PointEService
                self.point_e_service = PointEService(quality='high')
                logger.info("✓ Point-E service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Point-E: {e}")
                self.point_e_service = None
                
    def _lazy_init_openscad(self):
        """Lazy initialize OpenSCAD service."""
        if self.openscad_service is None:
            try:
                from openscad_service import OpenSCADService
                self.openscad_service = OpenSCADService()
                logger.info("✓ OpenSCAD service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenSCAD: {e}")
                self.openscad_service = None
    
    def analyze_prompt(self, prompt: str) -> Dict:
        """
        Analyze prompt to determine the best generation strategy.
        
        Returns:
            Dict with 'strategy', 'mechanical_score', 'organic_score', 'components'
        """
        prompt_lower = prompt.lower()
        
        # Count keyword matches
        mechanical_matches = sum(1 for kw in MECHANICAL_KEYWORDS if kw in prompt_lower)
        organic_matches = sum(1 for kw in ORGANIC_KEYWORDS if kw in prompt_lower)
        hybrid_matches = sum(1 for kw in HYBRID_KEYWORDS if kw in prompt_lower)
        
        # Calculate scores (0-1)
        total = max(mechanical_matches + organic_matches + hybrid_matches, 1)
        mechanical_score = (mechanical_matches + hybrid_matches * 0.5) / total
        organic_score = (organic_matches + hybrid_matches * 0.5) / total
        
        # Determine strategy
        if mechanical_matches > organic_matches * 2:
            strategy = 'openscad_primary'
        elif organic_matches > mechanical_matches * 2:
            strategy = 'point_e_primary'
        elif hybrid_matches > 0 or (mechanical_matches > 0 and organic_matches > 0):
            strategy = 'hybrid'
        else:
            # Default to hybrid for unknown prompts
            strategy = 'hybrid'
        
        # Extract components for hybrid generation
        components = self._extract_components(prompt)
        
        return {
            'strategy': strategy,
            'mechanical_score': mechanical_score,
            'organic_score': organic_score,
            'hybrid_score': hybrid_matches / max(total, 1),
            'components': components,
            'prompt': prompt
        }
    
    def _extract_components(self, prompt: str) -> List[Dict]:
        """
        Extract distinct components from prompt for separate generation.
        E.g., "a car with wheels" -> [{'type': 'body', 'desc': 'car body'}, {'type': 'wheels', 'desc': 'wheels'}]
        """
        components = []
        prompt_lower = prompt.lower()
        
        # Common component patterns
        patterns = [
            (r'with\s+(\w+(?:\s+\w+)?)', 'attachment'),
            (r'and\s+(\w+(?:\s+\w+)?)', 'part'),
            (r'(\w+)\s+on\s+top', 'top_part'),
            (r'(\w+)\s+at\s+(?:the\s+)?bottom', 'bottom_part'),
            (r'(\w+)\s+handle', 'handle'),
            (r'(\w+)\s+legs?', 'legs'),
            (r'(\w+)\s+wheels?', 'wheels'),
            (r'(\w+)\s+base', 'base'),
        ]
        
        for pattern, comp_type in patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                components.append({
                    'type': comp_type,
                    'description': match,
                    'method': 'openscad' if any(kw in match for kw in MECHANICAL_KEYWORDS) else 'point_e'
                })
        
        # Main object (everything before 'with', 'and', etc.)
        main_match = re.match(r'^(?:a\s+)?(.+?)(?:\s+with|\s+and|\s+on|\s+having|$)', prompt_lower)
        if main_match:
            main_obj = main_match.group(1).strip()
            if main_obj:
                method = 'openscad' if any(kw in main_obj for kw in MECHANICAL_KEYWORDS) else 'point_e'
                components.insert(0, {
                    'type': 'main',
                    'description': main_obj,
                    'method': method
                })
        
        return components if components else [{'type': 'main', 'description': prompt, 'method': 'hybrid'}]
    
    async def generate_point_cloud(self, prompt: str, num_points: int = 8192, 
                                   strategy: str = 'auto') -> Tuple[List[Dict], Dict]:
        """
        Generate point cloud using multi-model approach.
        
        Args:
            prompt: Text description
            num_points: Target number of points
            strategy: 'auto', 'point_e', 'openscad', 'hybrid', 'ensemble'
            
        Returns:
            Tuple of (points list, metadata dict)
        """
        # Analyze prompt if auto
        if strategy == 'auto':
            analysis = self.analyze_prompt(prompt)
            strategy = analysis['strategy']
        else:
            analysis = self.analyze_prompt(prompt)
        
        logger.info(f"Multi-model generation: strategy={strategy}, prompt='{prompt[:50]}...'")
        
        metadata = {
            'strategy': strategy,
            'analysis': analysis,
            'methods_used': [],
            'point_counts': {}
        }
        
        try:
            if strategy == 'openscad_primary':
                points = await self._generate_openscad_primary(prompt, num_points, metadata)
            elif strategy == 'point_e_primary':
                points = await self._generate_point_e_primary(prompt, num_points, metadata)
            elif strategy == 'hybrid':
                points = await self._generate_hybrid(prompt, num_points, analysis, metadata)
            elif strategy == 'ensemble':
                points = await self._generate_ensemble(prompt, num_points, metadata)
            else:
                # Default to hybrid
                points = await self._generate_hybrid(prompt, num_points, analysis, metadata)
            
            # Post-process combined point cloud
            points = self._post_process_multimodel(points, num_points)
            
            metadata['total_points'] = len(points)
            metadata['message'] = f"Generated {len(points)} points using {', '.join(metadata['methods_used'])}"
            
            return points, metadata
            
        except Exception as e:
            logger.error(f"Multi-model generation failed: {e}")
            # Fallback to synthetic
            points = self._generate_synthetic_fallback(prompt, num_points)
            metadata['methods_used'] = ['synthetic_fallback']
            metadata['error'] = str(e)
            return points, metadata
    
    async def generate_for_printing(self, prompt: str, num_points: int = 16384,
                                   enhance_density: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Generate point cloud optimized for 3D printing.
        Uses higher density and ensures watertight geometry.
        
        Args:
            prompt: Text description
            num_points: Target number of points (higher for printing)
            enhance_density: Whether to post-process for density
            
        Returns:
            Tuple of (points list, metadata dict)
        """
        # Use ensemble strategy for best quality
        points, metadata = await self.generate_point_cloud(prompt, num_points, strategy='ensemble')
        
        if enhance_density and len(points) > 100:
            # Add density enhancement for printing
            points = self._enhance_for_printing(points, num_points)
            metadata['print_enhanced'] = True
        
        metadata['print_ready'] = True
        return points, metadata
    
    def _enhance_for_printing(self, points: List[Dict], target_count: int) -> List[Dict]:
        """
        Enhance point cloud for better mesh reconstruction.
        Adds points to fill gaps and improve surface density.
        """
        if len(points) >= target_count:
            return points
        
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        colors = np.array([[p.get('r', 0.5), p.get('g', 0.5), p.get('b', 0.5)] for p in points])
        
        enhanced = list(zip(coords.tolist(), colors.tolist()))
        
        while len(enhanced) < target_count:
            # Random pair interpolation
            idx1, idx2 = np.random.choice(len(coords), 2, replace=False)
            t = np.random.uniform(0.3, 0.7)
            
            new_coord = coords[idx1] * (1 - t) + coords[idx2] * t
            new_color = colors[idx1] * (1 - t) + colors[idx2] * t
            
            # Add small jitter for surface variation
            jitter = np.random.normal(0, 0.005, 3)
            new_coord = new_coord + jitter
            
            enhanced.append((new_coord.tolist(), new_color.tolist()))
        
        result = []
        for coord, color in enhanced:
            result.append({
                'x': float(coord[0]),
                'y': float(coord[1]),
                'z': float(coord[2]),
                'r': float(np.clip(color[0], 0, 1)),
                'g': float(np.clip(color[1], 0, 1)),
                'b': float(np.clip(color[2], 0, 1))
            })
        
        logger.info(f"Enhanced point cloud: {len(points)} -> {len(result)} points for printing")
        return result

    
    async def _generate_openscad_primary(self, prompt: str, num_points: int, 
                                         metadata: Dict) -> List[Dict]:
        """Generate primarily using OpenSCAD with Point-E enhancement."""
        self._lazy_init_openscad()
        
        points = []
        
        if self.openscad_service:
            try:
                scad_points, scad_code = self.openscad_service.generate_point_cloud(prompt, num_points)
                if scad_points and len(scad_points) > 100:
                    points = self._convert_to_standard_format(scad_points)
                    metadata['methods_used'].append('openscad')
                    metadata['point_counts']['openscad'] = len(points)
                    metadata['openscad_code'] = scad_code[:1000] if scad_code else None
            except Exception as e:
                logger.warning(f"OpenSCAD generation failed: {e}")
        
        # If OpenSCAD failed or returned few points, enhance with Point-E
        if len(points) < num_points * 0.5:
            self._lazy_init_point_e()
            if self.point_e_service:
                try:
                    pe_points = self.point_e_service.generate_point_cloud(prompt)
                    if pe_points:
                        pe_converted = self._convert_to_standard_format(pe_points)
                        points = self._merge_point_clouds(points, pe_converted, num_points)
                        metadata['methods_used'].append('point_e_enhancement')
                        metadata['point_counts']['point_e'] = len(pe_converted)
                except Exception as e:
                    logger.warning(f"Point-E enhancement failed: {e}")
        
        return points
    
    async def _generate_point_e_primary(self, prompt: str, num_points: int,
                                        metadata: Dict) -> List[Dict]:
        """Generate primarily using Point-E with OpenSCAD for precise parts."""
        self._lazy_init_point_e()
        
        points = []
        
        if self.point_e_service:
            try:
                pe_points = self.point_e_service.generate_point_cloud(prompt)
                if pe_points:
                    points = self._convert_to_standard_format(pe_points)
                    metadata['methods_used'].append('point_e')
                    metadata['point_counts']['point_e'] = len(points)
            except Exception as e:
                logger.warning(f"Point-E generation failed: {e}")
        
        # Enhance density if needed
        if len(points) < num_points * 0.5:
            points = self._upsample_points(points, num_points)
            metadata['methods_used'].append('upsampled')
        
        return points
    
    async def _generate_hybrid(self, prompt: str, num_points: int, 
                               analysis: Dict, metadata: Dict) -> List[Dict]:
        """
        Generate using hybrid approach - different methods for different components.
        """
        self._lazy_init_point_e()
        self._lazy_init_openscad()
        
        all_points = []
        components = analysis.get('components', [])
        
        # Calculate points per component
        points_per_component = num_points // max(len(components), 1)
        
        for comp in components:
            comp_points = []
            comp_desc = comp.get('description', prompt)
            comp_method = comp.get('method', 'hybrid')
            comp_type = comp.get('type', 'main')
            
            # Generate based on recommended method
            if comp_method == 'openscad' and self.openscad_service:
                try:
                    scad_points, _ = self.openscad_service.generate_point_cloud(
                        comp_desc, points_per_component
                    )
                    if scad_points:
                        comp_points = self._convert_to_standard_format(scad_points)
                        metadata['methods_used'].append(f'openscad_{comp_type}')
                except Exception as e:
                    logger.warning(f"OpenSCAD failed for {comp_type}: {e}")
            
            elif comp_method == 'point_e' and self.point_e_service:
                try:
                    pe_points = self.point_e_service.generate_point_cloud(comp_desc)
                    if pe_points:
                        comp_points = self._convert_to_standard_format(pe_points)
                        # Take subset
                        if len(comp_points) > points_per_component:
                            indices = np.random.choice(len(comp_points), points_per_component, replace=False)
                            comp_points = [comp_points[i] for i in indices]
                        metadata['methods_used'].append(f'point_e_{comp_type}')
                except Exception as e:
                    logger.warning(f"Point-E failed for {comp_type}: {e}")
            
            # Fallback to both methods if single method failed
            if not comp_points:
                if self.point_e_service:
                    try:
                        pe_points = self.point_e_service.generate_point_cloud(prompt)
                        if pe_points:
                            comp_points = self._convert_to_standard_format(pe_points)
                            metadata['methods_used'].append('point_e_fallback')
                    except:
                        pass
            
            # Apply transform based on component type
            if comp_points:
                comp_points = self._transform_component(comp_points, comp_type, len(all_points))
                all_points.extend(comp_points)
        
        # If still no points, use full prompt with both methods
        if not all_points:
            all_points = await self._generate_ensemble(prompt, num_points, metadata)
        
        return all_points
    
    async def _generate_ensemble(self, prompt: str, num_points: int, 
                                 metadata: Dict) -> List[Dict]:
        """
        Generate with both methods and ensemble the results for best quality.
        """
        self._lazy_init_point_e()
        self._lazy_init_openscad()
        
        point_e_points = []
        openscad_points = []
        
        # Generate with Point-E
        if self.point_e_service:
            try:
                pe_result = self.point_e_service.generate_point_cloud(prompt)
                if pe_result:
                    point_e_points = self._convert_to_standard_format(pe_result)
                    metadata['methods_used'].append('point_e')
                    metadata['point_counts']['point_e'] = len(point_e_points)
            except Exception as e:
                logger.warning(f"Point-E failed in ensemble: {e}")
        
        # Generate with OpenSCAD
        if self.openscad_service:
            try:
                scad_result, scad_code = self.openscad_service.generate_point_cloud(prompt, num_points)
                if scad_result:
                    openscad_points = self._convert_to_standard_format(scad_result)
                    metadata['methods_used'].append('openscad')
                    metadata['point_counts']['openscad'] = len(openscad_points)
                    metadata['openscad_code'] = scad_code[:1000] if scad_code else None
            except Exception as e:
                logger.warning(f"OpenSCAD failed in ensemble: {e}")
        
        # Ensemble merge
        if point_e_points and openscad_points:
            # Weight based on analysis
            analysis = self.analyze_prompt(prompt)
            pe_weight = analysis.get('organic_score', 0.5)
            scad_weight = analysis.get('mechanical_score', 0.5)
            
            # Normalize weights
            total_weight = pe_weight + scad_weight
            pe_ratio = pe_weight / total_weight
            scad_ratio = scad_weight / total_weight
            
            # Sample from each based on weights
            pe_sample_size = int(num_points * pe_ratio)
            scad_sample_size = int(num_points * scad_ratio)
            
            pe_sampled = self._sample_points(point_e_points, pe_sample_size)
            scad_sampled = self._sample_points(openscad_points, scad_sample_size)
            
            # Merge with blending at overlap regions
            merged = self._blend_point_clouds(pe_sampled, scad_sampled)
            metadata['methods_used'].append('ensemble_blend')
            
            return merged
        elif point_e_points:
            return point_e_points
        elif openscad_points:
            return openscad_points
        else:
            # Both failed - synthetic fallback
            return self._generate_synthetic_fallback(prompt, num_points)
    
    def _convert_to_standard_format(self, points: List) -> List[Dict]:
        """Convert various point formats to standard {x,y,z,r,g,b}."""
        converted = []
        for p in points:
            if isinstance(p, dict):
                if 'pos' in p:
                    converted.append({
                        'x': float(p['pos'][0]),
                        'y': float(p['pos'][1]),
                        'z': float(p['pos'][2]),
                        'r': float(p.get('color', [0.5, 0.5, 0.5])[0]),
                        'g': float(p.get('color', [0.5, 0.5, 0.5])[1]),
                        'b': float(p.get('color', [0.5, 0.5, 0.5])[2])
                    })
                elif 'x' in p:
                    converted.append({
                        'x': float(p['x']),
                        'y': float(p['y']),
                        'z': float(p['z']),
                        'r': float(p.get('r', 0.5)),
                        'g': float(p.get('g', 0.5)),
                        'b': float(p.get('b', 0.5))
                    })
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                converted.append({
                    'x': float(p[0]),
                    'y': float(p[1]),
                    'z': float(p[2]),
                    'r': float(p[3]) if len(p) > 3 else 0.5,
                    'g': float(p[4]) if len(p) > 4 else 0.5,
                    'b': float(p[5]) if len(p) > 5 else 0.5
                })
        return converted
    
    def _merge_point_clouds(self, cloud1: List[Dict], cloud2: List[Dict], 
                            target_count: int) -> List[Dict]:
        """Merge two point clouds intelligently."""
        if not cloud1:
            return cloud2[:target_count] if cloud2 else []
        if not cloud2:
            return cloud1[:target_count]
        
        # Calculate how many from each
        ratio = len(cloud1) / (len(cloud1) + len(cloud2))
        count1 = int(target_count * ratio)
        count2 = target_count - count1
        
        # Sample from each
        sampled1 = self._sample_points(cloud1, count1)
        sampled2 = self._sample_points(cloud2, count2)
        
        return sampled1 + sampled2
    
    def _sample_points(self, points: List[Dict], count: int) -> List[Dict]:
        """Sample points uniformly."""
        if len(points) <= count:
            return points
        indices = np.random.choice(len(points), count, replace=False)
        return [points[i] for i in indices]
    
    def _blend_point_clouds(self, cloud1: List[Dict], cloud2: List[Dict]) -> List[Dict]:
        """Blend two point clouds with smooth transition."""
        if not cloud1:
            return cloud2
        if not cloud2:
            return cloud1
        
        # Find overlapping region and blend colors
        blended = []
        
        # Convert to numpy for efficiency
        coords1 = np.array([[p['x'], p['y'], p['z']] for p in cloud1])
        coords2 = np.array([[p['x'], p['y'], p['z']] for p in cloud2])
        
        # Normalize both to same scale
        center1 = np.mean(coords1, axis=0)
        center2 = np.mean(coords2, axis=0)
        scale1 = np.max(np.linalg.norm(coords1 - center1, axis=1))
        scale2 = np.max(np.linalg.norm(coords2 - center2, axis=1))
        
        if scale1 > 0:
            coords1 = (coords1 - center1) / scale1
        if scale2 > 0:
            coords2 = (coords2 - center2) / scale2
        
        # Add all points with adjusted coordinates
        for i, p in enumerate(cloud1):
            blended.append({
                'x': coords1[i, 0],
                'y': coords1[i, 1],
                'z': coords1[i, 2],
                'r': p['r'],
                'g': p['g'],
                'b': p['b']
            })
        
        for i, p in enumerate(cloud2):
            blended.append({
                'x': coords2[i, 0],
                'y': coords2[i, 1],
                'z': coords2[i, 2],
                'r': p['r'],
                'g': p['g'],
                'b': p['b']
            })
        
        return blended
    
    def _transform_component(self, points: List[Dict], comp_type: str, 
                            existing_count: int) -> List[Dict]:
        """Transform component points based on type."""
        if not points:
            return points
        
        # Define offsets for different component types
        offsets = {
            'main': (0, 0, 0),
            'handle': (0.8, 0.3, 0),
            'legs': (0, -0.5, 0),
            'wheels': (0, -0.3, 0),
            'base': (0, -0.4, 0),
            'top_part': (0, 0.5, 0),
            'bottom_part': (0, -0.5, 0),
            'attachment': (0.5, 0, 0),
            'part': (0, 0, 0.5)
        }
        
        offset = offsets.get(comp_type, (0, 0, 0))
        
        # Apply offset
        transformed = []
        for p in points:
            transformed.append({
                'x': p['x'] + offset[0],
                'y': p['y'] + offset[1],
                'z': p['z'] + offset[2],
                'r': p['r'],
                'g': p['g'],
                'b': p['b']
            })
        
        return transformed
    
    def _upsample_points(self, points: List[Dict], target_count: int) -> List[Dict]:
        """Upsample point cloud using interpolation."""
        if len(points) >= target_count or len(points) < 2:
            return points
        
        upsampled = list(points)
        
        while len(upsampled) < target_count:
            # Random pair interpolation
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx1], points[idx2]
            
            # Random interpolation factor
            t = np.random.uniform(0.3, 0.7)
            
            new_point = {
                'x': p1['x'] * (1-t) + p2['x'] * t,
                'y': p1['y'] * (1-t) + p2['y'] * t,
                'z': p1['z'] * (1-t) + p2['z'] * t,
                'r': p1['r'] * (1-t) + p2['r'] * t,
                'g': p1['g'] * (1-t) + p2['g'] * t,
                'b': p1['b'] * (1-t) + p2['b'] * t
            }
            
            # Add small noise
            noise = np.random.normal(0, 0.01, 3)
            new_point['x'] += noise[0]
            new_point['y'] += noise[1]
            new_point['z'] += noise[2]
            
            upsampled.append(new_point)
        
        return upsampled
    
    def _post_process_multimodel(self, points: List[Dict], target_count: int) -> List[Dict]:
        """Post-process combined point cloud for quality."""
        if not points:
            return points
        
        # Convert to numpy
        coords = np.array([[p['x'], p['y'], p['z']] for p in points])
        colors = np.array([[p['r'], p['g'], p['b']] for p in points])
        
        # Center and normalize
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        
        max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
        if max_dist > 0:
            coords_normalized = coords_centered / max_dist
        else:
            coords_normalized = coords_centered
        
        # Remove outliers
        distances = np.linalg.norm(coords_normalized, axis=1)
        threshold = np.percentile(distances, 98)
        valid_mask = distances <= threshold
        
        coords_filtered = coords_normalized[valid_mask]
        colors_filtered = colors[valid_mask]
        
        # Downsample if needed
        if len(coords_filtered) > target_count:
            indices = np.random.choice(len(coords_filtered), target_count, replace=False)
            coords_filtered = coords_filtered[indices]
            colors_filtered = colors_filtered[indices]
        
        # Reconstruct points
        processed = []
        for i in range(len(coords_filtered)):
            processed.append({
                'x': float(coords_filtered[i, 0]),
                'y': float(coords_filtered[i, 1]),
                'z': float(coords_filtered[i, 2]),
                'r': float(np.clip(colors_filtered[i, 0], 0, 1)),
                'g': float(np.clip(colors_filtered[i, 1], 0, 1)),
                'b': float(np.clip(colors_filtered[i, 2], 0, 1))
            })
        
        return processed
    
    def _generate_synthetic_fallback(self, prompt: str, num_points: int) -> List[Dict]:
        """Generate synthetic point cloud as fallback."""
        import math
        import random
        
        prompt_lower = prompt.lower()
        points = []
        
        # Detect shape from prompt
        if 'cube' in prompt_lower or 'box' in prompt_lower:
            points = self._synth_cube(num_points)
        elif 'sphere' in prompt_lower or 'ball' in prompt_lower:
            points = self._synth_sphere(num_points)
        elif 'cylinder' in prompt_lower or 'tube' in prompt_lower:
            points = self._synth_cylinder(num_points)
        elif 'cone' in prompt_lower:
            points = self._synth_cone(num_points)
        elif 'torus' in prompt_lower or 'donut' in prompt_lower:
            points = self._synth_torus(num_points)
        else:
            # Default composite
            points = self._synth_composite(num_points)
        
        return points
    
    def _synth_cube(self, n: int) -> List[Dict]:
        """Generate cube points."""
        import random
        points = []
        for _ in range(n):
            face = random.randint(0, 5)
            u, v = random.uniform(-1, 1), random.uniform(-1, 1)
            if face == 0: x, y, z = u, v, 1
            elif face == 1: x, y, z = u, v, -1
            elif face == 2: x, y, z = u, 1, v
            elif face == 3: x, y, z = u, -1, v
            elif face == 4: x, y, z = 1, u, v
            else: x, y, z = -1, u, v
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.6, 'g': 0.7, 'b': 0.9})
        return points
    
    def _synth_sphere(self, n: int) -> List[Dict]:
        """Generate sphere points."""
        import random
        import math
        points = []
        for _ in range(n):
            theta = random.uniform(0, 2 * math.pi)
            phi = math.acos(random.uniform(-1, 1))
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.9, 'g': 0.5, 'b': 0.5})
        return points
    
    def _synth_cylinder(self, n: int) -> List[Dict]:
        """Generate cylinder points."""
        import random
        import math
        points = []
        for _ in range(n):
            theta = random.uniform(0, 2 * math.pi)
            y = random.uniform(-1, 1)
            x = math.cos(theta)
            z = math.sin(theta)
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.5, 'g': 0.9, 'b': 0.5})
        return points
    
    def _synth_cone(self, n: int) -> List[Dict]:
        """Generate cone points."""
        import random
        import math
        points = []
        for _ in range(n):
            h = random.uniform(0, 1)
            r = 1 - h
            theta = random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            y = h - 0.5
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.9, 'g': 0.7, 'b': 0.3})
        return points
    
    def _synth_torus(self, n: int) -> List[Dict]:
        """Generate torus points."""
        import random
        import math
        R, r = 0.7, 0.3
        points = []
        for _ in range(n):
            u = random.uniform(0, 2 * math.pi)
            v = random.uniform(0, 2 * math.pi)
            x = (R + r * math.cos(v)) * math.cos(u)
            y = r * math.sin(v)
            z = (R + r * math.cos(v)) * math.sin(u)
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.8, 'g': 0.4, 'b': 0.8})
        return points
    
    def _synth_composite(self, n: int) -> List[Dict]:
        """Generate composite shape."""
        import random
        import math
        points = []
        # Cylinder base
        for _ in range(n // 2):
            theta = random.uniform(0, 2 * math.pi)
            y = random.uniform(-0.5, 0.5)
            x = 0.8 * math.cos(theta)
            z = 0.8 * math.sin(theta)
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.5, 'g': 0.7, 'b': 0.9})
        # Hemisphere top
        for _ in range(n // 2):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi / 2)
            x = 0.8 * math.sin(phi) * math.cos(theta)
            y = 0.5 + 0.8 * math.cos(phi)
            z = 0.8 * math.sin(phi) * math.sin(theta)
            points.append({'x': x, 'y': y, 'z': z, 'r': 0.5, 'g': 0.7, 'b': 0.9})
        return points


# Singleton instance
_multimodel_service = None

def get_multimodel_service() -> MultiModelService:
    """Get or create the multi-model service singleton."""
    global _multimodel_service
    if _multimodel_service is None:
        _multimodel_service = MultiModelService()
    return _multimodel_service
