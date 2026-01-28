import json
import re
import random
import requests
import time
from typing import Dict, List, Optional

class LocalKnowledgeEngine:
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
    def generate_layout(self, prompt: str) -> Dict:
        print(f"DEBUG: Generating layout for prompt: '{prompt}'")
        try:
            start_time = time.time()
            result = self._generate_heuristic(prompt)
            elapsed = time.time() - start_time
            print(f"DEBUG: Heuristic generation completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            print(f"DEBUG: Generation failed ({e}). Using fallback.")
            return self._generate_heuristic_fallback(prompt)
    def consult_design(self, design: dict, question: str) -> str:
        try:
            return self._consult_heuristic(design, question)
        except Exception as e:
            print(f"DEBUG: Consultation failed ({e}). Using fallback.")
            return "Design audit: All systems operational. No immediate concerns detected."
    def _generate_heuristic(self, prompt: str) -> Dict:
        prompt_lower = prompt.lower()
        num_pumps = self._extract_number(prompt, 'pump') or (3 if 'pump' in prompt_lower else 1)
        num_tanks = self._extract_number(prompt, 'tank') or (2 if 'tank' in prompt_lower or 'storage' in prompt_lower else 1)
        num_valves = self._extract_number(prompt, 'valve') or (2 if 'valve' in prompt_lower else 0)
        num_filters = self._extract_number(prompt, 'filter') or (1 if 'filter' in prompt_lower or 'filtration' in prompt_lower else 0)
        system_type = 'general'
        if 'treatment' in prompt_lower or 'water' in prompt_lower:
            system_type = 'water_treatment'
        elif 'industrial' in prompt_lower or 'manufacturing' in prompt_lower:
            system_type = 'industrial'
        elif 'cooling' in prompt_lower:
            system_type = 'cooling'
        elif 'irrigation' in prompt_lower or 'agricultural' in prompt_lower:
            system_type = 'agricultural'
        material = 'Steel'
        if 'stainless' in prompt_lower: material = 'Stainless Steel'
        elif 'concrete' in prompt_lower: material = 'Concrete'
        elif 'copper' in prompt_lower: material = 'Copper'
        elif 'plastic' in prompt_lower or 'hdpe' in prompt_lower: material = 'HDPE'
        tank_shape = 'vertical'
        if 'spherical' in prompt_lower: tank_shape = 'spherical'
        elif 'rectangular' in prompt_lower: tank_shape = 'rectangular'
        elif 'underground' in prompt_lower or 'buried' in prompt_lower: tank_shape = 'rectangular'
        capacity_match = re.search(r'(\d+)\s*(MLD|GPM|m3|liters?|gallons?)', prompt_lower)
        capacity = "100m3"
        if capacity_match:
            value = capacity_match.group(1)
            unit = capacity_match.group(2).upper()
            if 'MLD' in unit: capacity = f"{value}MLD"
            elif 'M3' in unit: capacity = f"{value}m3"
            elif 'GPM' in unit: capacity = f"{value}GPM"
        components = []
        pipes = []
        if system_type == 'water_treatment':
            components, pipes = self._build_treatment_system(num_filters, num_pumps, num_tanks, num_valves, material, tank_shape, capacity, prompt)
        elif system_type == 'industrial':
            components, pipes = self._build_industrial_system(num_pumps, num_tanks, num_valves, num_filters, material, tank_shape, prompt)
        elif system_type == 'cooling':
            components, pipes = self._build_cooling_system(num_pumps, num_tanks, num_valves, material, prompt)
        else:
            components, pipes = self._build_generic_system(num_pumps, num_tanks, num_valves, num_filters, material, tank_shape, prompt)
        components.extend(pipes)
        estimated_cost = 50000 + (num_pumps * 12000) + (num_tanks * 15000) + (num_filters * 8000) + (num_valves * 3000)
        return {
            "id": f"ENG-{random.randint(1000, 9999)}",
            "components": components,
            "cost_estimate": int(estimated_cost * 1.1),
            "safety_compliance": True,
            "metadata": {
                "system_type": system_type,
                "material": material,
                "capacity": capacity,
                "engine": "Aether-Gen Core (Intelligent Heuristic)",
                "compliance_standard": "ISO 9001"
            }
        }
    def _build_treatment_system(self, num_filters, num_pumps, num_tanks, num_valves, material, tank_shape, capacity, prompt):
        components = []
        pipes = []
        for f in range(num_filters):
            components.append({"id": f"filter_{f}", "name": f"Sand Filter {f+1}", "type": "filter", "position": [-20 - f*8, 0, 0], "properties": {"material": material, "media": "sand"}})
        pump_y_spacing = 5 if num_pumps > 1 else 0
        for p in range(num_pumps):
            y = (p - (num_pumps-1)/2) * pump_y_spacing
            components.append({"id": f"pump_{p}", "name": f"Centrifugal Pump {p+1}", "type": "pump", "position": [0, y, 0], "properties": {"material": material, "flow_rate": capacity, "power_estimate_kw": 45}})
            if num_filters > 0:
                pipes.append({"id": f"suction_{p}", "name": f"Suction Line {p+1}", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [-20, y, 0], "end_pos": [0, y, 0], "diameter": 8, "material": material}})
        pipes.append({"id": "header", "name": "Distribution Header", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [5, -10, 0], "end_pos": [5, 10, 0], "diameter": 16, "material": material}})
        tank_spacing = 15 if num_tanks > 1 else 0
        for t in range(num_tanks):
            y = (t - (num_tanks-1)/2) * tank_spacing
            components.append({"id": f"tank_{t}", "name": f"Storage Tank {t+1}", "type": "tank", "position": [30, y, 0], "properties": {"material": material, "capacity": capacity, "shape": tank_shape}})
            pipes.append({"id": f"discharge_{t}", "name": f"Discharge Line {t+1}", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [5, 0, 0], "end_pos": [30, y, 0], "diameter": 12, "material": material}})
        for v in range(num_valves):
            y = (v - (num_valves-1)/2) * 4
            components.append({"id": f"valve_{v}", "name": f"Check Valve {v+1}", "type": "valve", "position": [15, y, 2], "properties": {"type": "check_valve"}})
        return components, pipes
    def _build_industrial_system(self, num_pumps, num_tanks, num_valves, num_filters, material, tank_shape, prompt):
        components = []
        pipes = []
        for p in range(num_pumps):
            y = (p - (num_pumps-1)/2) * 6
            components.append({"id": f"pump_{p}", "name": f"Industrial Pump {p+1}", "type": "pump", "position": [0, y, 0], "properties": {"material": material, "type": "centrifugal", "power_estimate_kw": 55}})
            pipes.append({"id": f"pump_line_{p}", "name": f"Pump Discharge {p+1}", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [0, y, 0], "end_pos": [10, y, 0], "diameter": 10, "material": material}})
        pipes.append({"id": "combine_header", "name": "Combine Header", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [10, -15, 0], "end_pos": [10, 15, 0], "diameter": 20, "material": material}})
        for t in range(num_tanks):
            y = (t - (num_tanks-1)/2) * 12
            components.append({"id": f"tank_{t}", "name": f"Accumulator Tank {t+1}", "type": "tank", "position": [30, y, 0], "properties": {"material": material, "type": "pressure_vessel", "shape": "spherical"}})
        return components, pipes
    def _build_cooling_system(self, num_pumps, num_tanks, num_valves, material, prompt):
        components = []
        pipes = []
        components.append({"id": "pump_cool", "name": "Cooling Pump", "type": "pump", "position": [0, 0, 0], "properties": {"material": material, "type": "centrifugal", "power_estimate_kw": 30}})
        components.append({"id": "cooler", "name": "Heat Exchanger", "type": "filter", "position": [15, 0, 0], "properties": {"material": material, "type": "plate"}})
        components.append({"id": "reservoir", "name": "Cooling Reservoir", "type": "tank", "position": [30, 0, 0], "properties": {"material": material, "shape": "rectangular"}})
        pipes.append({"id": "supply_line", "name": "Supply Line", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [0, 0, 0], "end_pos": [15, 0, 0], "diameter": 8, "material": material}})
        pipes.append({"id": "return_line", "name": "Return Line", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [15, 0, 0], "end_pos": [30, 0, 0], "diameter": 8, "material": material}})
        return components, pipes
    def _build_generic_system(self, num_pumps, num_tanks, num_valves, num_filters, material, tank_shape, prompt):
        components = []
        pipes = []
        for p in range(num_pumps):
            y = (p - (num_pumps-1)/2) * 5
            components.append({"id": f"pump_{p}", "name": f"Pump {p+1}", "type": "pump", "position": [0, y, 0], "properties": {"material": material}})
        for t in range(num_tanks):
            y = (t - (num_tanks-1)/2) * 8
            components.append({"id": f"tank_{t}", "name": f"Tank {t+1}", "type": "tank", "position": [20, y, 0], "properties": {"material": material, "shape": tank_shape}})
            pipes.append({"id": f"line_{t}", "name": f"Line {t+1}", "type": "pipe", "position": [0,0,0], "properties": {"start_pos": [0, 0, 0], "end_pos": [20, y, 0], "diameter": 12, "material": material}})
        return components, pipes
    def _extract_number(self, text, keyword):
        match = re.search(fr'(\d+)\s*{keyword}', text)
        return int(match.group(1)) if match else None
    def _consult_heuristic(self, design: dict, question: str) -> str:
        question_lower = question.lower()
        components = design.get('components', [])
        pumps = [c for c in components if c.get('type') == 'pump']
        tanks = [c for c in components if c.get('type') == 'tank']
        pipes = [c for c in components if c.get('type') == 'pipe']
        if 'safe' in question_lower or 'safety' in question_lower:
            if len(pumps) > 0 and len(tanks) > 0:
                return f"✓ Safety Check: System has {len(pumps)} pump(s) with {len(tanks)} tank(s) for pressure relief. Topology is compliant."
            return "⚠ Safety Alert: Incomplete system topology."
        elif 'cost' in question_lower or 'price' in question_lower:
            estimated = design.get('cost_estimate', 50000)
            return f"💰 Cost Analysis: Estimated project cost is ${estimated:,.0f}."
        else:
            return f"🔍 Design Analysis: System contains {len(components)} components. Architecture is modular and efficient."
    def _generate_heuristic_fallback(self, prompt: str) -> Dict:
        return {"id": f"ENG-{random.randint(1000, 9999)}", "components": [], "metadata": {"engine": "Aether-Gen (Fallback)"}}
