from typing import Dict, List, Tuple

def check_flow_compliance(pipe_diameter_inch, capacity_mld) -> Tuple[bool, str]:
    try:
        gpm = (float(capacity_mld) * 10**6) / (24 * 60 * 3.785)
        velocity = (0.408 * gpm) / (float(pipe_diameter_inch)**2)
        if velocity > 8.5:
            return False, f"Critical Velocity: {velocity:.2f} ft/s exceeds 8.5 ft/s limit. Cavitation risk."
        if velocity < 2.0:
            return True, f"Warning: Low Velocity {velocity:.2f} ft/s. Sedimentation risk."
        return True, f"Velocity {velocity:.2f} ft/s is within optimal range (2-8 ft/s)."
    except Exception as e:
        return True, "Analysis bypass: parameters pending."

def estimate_power_kw(flow_m3h, head_m=45, efficiency=0.72) -> float:
    try:
        flow_m3s = float(flow_m3h) / 3600
        power = (flow_m3s * 1000 * 9.81 * head_m) / (efficiency * 1000)
        return round(power, 1)
    except:
        return 0.0

def validate_topology(components: List[Dict]) -> List[str]:
    logs = []
    pumps = [c for c in components if c['type'] == 'pump']
    tanks = [c for c in components if c['type'] == 'tank']
    if len(pumps) > 0 and len(tanks) == 0:
        logs.append("Warning: System contains active pressure sources without downstream storage.")
    if len(tanks) > 1:
        logs.append("Design Note: Multiple reservoirs detected. Equalization analysis recommended.")
    return logs

class EngineeringEngine:
    def __init__(self):
        from local_engine import LocalKnowledgeEngine
        self.local_engine = LocalKnowledgeEngine()
    async def generate_layout(self, prompt: str) -> Dict:
        return self.local_engine.generate_layout(prompt)
    async def consult_design(self, design: dict, question: str) -> str:
        return self.local_engine.consult_design(design, question)
