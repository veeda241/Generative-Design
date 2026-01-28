import os
import sys
import torch
import logging

# Ensure absolute paths and no locks
os.environ['POINT_E_CACHE_DIR'] = r'c:\hackathon\Gemini_CLI\Generative-Design\point_e_cache_fixed'
os.environ['POINT_E_DEVICE'] = 'cuda'

# Redirect logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='point_e_diagnostic.log',
    filemode='w'
)
logger = logging.getLogger('diagnostic')

try:
    sys.path.append(os.path.join(os.getcwd(), 'backend'))
    from point_e_service import PointEService
    
    logger.info("Starting PointEService diagnostic...")
    service = PointEService(quality='fast') # Use fast to minimize load
    logger.info(f"Service initialized on {service.device}")
    
    logger.info("Generating test point cloud...")
    points = service.generate_point_cloud("a small cube")
    logger.info(f"Success! Generated {len(points)} points.")
    
except Exception as e:
    logger.error(f"Diagnostic failed: {str(e)}", exc_info=True)
    print(f"DIAGNOSTIC_ERROR: {str(e)}")
