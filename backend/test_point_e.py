#!/usr/bin/env python3
"""Quick test of Point-E service."""

from point_e_service import PointEService
import sys

try:
    print("=" * 60)
    print("TESTING POINT-E SERVICE")
    print("=" * 60)
    
    print("\n1. Initializing Point-E service (fast mode)...")
    service = PointEService(quality='fast')
    print("   ✓ Service initialized successfully!")
    
    print("\n2. Testing point cloud generation...")
    print("   Prompt: 'a simple red cube'")
    points = service.generate_point_cloud('a simple red cube')
    
    print(f"\n   ✓ Generated {len(points)} points successfully!")
    
    if points:
        print(f"\n3. Sample data:")
        print(f"   First point position: {points[0]['pos']}")
        print(f"   First point color: {points[0]['color']}")
        print(f"   Last point position: {points[-1]['pos']}")
        print(f"   Last point color: {points[-1]['color']}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ TEST FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
