#!/usr/bin/env python3
"""
Point-E Integration Examples
Demonstrates various ways to use the Point-E service
"""

import asyncio
import aiohttp
import json

API_URL = "http://localhost:8000"

async def example_1_basic_generation():
    """Example 1: Basic point cloud generation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Point Cloud Generation")
    print("="*60)
    
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": "a red industrial pump",
            "quality": "fast"
        }
        
        print(f"\nGenerating point cloud for: '{payload['prompt']}'")
        print(f"Quality: {payload['quality']}")
        
        async with session.post(f"{API_URL}/generate-points", json=payload) as resp:
            result = await resp.json()
            print(f"\n✓ Generated {result['count']} points")
            if result['points']:
                first_point = result['points'][0]
                print(f"  Sample point:")
                print(f"    Position: {first_point['pos']}")
                print(f"    Color: {first_point['color']}")

async def example_2_quality_comparison():
    """Example 2: Compare different quality levels"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Quality Levels Comparison")
    print("="*60)
    
    qualities = ["fast", "normal", "high"]
    prompt = "a blue cube"
    
    async with aiohttp.ClientSession() as session:
        for quality in qualities:
            payload = {"prompt": prompt, "quality": quality}
            print(f"\nGenerating with quality: {quality}")
            
            try:
                async with session.post(f"{API_URL}/generate-points", 
                                       json=payload, 
                                       timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"  ✓ {result['count']} points generated")
                    else:
                        print(f"  ✗ Error: {resp.status}")
            except asyncio.TimeoutError:
                print(f"  ⏱ Timeout (still generating...)")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")

async def example_3_export_point_cloud():
    """Example 3: Export point cloud to different formats"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Export Point Cloud")
    print("="*60)
    
    # First, generate a point cloud
    async with aiohttp.ClientSession() as session:
        print("\n1. Generating point cloud...")
        gen_payload = {
            "prompt": "a yellow sphere",
            "quality": "fast"
        }
        
        async with session.post(f"{API_URL}/generate-points", 
                               json=gen_payload,
                               timeout=aiohttp.ClientTimeout(total=120)) as resp:
            points = (await resp.json())['points']
        
        # Export to different formats
        formats = ["ply", "obj", "json"]
        for fmt in formats:
            print(f"\n2. Exporting to {fmt.upper()}...")
            export_payload = {
                "points": points,
                "format": fmt,
                "filename": f"sample_{fmt}"
            }
            
            async with session.post(f"{API_URL}/export-point-cloud", 
                                   json=export_payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"  ✓ Exported to: {result['url']}")
                else:
                    print(f"  ✗ Failed: {resp.status}")

async def example_4_batch_generation():
    """Example 4: Generate multiple point clouds"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Generation")
    print("="*60)
    
    prompts = [
        "a red pump",
        "a blue tank",
        "a green valve",
        "a yellow pipe",
        "an orange wrench"
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}/{len(prompts)}: {prompt}")
            
            payload = {"prompt": prompt, "quality": "fast"}
            
            try:
                async with session.post(f"{API_URL}/generate-points", 
                                       json=payload,
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    result = await resp.json()
                    print(f"  ✓ {result['count']} points")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")

async def example_5_color_extraction():
    """Example 5: Analyze color distribution in point cloud"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Color Analysis")
    print("="*60)
    
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": "a red object", "quality": "fast"}
        print(f"\nGenerating point cloud for color analysis...")
        
        async with session.post(f"{API_URL}/generate-points", 
                               json=payload,
                               timeout=aiohttp.ClientTimeout(total=120)) as resp:
            result = await resp.json()
            points = result['points']
        
        # Analyze colors
        import statistics
        reds = [p['color'][0] for p in points]
        greens = [p['color'][1] for p in points]
        blues = [p['color'][2] for p in points]
        
        print(f"\nColor Statistics ({len(points)} points):")
        print(f"  Red   - Mean: {statistics.mean(reds):.1f}, Median: {statistics.median(reds):.1f}")
        print(f"  Green - Mean: {statistics.mean(greens):.1f}, Median: {statistics.median(greens):.1f}")
        print(f"  Blue  - Mean: {statistics.mean(blues):.1f}, Median: {statistics.median(blues):.1f}")

async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("POINT-E INTEGRATION EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate the Point-E integration capabilities.")
    print("Ensure the backend server is running on http://localhost:8000")
    
    try:
        # Check if server is running
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{API_URL}/docs", timeout=aiohttp.ClientTimeout(total=2)):
                    pass
            except:
                print("\n✗ Error: Backend server not found at {API_URL}")
                print("Please start the backend with: python main.py")
                return
        
        # Run examples
        await example_1_basic_generation()
        
        # Note: Other examples commented out to avoid long waits
        # await example_2_quality_comparison()
        # await example_3_export_point_cloud()
        # await example_4_batch_generation()
        # await example_5_color_extraction()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
