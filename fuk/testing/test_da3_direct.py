#!/usr/bin/env python3
"""
Direct DA3 Test Script - Bypass web server for debugging

Tests:
1. Single image depth estimation
2. Video batch depth estimation  
3. Compares output ranges and quality

Run: python test_da3_direct.py /path/to/video.mp4
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time

# Add vendor path
sys.path.insert(0, "/home/brad/ai/fuk/fuk/vendor/Depth-Anything-3")

def extract_frames(video_path: Path, output_dir: Path, max_frames: int = 30):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        frame_idx += 1
    
    cap.release()
    return frames


def test_single_image(model, image_path: str, process_res: int = 1000):
    """Test single image inference"""
    print(f"\n=== SINGLE IMAGE TEST ===")
    print(f"Image: {image_path}")
    print(f"Process res: {process_res}")
    
    start = time.time()
    prediction = model.inference(
        image=[image_path],
        process_res=process_res,
        process_res_method="lower_bound_resize",
    )
    elapsed = time.time() - start
    
    depth = prediction.depth[0]
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth dtype: {depth.dtype}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Depth mean: {depth.mean():.4f}")
    print(f"Depth std: {depth.std():.4f}")
    
    # Check for problematic ranges
    depth_range = depth.max() - depth.min()
    if depth_range < 0.3:
        print(f"⚠️  WARNING: Very narrow depth range ({depth_range:.4f}) - will cause noise when normalized!")
    
    return depth


def test_batch_video(model, frame_paths: list, process_res: int = 1344):
    """Test batch video inference with different strategies"""
    print(f"\n=== BATCH VIDEO TEST ({len(frame_paths)} frames) ===")
    
    frame_strs = [str(p) for p in frame_paths]
    
    # Test different ref_view_strategies
    strategies = ["middle", "first", "saddle_balanced"]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        try:
            start = time.time()
            prediction = model.inference(
                image=frame_strs,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                ref_view_strategy=strategy,
            )
            elapsed = time.time() - start
            
            all_depths = prediction.depth
            
            print(f"Time: {elapsed:.2f}s")
            print(f"Output shape: {all_depths.shape}")
            print(f"Global range: [{all_depths.min():.4f}, {all_depths.max():.4f}]")
            print(f"Global mean: {all_depths.mean():.4f}")
            print(f"Global std: {all_depths.std():.4f}")
            
            # Per-frame stats
            frame_ranges = []
            for i, depth in enumerate(all_depths):
                frame_range = depth.max() - depth.min()
                frame_ranges.append(frame_range)
            
            print(f"Per-frame range: min={min(frame_ranges):.4f}, max={max(frame_ranges):.4f}, mean={np.mean(frame_ranges):.4f}")
            
            # Check for issues
            global_range = all_depths.max() - all_depths.min()
            if global_range < 0.3:
                print(f"⚠️  WARNING: Narrow global range ({global_range:.4f})")
            
        except Exception as e:
            print(f"❌ Strategy '{strategy}' failed: {e}")


def test_resolution_comparison(model, image_path: str):
    """Test different resolutions"""
    print(f"\n=== RESOLUTION COMPARISON ===")
    
    resolutions = [504, 756, 1008, 1344]
    
    for res in resolutions:
        try:
            prediction = model.inference(
                image=[image_path],
                process_res=res,
                process_res_method="lower_bound_resize",
            )
            depth = prediction.depth[0]
            depth_range = depth.max() - depth.min()
            print(f"  {res}px: range=[{depth.min():.4f}, {depth.max():.4f}] (span={depth_range:.4f}), shape={depth.shape}")
        except Exception as e:
            print(f"  {res}px: FAILED - {e}")


def save_depth_visualization(depth: np.ndarray, output_path: Path, colormap: str = "inferno"):
    """Save depth map as visualization"""
    # Normalize
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap
    colormap_func = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
    colored = cv2.applyColorMap(depth_uint8, colormap_func)
    
    cv2.imwrite(str(output_path), colored)
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_da3_direct.py <video_or_image_path> [model_id]")
        print()
        print("Models:")
        print("  depth-anything/DA3MONO-LARGE  - Monocular (single images)")
        print("  depth-anything/DA3-LARGE-1.1  - Multi-view (video)")
        print("  depth-anything/DA3-GIANT-1.1  - Largest model")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    model_id = sys.argv[2] if len(sys.argv) > 2 else "depth-anything/DA3-LARGE-1.1"
    
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("DA3 Direct Test")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Model: {model_id}")
    
    # Load model
    print(f"\nLoading model...")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device="cuda")
    print("✔ Model loaded")
    
    # Determine if video or image
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    is_video = input_path.suffix.lower() in video_extensions
    
    if is_video:
        # Extract frames to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print(f"\nExtracting frames...")
            frame_paths = extract_frames(input_path, temp_path, max_frames=30)
            print(f"Extracted {len(frame_paths)} frames")
            
            if frame_paths:
                # Test single frame first
                single_depth = test_single_image(model, str(frame_paths[0]))
                
                # Test batch
                test_batch_video(model, frame_paths)
                
                # Test resolutions
                test_resolution_comparison(model, str(frame_paths[0]))
                
                # Save visualizations
                output_dir = input_path.parent / "da3_test_output"
                output_dir.mkdir(exist_ok=True)
                
                save_depth_visualization(single_depth, output_dir / "single_frame_depth.png")
                print(f"\nTest outputs saved to: {output_dir}")
    else:
        # Single image
        single_depth = test_single_image(model, str(input_path))
        test_resolution_comparison(model, str(input_path))
        
        # Save visualization
        output_path = input_path.parent / f"{input_path.stem}_depth.png"
        save_depth_visualization(single_depth, output_path)
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
