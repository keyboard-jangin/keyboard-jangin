#!/usr/bin/env python3
"""
Real-Time Scan Coverage Feedback Demo

Demonstrates the AR-guided scan feedback system that provides
live visual guidance during fracture capture.

FEATURES DEMONSTRATED:
- Voxel-based coverage tracking
- Quality heatmap generation
- AR overlay with progress indicators
- Rescan area detection
- Coverage completeness assessment
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glass_fracture_forensics import (
    ScanCoverageTracker,
    VoxelGrid,
    ScanQuality,
    ARFeedbackOverlay,
)


def simulate_camera_trajectory(n_frames: int = 30) -> list:
    """
    Simulate camera moving around a scan target

    Returns list of camera poses (4x4 matrices)
    """
    poses = []
    radius = 0.5  # 50cm from center

    for i in range(n_frames):
        # Circular trajectory
        angle = 2 * np.pi * i / n_frames
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.3  # 30cm above target

        # Camera position
        camera_pos = np.array([x, y, z])

        # Look at center
        look_at = np.array([0, 0, 0.1])
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)

        # Up vector
        up = np.array([0, 0, 1])

        # Right vector (cross product)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Recompute up (for orthogonality)
        up = np.cross(right, forward)

        # Build rotation matrix [right, up, forward]
        R = np.column_stack([right, up, forward])

        # Build 4x4 pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = camera_pos

        poses.append(pose)

    return poses


def generate_fracture_points(n_points: int = 100) -> np.ndarray:
    """
    Generate simulated fracture points

    Returns (N, 3) array of 3D points
    """
    points = []

    # Create radial fracture pattern (3 lines)
    for angle in [0, np.pi * 2/3, np.pi * 4/3]:
        # Line from center outward
        t = np.linspace(0, 0.3, n_points // 3)
        x = t * np.cos(angle)
        y = t * np.sin(angle)
        z = np.ones_like(t) * 0.1

        # Add some noise
        x += np.random.randn(len(t)) * 0.01
        y += np.random.randn(len(t)) * 0.01

        line_points = np.column_stack([x, y, z])
        points.append(line_points)

    return np.vstack(points)


def main():
    """Run real-time feedback demo"""

    print("="*70)
    print("REAL-TIME SCAN COVERAGE FEEDBACK DEMO")
    print("="*70)

    # Define scan volume
    scan_bounds = (
        np.array([-0.4, -0.4, 0.0]),  # min: 40cm x 40cm x 30cm
        np.array([0.4, 0.4, 0.3])     # max
    )

    print("\nScan Volume:")
    print(f"  Bounds: {scan_bounds[0]} to {scan_bounds[1]}")

    # Create voxel grid (2cm resolution)
    print("\nInitializing voxel grid...")
    voxel_grid = VoxelGrid(
        bounds_min=scan_bounds[0],
        bounds_max=scan_bounds[1],
        resolution=0.02  # 2cm voxels
    )

    print(f"  Grid size: {voxel_grid.grid_size}")
    print(f"  Total voxels: {np.prod(voxel_grid.grid_size)}")

    # Camera intrinsics
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)

    # Create tracker
    print("\nInitializing scan tracker...")
    tracker = ScanCoverageTracker(
        voxel_grid=voxel_grid,
        camera_matrix=camera_matrix,
        min_observations=3,
        min_view_diversity=30.0
    )

    # Generate fracture points (static)
    fracture_points = generate_fracture_points(150)
    print(f"  Generated {len(fracture_points)} fracture points")

    # Simulate camera trajectory
    print("\nSimulating AR capture...")
    camera_poses = simulate_camera_trajectory(n_frames=30)
    print(f"  Camera trajectory: {len(camera_poses)} frames")

    # Create AR overlay renderer
    image_size = (640, 480)
    ar_overlay = ARFeedbackOverlay(image_size)

    # Process each frame
    for frame_idx, pose in enumerate(camera_poses):
        # Simulate observing subset of points visible from this view
        # (In real system, this would be actual detected features)
        camera_pos = pose[:3, 3]
        camera_forward = pose[:3, 2]

        # Simple visibility check
        visible_points = []
        for point in fracture_points:
            to_point = point - camera_pos
            to_point_norm = to_point / (np.linalg.norm(to_point) + 1e-8)

            # Check if point is in front of camera
            if np.dot(to_point_norm, camera_forward) > 0.5:
                visible_points.append(point)

        if len(visible_points) > 0:
            visible_points = np.array(visible_points)

            # Update tracker
            tracker.update_from_points(visible_points, pose)

        # Every 5 frames, show progress
        if (frame_idx + 1) % 5 == 0:
            tracker.compute_coverage_quality()
            is_complete, stats = tracker.is_scan_complete()

            print(f"\n  Frame {frame_idx + 1}/{len(camera_poses)}:")
            print(f"    Coverage: {stats['coverage_ratio']:.1%}")
            print(f"    Good quality: {stats['good_quality_ratio']:.1%}")
            print(f"    Status: {'COMPLETE' if is_complete else 'IN PROGRESS'}")

    # Final quality computation
    print("\n" + "="*70)
    print("FINAL SCAN QUALITY ASSESSMENT")
    print("="*70)

    tracker.compute_coverage_quality()
    is_complete, stats = tracker.is_scan_complete()

    print(f"\nCoverage Statistics:")
    print(f"  Total voxels: {stats['total_voxels']}")
    print(f"  Observed voxels: {stats['observed_voxels']}")
    print(f"  Coverage ratio: {stats['coverage_ratio']:.1%}")
    print(f"  Good quality ratio: {stats['good_quality_ratio']:.1%}")
    print(f"  Needs rescan: {stats['needs_rescan_ratio']:.1%}")

    print(f"\nQuality Distribution:")
    dist = stats['quality_distribution']
    total = stats['total_voxels']
    print(f"  Unscanned:  {dist['unscanned']:5d} ({dist['unscanned']/total*100:5.1f}%)")
    print(f"  Poor:       {dist['poor']:5d} ({dist['poor']/total*100:5.1f}%)")
    print(f"  Partial:    {dist['partial']:5d} ({dist['partial']/total*100:5.1f}%)")
    print(f"  Good:       {dist['good']:5d} ({dist['good']/total*100:5.1f}%)")
    print(f"  Excellent:  {dist['excellent']:5d} ({dist['excellent']/total*100:5.1f}%)")

    print(f"\nScan Status: {'✓ COMPLETE' if is_complete else '⚠ INCOMPLETE'}")

    # Get rescan regions
    rescan_regions = tracker.get_rescan_regions()
    if rescan_regions:
        print(f"\nRescan Required:")
        print(f"  {len(rescan_regions)} regions need additional coverage")
        print(f"  Rescan positions (first 5):")
        for i, pos in enumerate(rescan_regions[:5]):
            print(f"    {i+1}. {pos}")
    else:
        print(f"\n✓ No rescan required - all regions adequately covered")

    # Generate visualization for final camera pose
    print("\n" + "="*70)
    print("GENERATING AR OVERLAY VISUALIZATION")
    print("="*70)

    final_pose = camera_poses[-1]

    # Generate heatmap
    print("\nGenerating quality heatmap...")
    heatmap = tracker.generate_heatmap_2d(final_pose, image_size)

    # Create AR overlay
    print("Creating AR overlay...")
    overlay = ar_overlay.create_overlay(heatmap, stats)

    # Save visualization
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = output_dir / "scan_coverage_heatmap.png"
    overlay_path = output_dir / "scan_ar_overlay.png"

    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(overlay_path), overlay)

    print(f"\n✓ Heatmap saved: {heatmap_path}")
    print(f"✓ AR overlay saved: {overlay_path}")

    # Usage recommendations
    print("\n" + "="*70)
    print("INTEGRATION GUIDE")
    print("="*70)

    print("""
During AR capture, the system provides:

1. REAL-TIME HEATMAP (Color-coded quality):
   - Red:    Unscanned or poor quality → SCAN HERE
   - Yellow:  Partial coverage → NEEDS MORE VIEWS
   - Green:   Good coverage → WELL SCANNED
   - Blue:    Excellent coverage → OPTIMAL

2. PROGRESS INDICATORS:
   - Coverage bar: Overall scan progress
   - Quality score: Percentage of high-quality regions

3. RESCAN HINTS:
   - Red circles mark areas needing more coverage
   - Move camera to highlighted regions

4. COMPLETION CHECK:
   - System alerts when scan meets quality thresholds
   - Safe to proceed with reconstruction

USAGE IN PRODUCTION:
- Integrate tracker.update_from_points() in capture loop
- Call tracker.compute_coverage_quality() every N frames
- Display ar_overlay.create_overlay() on screen
- Check tracker.is_scan_complete() before finishing
""")

    print("="*70)
    print("Demo complete. Check output/ for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
