#!/usr/bin/env python3
"""
Basic Glass Fracture Analysis Example

This example demonstrates how to use the Glass Fracture Forensic System
for a basic analysis workflow.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glass_fracture_forensics import (
    GlassFractureForensicSystem,
    GlassMaterialProperties,
    SystemThresholds,
)


def main():
    """Run basic forensic analysis example"""

    print("="*70)
    print("GLASS FRACTURE FORENSIC SYSTEM - BASIC EXAMPLE")
    print("="*70)

    # Step 1: Configure material properties (default: soda-lime glass)
    print("\n[1/5] Configuring material properties...")
    material = GlassMaterialProperties(
        E=72.0e9,           # Young's Modulus [Pa]
        nu=0.23,            # Poisson's Ratio
        K_Ic=0.75e6,        # Fracture Toughness [Pa·√m]
        rho=2500.0,         # Density [kg/m³]
        stress_state='plane_stress'
    )
    print(f"  Material: Soda-lime glass (ASTM C1036)")
    print(f"  Young's Modulus: {material.E/1e9:.1f} GPa")
    print(f"  Fracture Toughness: {material.K_Ic/1e6:.2f} MPa·√m")

    # Step 2: Configure system thresholds
    print("\n[2/5] Configuring system thresholds...")
    thresholds = SystemThresholds(
        min_parallax_angle=5.0,
        min_valid_tracks=10,
        ransac_confidence=0.999,
    )
    print(f"  Minimum parallax: {thresholds.min_parallax_angle}°")
    print(f"  RANSAC confidence: {thresholds.ransac_confidence}")

    # Step 3: Initialize forensic system
    print("\n[3/5] Initializing forensic system...")
    system = GlassFractureForensicSystem(
        material=material,
        thresholds=thresholds
    )
    print("  System initialized successfully")

    # Step 4: Prepare dummy data (in real use, load actual images)
    print("\n[4/5] Preparing sample data...")
    print("  NOTE: Using synthetic data for demonstration")
    print("        In production, use actual fracture images")

    # Create dummy image sequence
    n_frames = 5
    height, width = 480, 640
    images = [
        np.random.randint(0, 255, (height, width), dtype=np.uint8)
        for _ in range(n_frames)
    ]

    # Camera intrinsic matrix (example values)
    focal_length = 800.0
    cx, cy = width / 2, height / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Dummy fracture masks
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(n_frames)]

    print(f"  Images: {n_frames} frames of {width}x{height}")
    print(f"  Camera: f={focal_length}px, principal point=({cx:.0f}, {cy:.0f})")

    # Step 5: Run analysis
    print("\n[5/5] Running forensic analysis...")
    print("-" * 70)

    report = system.analyze(
        image_sequence=images,
        camera_matrix=K,
        fracture_masks=masks
    )

    print("-" * 70)

    # Display results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    print(f"\nOrigin Estimate:")
    print(f"  Position: {report.origin.position}")
    print(f"  Confidence: {report.origin.confidence:.3f}")
    print(f"  Condition Number: {report.origin.condition_number:.2e}")

    print(f"\nFailure Mode: {report.failure_mode.value}")

    print(f"\nTrajectories: {len(report.trajectories)} detected")
    for i, traj in enumerate(report.trajectories[:3]):  # Show first 3
        print(f"  Trajectory {i+1}:")
        print(f"    Points: {len(traj.points)}")
        print(f"    Curvature: {traj.curvature:.6f}")

    print(f"\nStress Intensity Factors:")
    for i, sif in enumerate(report.stress_factors[:3]):  # Show first 3
        print(f"  Factor {i+1}:")
        print(f"    K_I:  {sif.K_I/1e6:.3f} MPa·√m (Mode I - opening)")
        print(f"    K_II: {sif.K_II/1e6:.3f} MPa·√m (Mode II - sliding)")
        print(f"    Angle: {np.degrees(sif.theta):.1f}°")

    # Compute confidence ellipsoid
    ellipsoid = report.origin.compute_ellipsoid()
    print(f"\n95% Confidence Ellipsoid:")
    print(f"  Center: {ellipsoid['center']}")
    print(f"  Radii: {ellipsoid['radii']}")
    print(f"  Volume: {ellipsoid['volume']:.6e}")

    print(f"\nCapture Quality:")
    print(f"  Valid tracks: {report.capture_quality['n_valid_tracks']}")
    print(f"  Coverage: {report.capture_quality['coverage_fraction']:.1%}")
    print(f"  Status: {'READY' if report.capture_quality['is_ready'] else 'NOT READY'}")

    # Step 6: Save report
    print("\n" + "="*70)
    print("SAVING REPORT")
    print("="*70)

    output_dir = Path(__file__).parent.parent / "output"
    output_path = system.save_report(report, output_dir)

    print(f"\nEvidence Hash: {report.evidence_hash}")
    print(f"Timestamp: {report.timestamp}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nAll results are deterministic and reproducible.")
    print("Ready for expert testimony and legal proceedings.")
    print("\n")


if __name__ == "__main__":
    main()
