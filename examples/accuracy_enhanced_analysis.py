#!/usr/bin/env python3
"""
Accuracy-Enhanced Forensic Analysis Example

Demonstrates integration of accuracy improvements into the
forensic analysis pipeline.

ENHANCEMENTS:
- Accurate parallax computation
- Real spatial coverage assessment
- Uncertainty propagation
- Statistical validation
- Improved branching angle calculation
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glass_fracture_forensics import (
    GlassFractureForensicSystem,
    GlassMaterialProperties,
    SystemThresholds,
    Track2D,
    Trajectory3D,
)

from glass_fracture_forensics.accuracy_improvements import (
    compute_mean_parallax,
    compute_spatial_coverage,
    AccuracyEnhancedCaptureValidator,
    propagate_triangulation_uncertainty,
    compute_reprojection_error,
)

from glass_fracture_forensics.statistical_validation import (
    bootstrap_origin_estimation,
    generate_validation_report,
    robust_outlier_detection,
    OutlierMethod,
)


def improve_branching_angle_calculation(trajectory: Trajectory3D,
                                       origin_position: np.ndarray) -> float:
    """
    Improved branching angle calculation

    Computes angle between trajectory direction and radial direction
    from origin (Mode I direction).

    EQUATION:
    θ = arccos(|d · r̂|)

    where:
    - d: trajectory direction
    - r̂: unit vector from origin to trajectory

    Args:
        trajectory: Trajectory3D object
        origin_position: (3,) fracture origin position

    Returns:
        branching_angle: Angle in radians
    """
    # Vector from origin to trajectory center
    radial_vector = trajectory.origin - origin_position

    # Normalize
    radial_unit = radial_vector / (np.linalg.norm(radial_vector) + 1e-10)

    # Trajectory direction (already normalized)
    traj_dir = trajectory.direction

    # Compute angle
    dot_product = np.abs(np.dot(radial_unit, traj_dir))
    dot_product = np.clip(dot_product, 0.0, 1.0)

    # Branching angle from radial direction
    branching_angle = np.arccos(dot_product)

    return branching_angle


def main():
    """Run accuracy-enhanced forensic analysis"""

    print("="*70)
    print("ACCURACY-ENHANCED FORENSIC ANALYSIS")
    print("="*70)

    # ========================================================================
    # STEP 1: Initialize system with material properties
    # ========================================================================
    print("\n[STEP 1] Initializing forensic system...")
    print("-" * 70)

    material = GlassMaterialProperties(
        E=72.0e9,
        nu=0.23,
        K_Ic=0.75e6,
        rho=2500.0,
        stress_state='plane_stress'
    )

    thresholds = SystemThresholds(
        min_parallax_angle=5.0,
        min_valid_tracks=10,
        ransac_confidence=0.999,
    )

    system = GlassFractureForensicSystem(
        material=material,
        thresholds=thresholds
    )

    print(f"  Material: Soda-lime glass")
    print(f"  K_Ic: {material.K_Ic/1e6:.2f} MPa·√m")
    print(f"  Min parallax: {thresholds.min_parallax_angle}°")

    # ========================================================================
    # STEP 2: Generate synthetic data with realistic geometry
    # ========================================================================
    print("\n[STEP 2] Generating synthetic fracture data...")
    print("-" * 70)

    # Camera parameters
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)

    image_size = (640, 480)

    # Generate realistic tracks
    np.random.seed(42)
    tracks = []

    for i in range(15):
        # Simulate track with realistic motion
        start_point = np.random.rand(2) * np.array([640, 480])
        motion = np.random.randn(2) * 30  # 30 pixel typical motion

        n_frames = np.random.randint(8, 15)
        t = np.linspace(0, 1, n_frames)

        # Linear motion with small noise
        points = start_point + np.outer(t, motion) + np.random.randn(n_frames, 2) * 0.5

        # FB errors
        fb_errors = np.random.rand(n_frames) * 0.3

        track = Track2D(points, np.arange(n_frames), fb_errors)
        track.validate(thresholds.fb_error_threshold)
        tracks.append(track)

    print(f"  Generated {len(tracks)} tracks")
    print(f"  Image size: {image_size}")

    # ========================================================================
    # STEP 3: Accurate capture quality validation
    # ========================================================================
    print("\n[STEP 3] Accurate capture quality validation...")
    print("-" * 70)

    validator = AccuracyEnhancedCaptureValidator(
        thresholds, camera_matrix, image_size
    )

    quality_metrics = validator.validate_tracks_accurate(tracks)

    print(f"  Valid tracks: {quality_metrics['n_valid_tracks']}")
    print(f"  Mean parallax: {quality_metrics['mean_parallax_degrees']:.2f}° "
          f"± {quality_metrics['std_parallax_degrees']:.2f}°")
    print(f"  Coverage fraction: {quality_metrics['coverage_fraction']:.1%}")
    print(f"  Coverage uniformity (CV): {quality_metrics['coverage_uniformity']:.3f}")
    print(f"  Overall quality score: {quality_metrics['quality_score']:.3f}")
    print(f"  Status: {'✓ READY' if quality_metrics['is_ready'] else '✗ NOT READY'}")

    # ========================================================================
    # STEP 4: Generate 3D trajectories with uncertainty
    # ========================================================================
    print("\n[STEP 4] 3D reconstruction with uncertainty propagation...")
    print("-" * 70)

    # Simulate 3D trajectories (in practice, from reconstruction)
    trajectories = []
    trajectory_uncertainties = []

    for i in range(3):
        # Radial pattern from origin
        angle = i * 2 * np.pi / 3
        direction_2d = np.array([np.cos(angle), np.sin(angle)])

        # 3D points along line
        t = np.linspace(0, 0.3, 20)
        points_3d = np.column_stack([
            t * direction_2d[0],
            t * direction_2d[1],
            np.ones(20) * 0.1
        ])

        # Add noise
        points_3d += np.random.randn(20, 3) * 0.005

        traj = Trajectory3D(
            points=points_3d,
            origin=np.zeros(3),
            direction=np.zeros(3)
        )
        traj.fit()

        trajectories.append(traj)

        # Simulate uncertainties (would come from triangulation)
        unc = np.ones((20, 3)) * 0.01  # 1cm standard deviation
        trajectory_uncertainties.append(unc)

    print(f"  Reconstructed {len(trajectories)} trajectories")

    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i+1}:")
        print(f"    Points: {len(traj.points)}")
        print(f"    Direction: {traj.direction}")
        mean_unc = np.mean(trajectory_uncertainties[i], axis=0)
        print(f"    Mean uncertainty: {mean_unc * 1000:.2f} mm")

    # ========================================================================
    # STEP 5: Origin estimation with statistical validation
    # ========================================================================
    print("\n[STEP 5] Origin estimation with bootstrap validation...")
    print("-" * 70)

    # Estimate origin
    origin_estimate = system.origin_estimator.estimate_origin(trajectories)

    print(f"  Origin position: {origin_estimate.position}")
    print(f"  Confidence: {origin_estimate.confidence:.3f}")
    print(f"  Condition number: {origin_estimate.condition_number:.2e}")

    # Bootstrap confidence intervals
    def estimate_origin_wrapper(trajs):
        return system.origin_estimator.estimate_origin(trajs)

    print("\n  Running bootstrap analysis (500 samples)...")
    bootstrap_result = bootstrap_origin_estimation(
        trajectories,
        estimate_origin_wrapper,
        n_bootstrap=500,
        random_seed=42
    )

    print(f"  Bootstrap mean: {bootstrap_result.mean}")
    print(f"  Bootstrap std: {bootstrap_result.std}")
    print(f"  95% CI:")
    print(f"    Lower: {bootstrap_result.confidence_interval[0]}")
    print(f"    Upper: {bootstrap_result.confidence_interval[1]}")

    # Compute CI width
    ci_width = np.linalg.norm(
        bootstrap_result.confidence_interval[1] -
        bootstrap_result.confidence_interval[0]
    )
    print(f"  CI width: {ci_width * 1000:.1f} mm")

    # ========================================================================
    # STEP 6: Improved fracture mechanics analysis
    # ========================================================================
    print("\n[STEP 6] Fracture mechanics with improved branching angles...")
    print("-" * 70)

    stress_factors = []

    for i, traj in enumerate(trajectories):
        # Compute curvature
        curvature = system.mechanics_analyzer.compute_curvature(
            traj.points, thresholds.curvature_window
        )
        traj.curvature = curvature

        # IMPROVED: Compute actual branching angle
        branching_angle = improve_branching_angle_calculation(
            traj, origin_estimate.position
        )

        # Compute stress intensities
        sif = system.mechanics_analyzer.compute_stress_intensity(branching_angle)
        stress_factors.append(sif)

        print(f"  Trajectory {i+1}:")
        print(f"    Curvature: {curvature:.4f}")
        print(f"    Branching angle: {np.degrees(branching_angle):.1f}°")
        print(f"    K_I: {sif.K_I/1e6:.3f} MPa·√m")
        print(f"    K_II: {sif.K_II/1e6:.3f} MPa·√m")

    # ========================================================================
    # STEP 7: Outlier detection
    # ========================================================================
    print("\n[STEP 7] Outlier detection for trajectory quality...")
    print("-" * 70)

    curvatures = np.array([t.curvature for t in trajectories])

    inlier_mask = robust_outlier_detection(
        curvatures.reshape(-1, 1),
        method=OutlierMethod.ZSCORE
    )

    n_outliers = np.sum(~inlier_mask)
    print(f"  Detected {n_outliers} outlier trajectories")
    print(f"  Inlier ratio: {np.mean(inlier_mask):.1%}")

    if n_outliers > 0:
        outlier_indices = np.where(~inlier_mask)[0]
        print(f"  Outlier indices: {outlier_indices}")

    # ========================================================================
    # STEP 8: Failure mode classification
    # ========================================================================
    print("\n[STEP 8] Failure mode classification...")
    print("-" * 70)

    failure_mode = system.classifier.classify(trajectories, origin_estimate)

    print(f"  Classified mode: {failure_mode.value}")

    mean_curvature = np.mean(curvatures)
    eigenvalues, _ = np.linalg.eigh(origin_estimate.covariance)
    max_eigenvalue = np.max(eigenvalues)

    print(f"  Mean curvature: {mean_curvature:.4f}")
    print(f"  Origin spread (max eigenvalue): {np.sqrt(max_eigenvalue) * 1000:.1f} mm")

    # ========================================================================
    # STEP 9: Comprehensive validation report
    # ========================================================================
    print("\n[STEP 9] Generating comprehensive validation report...")
    print("-" * 70)

    def mechanics_analyzer_wrapper(trajs, origin_pos):
        factors = []
        for traj in trajs:
            angle = improve_branching_angle_calculation(traj, origin_pos)
            sif = system.mechanics_analyzer.compute_stress_intensity(angle)
            factors.append(sif)
        return factors

    validation_report = generate_validation_report(
        origin_estimate,
        trajectories,
        stress_factors,
        failure_mode.value,
        estimate_origin_wrapper,
        mechanics_analyzer_wrapper
    )

    # Display key validation metrics
    print("\n  Validation Summary:")
    print(f"    Bootstrap CI width: {ci_width * 1000:.1f} mm")

    if validation_report['cross_validation']['valid']:
        cv = validation_report['cross_validation']
        print(f"    Cross-validation deviation: {cv['mean_deviation'] * 1000:.1f} mm")

    if validation_report['outlier_detection']['valid']:
        outliers = validation_report['outlier_detection']
        print(f"    Outlier ratio: {1 - outliers['inlier_ratio']:.1%}")

    print(f"    Ellipsoid volume: {validation_report['uncertainty']['ellipsoid_volume']:.2e}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("ACCURACY-ENHANCED ANALYSIS COMPLETE")
    print("="*70)

    print(f"\n✓ Origin Position: {origin_estimate.position}")
    print(f"  95% Confidence Interval Width: {ci_width * 1000:.1f} mm")
    print(f"  Statistical Confidence: {origin_estimate.confidence:.3f}")

    print(f"\n✓ Failure Mode: {failure_mode.value}")
    print(f"  Classification Confidence: Based on {len(trajectories)} trajectories")

    print(f"\n✓ Quality Metrics:")
    print(f"  Capture Quality: {quality_metrics['quality_score']:.3f}")
    print(f"  Parallax: {quality_metrics['mean_parallax_degrees']:.1f}°")
    print(f"  Coverage: {quality_metrics['coverage_fraction']:.1%}")

    print(f"\n✓ Stress Intensity:")
    mean_K_I = np.mean([sf.K_I for sf in stress_factors])
    mean_K_II = np.mean([sf.K_II for sf in stress_factors])
    print(f"  Mean K_I: {mean_K_I/1e6:.3f} MPa·√m")
    print(f"  Mean K_II: {mean_K_II/1e6:.3f} MPa·√m")
    print(f"  K_I / K_Ic: {mean_K_I / material.K_Ic:.1%}")

    print("\n" + "="*70)
    print("All results are statistically validated and reproducible")
    print("="*70 + "\n")

    # Save validation report
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    report_path = output_dir / "validation_report.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj

    validation_report_json = convert_numpy(validation_report)

    with open(report_path, 'w') as f:
        json.dump(validation_report_json, f, indent=2)

    print(f"✓ Validation report saved: {report_path}\n")


if __name__ == "__main__":
    main()
