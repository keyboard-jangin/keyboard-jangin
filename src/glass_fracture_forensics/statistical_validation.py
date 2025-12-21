#!/usr/bin/env python3
"""
STATISTICAL VALIDATION TOOLS
=============================

Rigorous statistical methods for uncertainty quantification
and result validation in forensic analysis.

METHODS:
1. Bootstrap resampling for confidence intervals
2. Monte Carlo simulation for error propagation
3. Outlier detection using robust statistics
4. Hypothesis testing for failure mode classification
5. Cross-validation for reconstruction quality

Author: Forensic Engineering Team
Version: 2.0
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from scipy import stats
from enum import Enum


class OutlierMethod(Enum):
    """Outlier detection methods"""
    ZSCORE = "z-score"
    IQR = "interquartile-range"
    MAHALANOBIS = "mahalanobis-distance"
    RANSAC = "ransac"


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis"""
    mean: np.ndarray               # Bootstrap mean estimate
    std: np.ndarray                # Bootstrap standard error
    confidence_interval: Tuple[np.ndarray, np.ndarray]  # (lower, upper) bounds
    samples: np.ndarray            # All bootstrap samples
    confidence_level: float        # Confidence level used


# ============================================================================
# BOOTSTRAP METHODS
# ============================================================================

def bootstrap_origin_estimation(trajectories: List,
                                origin_estimator: Callable,
                                n_bootstrap: int = 1000,
                                confidence_level: float = 0.95,
                                random_seed: Optional[int] = None) -> BootstrapResult:
    """
    Bootstrap confidence intervals for fracture origin

    Resamples trajectories with replacement and re-estimates origin.

    STATISTICAL FOUNDATION:
    - Nonparametric method (no distribution assumptions)
    - Asymptotically valid under regularity conditions
    - Provides empirical confidence intervals

    Args:
        trajectories: List of Trajectory3D objects
        origin_estimator: Function that estimates origin from trajectories
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult with confidence intervals
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_trajectories = len(trajectories)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_trajectories, size=n_trajectories, replace=True)
        resampled_trajectories = [trajectories[i] for i in indices]

        # Re-estimate origin
        try:
            origin_estimate = origin_estimator(resampled_trajectories)
            bootstrap_estimates.append(origin_estimate.position)
        except:
            # Skip failed estimates
            continue

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute statistics
    mean_estimate = np.mean(bootstrap_estimates, axis=0)
    std_estimate = np.std(bootstrap_estimates, axis=0)

    # Compute confidence interval (percentile method)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_estimates, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_estimates, upper_percentile, axis=0)

    return BootstrapResult(
        mean=mean_estimate,
        std=std_estimate,
        confidence_interval=(ci_lower, ci_upper),
        samples=bootstrap_estimates,
        confidence_level=confidence_level
    )


def bootstrap_stress_intensity(trajectories: List,
                               origin_position: np.ndarray,
                               mechanics_analyzer: Callable,
                               n_bootstrap: int = 1000) -> Dict[str, BootstrapResult]:
    """
    Bootstrap confidence intervals for stress intensity factors

    Args:
        trajectories: List of Trajectory3D objects
        origin_position: Estimated origin position
        mechanics_analyzer: Function to compute stress intensities
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with bootstrap results for K_I and K_II
    """
    K_I_samples = []
    K_II_samples = []

    n_trajectories = len(trajectories)

    for _ in range(n_bootstrap):
        # Resample trajectories
        indices = np.random.choice(n_trajectories, size=n_trajectories, replace=True)
        resampled = [trajectories[i] for i in indices]

        # Compute stress intensities
        try:
            stress_factors = mechanics_analyzer(resampled, origin_position)

            # Average K_I and K_II across trajectories
            mean_K_I = np.mean([sf.K_I for sf in stress_factors])
            mean_K_II = np.mean([sf.K_II for sf in stress_factors])

            K_I_samples.append(mean_K_I)
            K_II_samples.append(mean_K_II)
        except:
            continue

    K_I_samples = np.array(K_I_samples).reshape(-1, 1)
    K_II_samples = np.array(K_II_samples).reshape(-1, 1)

    results = {}

    for name, samples in [("K_I", K_I_samples), ("K_II", K_II_samples)]:
        mean = np.mean(samples)
        std = np.std(samples)
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)

        results[name] = BootstrapResult(
            mean=mean,
            std=std,
            confidence_interval=(ci_lower, ci_upper),
            samples=samples,
            confidence_level=0.95
        )

    return results


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_error_propagation(measurement_function: Callable,
                                 nominal_inputs: Dict[str, np.ndarray],
                                 input_uncertainties: Dict[str, float],
                                 n_samples: int = 10000,
                                 random_seed: Optional[int] = None) -> Dict[str, any]:
    """
    Monte Carlo uncertainty propagation

    Samples inputs from distributions and propagates through function.

    EQUATION:
    For Y = f(X₁, X₂, ...):
    Sample Xᵢ ~ N(μᵢ, σᵢ²)
    Compute Yⱼ = f(X₁ⱼ, X₂ⱼ, ...)
    Estimate σ_Y from samples {Yⱼ}

    Args:
        measurement_function: Function to evaluate
        nominal_inputs: Dictionary of nominal input values
        input_uncertainties: Dictionary of input standard deviations
        n_samples: Number of Monte Carlo samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with output statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    output_samples = []

    for _ in range(n_samples):
        # Sample inputs (assuming Gaussian)
        sampled_inputs = {}

        for key, nominal_value in nominal_inputs.items():
            uncertainty = input_uncertainties.get(key, 0.0)

            # Sample from N(nominal, uncertainty²)
            if isinstance(nominal_value, np.ndarray):
                sampled = nominal_value + np.random.randn(*nominal_value.shape) * uncertainty
            else:
                sampled = nominal_value + np.random.randn() * uncertainty

            sampled_inputs[key] = sampled

        # Evaluate function
        try:
            output = measurement_function(**sampled_inputs)
            output_samples.append(output)
        except:
            continue

    output_samples = np.array(output_samples)

    # Compute statistics
    mean = np.mean(output_samples, axis=0)
    std = np.std(output_samples, axis=0)
    median = np.median(output_samples, axis=0)

    # Percentiles
    p5 = np.percentile(output_samples, 5, axis=0)
    p95 = np.percentile(output_samples, 95, axis=0)

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'percentile_5': p5,
        'percentile_95': p95,
        'samples': output_samples
    }


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_zscore(data: np.ndarray,
                          threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using z-score method

    CRITERION:
    |z| = |(x - μ) / σ| > threshold

    Args:
        data: (N,) or (N, D) data array
        threshold: Z-score threshold (typically 3.0)

    Returns:
        inlier_mask: (N,) boolean mask (True = inlier)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Compute z-scores
    z_scores = np.abs((data - mean) / (std + 1e-10))

    # Inliers have z-score below threshold for all dimensions
    inlier_mask = np.all(z_scores < threshold, axis=1)

    return inlier_mask


def detect_outliers_iqr(data: np.ndarray,
                       factor: float = 1.5) -> np.ndarray:
    """
    Detect outliers using Interquartile Range (IQR) method

    CRITERION:
    x < Q1 - factor * IQR  OR  x > Q3 + factor * IQR

    Args:
        data: (N,) or (N, D) data array
        factor: IQR multiplier (typically 1.5)

    Returns:
        inlier_mask: (N,) boolean mask
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Inliers within bounds for all dimensions
    inlier_mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)

    return inlier_mask


def detect_outliers_mahalanobis(data: np.ndarray,
                               threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Mahalanobis distance

    CRITERION:
    D² = (x - μ)ᵀ Σ⁻¹ (x - μ) > threshold²

    Accounts for correlation between dimensions.

    Args:
        data: (N, D) data array
        threshold: Mahalanobis distance threshold

    Returns:
        inlier_mask: (N,) boolean mask
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # Add regularization for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        cov_inv = np.linalg.pinv(cov)

    # Compute Mahalanobis distances
    diff = data - mean
    mahal_dist_sq = np.sum(diff @ cov_inv * diff, axis=1)

    inlier_mask = mahal_dist_sq < threshold ** 2

    return inlier_mask


def robust_outlier_detection(data: np.ndarray,
                            method: OutlierMethod = OutlierMethod.MAHALANOBIS) -> np.ndarray:
    """
    Unified interface for outlier detection

    Args:
        data: (N,) or (N, D) data array
        method: Outlier detection method to use

    Returns:
        inlier_mask: (N,) boolean mask
    """
    if method == OutlierMethod.ZSCORE:
        return detect_outliers_zscore(data)
    elif method == OutlierMethod.IQR:
        return detect_outliers_iqr(data)
    elif method == OutlierMethod.MAHALANOBIS:
        return detect_outliers_mahalanobis(data)
    else:
        raise ValueError(f"Unknown outlier method: {method}")


# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

def test_failure_mode_classification(trajectory_curvatures: np.ndarray,
                                    classified_mode: str,
                                    alpha: float = 0.05) -> Dict[str, any]:
    """
    Statistical test for failure mode classification

    Tests whether curvature distribution is consistent with classified mode.

    Uses one-sample t-test against expected curvature for mode.

    Args:
        trajectory_curvatures: (N,) array of curvature values
        classified_mode: Classified failure mode
        alpha: Significance level

    Returns:
        test_results: Dictionary with test statistics and p-value
    """
    # Expected curvature ranges for each mode
    # (These would be calibrated from training data)
    expected_curvatures = {
        'Point Impact': 0.6,        # High curvature
        'Thermal Shock': 0.05,      # Low curvature
        'Mechanical Fatigue': 0.25  # Medium curvature
    }

    if classified_mode not in expected_curvatures:
        return {'valid': False, 'message': 'Unknown failure mode'}

    expected_value = expected_curvatures[classified_mode]

    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(trajectory_curvatures, expected_value)

    # Test result
    reject_null = p_value < alpha

    return {
        'valid': True,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'reject_null': reject_null,
        'interpretation': (
            f"Classification {'inconsistent' if reject_null else 'consistent'} "
            f"with curvature distribution (p={p_value:.4f})"
        ),
        'mean_curvature': float(np.mean(trajectory_curvatures)),
        'expected_curvature': expected_value
    }


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def leave_one_out_cross_validation(trajectories: List,
                                  origin_estimator: Callable) -> Dict[str, any]:
    """
    Leave-one-out cross-validation for origin estimation

    Estimates origin N times, each time leaving out one trajectory.
    Measures consistency of estimates.

    Args:
        trajectories: List of Trajectory3D objects
        origin_estimator: Function to estimate origin

    Returns:
        cv_results: Cross-validation metrics
    """
    n_trajectories = len(trajectories)

    if n_trajectories < 3:
        return {'valid': False, 'message': 'Need at least 3 trajectories for CV'}

    loo_estimates = []

    for i in range(n_trajectories):
        # Leave out trajectory i
        train_trajectories = trajectories[:i] + trajectories[i+1:]

        # Estimate origin
        try:
            origin_estimate = origin_estimator(train_trajectories)
            loo_estimates.append(origin_estimate.position)
        except:
            continue

    if len(loo_estimates) == 0:
        return {'valid': False, 'message': 'All LOO estimates failed'}

    loo_estimates = np.array(loo_estimates)

    # Compute variation
    mean_estimate = np.mean(loo_estimates, axis=0)
    std_estimate = np.std(loo_estimates, axis=0)

    # Maximum deviation from mean
    deviations = np.linalg.norm(loo_estimates - mean_estimate, axis=1)
    max_deviation = np.max(deviations)
    mean_deviation = np.mean(deviations)

    return {
        'valid': True,
        'n_folds': len(loo_estimates),
        'mean_position': mean_estimate,
        'std_position': std_estimate,
        'mean_deviation': float(mean_deviation),
        'max_deviation': float(max_deviation),
        'cv_estimates': loo_estimates
    }


# ============================================================================
# COMPREHENSIVE VALIDATION REPORT
# ============================================================================

def generate_validation_report(origin_estimate,
                              trajectories: List,
                              stress_factors: List,
                              failure_mode: str,
                              origin_estimator: Callable,
                              mechanics_analyzer: Callable) -> Dict[str, any]:
    """
    Generate comprehensive statistical validation report

    Combines multiple validation methods for forensic report.

    Args:
        origin_estimate: OriginEstimate object
        trajectories: List of Trajectory3D objects
        stress_factors: List of StressIntensityFactors
        failure_mode: Classified failure mode
        origin_estimator: Origin estimation function
        mechanics_analyzer: Fracture mechanics analyzer

    Returns:
        validation_report: Comprehensive validation metrics
    """
    print("Generating statistical validation report...")

    report = {}

    # 1. Bootstrap confidence intervals for origin
    print("  [1/5] Bootstrap analysis...")
    bootstrap_origin = bootstrap_origin_estimation(
        trajectories, origin_estimator, n_bootstrap=500
    )

    report['bootstrap_origin'] = {
        'mean': bootstrap_origin.mean.tolist(),
        'std': bootstrap_origin.std.tolist(),
        'ci_95_lower': bootstrap_origin.confidence_interval[0].tolist(),
        'ci_95_upper': bootstrap_origin.confidence_interval[1].tolist()
    }

    # 2. Cross-validation
    print("  [2/5] Cross-validation...")
    cv_results = leave_one_out_cross_validation(trajectories, origin_estimator)

    report['cross_validation'] = cv_results

    # 3. Outlier detection for trajectories
    print("  [3/5] Outlier detection...")
    curvatures = np.array([t.curvature for t in trajectories if t.curvature > 0])

    if len(curvatures) > 0:
        inlier_mask = robust_outlier_detection(
            curvatures.reshape(-1, 1),
            method=OutlierMethod.MAHALANOBIS
        )

        report['outlier_detection'] = {
            'n_trajectories': len(curvatures),
            'n_inliers': int(np.sum(inlier_mask)),
            'n_outliers': int(np.sum(~inlier_mask)),
            'inlier_ratio': float(np.mean(inlier_mask))
        }
    else:
        report['outlier_detection'] = {'valid': False}

    # 4. Hypothesis test for failure mode
    print("  [4/5] Failure mode validation...")
    if len(curvatures) > 0:
        mode_test = test_failure_mode_classification(curvatures, failure_mode)
        report['failure_mode_test'] = mode_test
    else:
        report['failure_mode_test'] = {'valid': False}

    # 5. Uncertainty metrics
    print("  [5/5] Uncertainty quantification...")
    ellipsoid = origin_estimate.compute_ellipsoid()

    report['uncertainty'] = {
        'ellipsoid_volume': float(ellipsoid['volume']),
        'ellipsoid_radii': ellipsoid['radii'].tolist(),
        'condition_number': float(origin_estimate.condition_number),
        'confidence': float(origin_estimate.confidence)
    }

    print("  ✓ Validation report complete")

    return report


if __name__ == "__main__":
    """
    Demonstration of statistical validation tools
    """
    print("="*70)
    print("STATISTICAL VALIDATION TOOLS - DEMONSTRATION")
    print("="*70)

    # 1. Outlier detection demo
    print("\n1. Outlier Detection:")
    print("-" * 70)

    # Generate data with outliers
    np.random.seed(42)
    normal_data = np.random.randn(100)
    outliers = np.array([10, -10, 15])
    data_with_outliers = np.concatenate([normal_data, outliers])

    for method in [OutlierMethod.ZSCORE, OutlierMethod.IQR]:
        inlier_mask = robust_outlier_detection(data_with_outliers, method=method)
        n_outliers = np.sum(~inlier_mask)
        print(f"  {method.value}: Detected {n_outliers} outliers")

    # 2. Monte Carlo simulation demo
    print("\n2. Monte Carlo Error Propagation:")
    print("-" * 70)

    def simple_function(x, y):
        return x ** 2 + y ** 2

    mc_results = monte_carlo_error_propagation(
        simple_function,
        {'x': 1.0, 'y': 2.0},
        {'x': 0.1, 'y': 0.1},
        n_samples=1000
    )

    print(f"  Nominal output: {simple_function(1.0, 2.0)}")
    print(f"  MC mean: {mc_results['mean']:.4f}")
    print(f"  MC std: {mc_results['std']:.4f}")
    print(f"  90% CI: [{mc_results['percentile_5']:.4f}, {mc_results['percentile_95']:.4f}]")

    print("\n" + "="*70)
    print("Statistical validation tools ready")
    print("="*70)
