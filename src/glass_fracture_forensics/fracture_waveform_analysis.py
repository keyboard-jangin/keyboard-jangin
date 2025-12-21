#!/usr/bin/env python3
"""
FRACTURE WAVEFORM ANALYSIS
===========================

Analyzes fracture patterns as waveforms for detailed characterization.
Extracts geometric signatures and physical properties from crack patterns.

FEATURES:
- Crack path extraction and parameterization
- Waveform generation along crack paths
- Frequency analysis (FFT)
- Tortuosity and roughness metrics
- Crack propagation direction inference
- Pattern matching for failure mode

PHYSICAL BASIS:
- Crack roughness relates to loading rate
- Periodic patterns indicate stress oscillations
- Branching frequency correlates with energy release rate

Author: Forensic Engineering Team
Version: 2.1
"""

import numpy as np
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import cv2


@dataclass
class CrackPath:
    """
    Parameterized crack path

    Represents crack as continuous curve in 2D/3D space
    """
    points: np.ndarray          # (N, 2) or (N, 3) coordinates
    arc_lengths: np.ndarray     # (N,) cumulative arc length
    tangents: np.ndarray        # (N, 2/3) unit tangent vectors
    curvatures: np.ndarray      # (N,) local curvature values
    total_length: float         # Total path length

    @classmethod
    def from_points(cls, points: np.ndarray):
        """Create CrackPath from point sequence"""
        # Compute arc lengths
        if points.shape[0] < 2:
            raise ValueError("Need at least 2 points")

        segments = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = arc_lengths[-1]

        # Compute tangents (forward differences)
        tangents = np.zeros_like(points)
        for i in range(len(points) - 1):
            tangents[i] = segments[i] / (segment_lengths[i] + 1e-10)
        tangents[-1] = tangents[-2]  # Repeat last

        # Compute curvatures (discrete)
        curvatures = np.zeros(len(points))
        for i in range(1, len(points) - 1):
            t1 = tangents[i - 1]
            t2 = tangents[i + 1]
            dt = t2 - t1
            ds = arc_lengths[i + 1] - arc_lengths[i - 1]
            curvatures[i] = np.linalg.norm(dt) / (ds + 1e-10)

        return cls(
            points=points,
            arc_lengths=arc_lengths,
            tangents=tangents,
            curvatures=curvatures,
            total_length=total_length
        )


@dataclass
class FractureWaveform:
    """
    Waveform representation of fracture pattern

    Properties:
    - Amplitude: deviation from straight line
    - Frequency: oscillation characteristics
    - Phase: relative position along crack
    """
    arc_length: np.ndarray      # (N,) sampling positions
    amplitude: np.ndarray       # (N,) perpendicular deviation
    angle: np.ndarray           # (N,) tangent angle
    curvature: np.ndarray       # (N,) local curvature

    # Frequency domain
    frequencies: np.ndarray = field(default=None)
    power_spectrum: np.ndarray = field(default=None)
    dominant_frequency: float = 0.0

    def compute_fft(self):
        """Compute FFT of amplitude signal"""
        N = len(self.amplitude)

        # Remove mean (DC component)
        signal_centered = self.amplitude - np.mean(self.amplitude)

        # Apply window to reduce spectral leakage
        window = signal.hann(N)
        signal_windowed = signal_centered * window

        # FFT
        fft_values = fft(signal_windowed)
        fft_freq = fftfreq(N, d=(self.arc_length[1] - self.arc_length[0]))

        # Power spectrum (positive frequencies only)
        positive_freq_idx = fft_freq > 0
        self.frequencies = fft_freq[positive_freq_idx]
        self.power_spectrum = np.abs(fft_values[positive_freq_idx]) ** 2

        # Dominant frequency
        if len(self.power_spectrum) > 0:
            max_idx = np.argmax(self.power_spectrum)
            self.dominant_frequency = self.frequencies[max_idx]


class CrackExtractor:
    """
    CRACK PATH EXTRACTION FROM BINARY MASK

    Extracts centerline of crack from binary segmentation mask.

    ALGORITHM:
    1. Skeletonize binary mask
    2. Extract connected components
    3. Order points along each component
    4. Smooth path (optional)
    """

    def __init__(self, min_crack_length: int = 50):
        self.min_crack_length = min_crack_length

    def extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract skeleton from binary mask

        Args:
            mask: Binary mask (255 = crack)

        Returns:
            Skeleton image
        """
        # Ensure binary
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Morphological thinning (skeletonization)
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            # Erosion
            eroded = cv2.erode(binary, element)

            # Opening
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)

            # Difference
            subset = binary - opened

            # Union
            skeleton = cv2.bitwise_or(skeleton, subset)

            # Update
            binary = eroded.copy()

            # Stop when nothing left
            if cv2.countNonZero(binary) == 0:
                break

        return skeleton

    def extract_paths(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """
        Extract ordered point paths from skeleton

        Args:
            skeleton: Skeletonized image

        Returns:
            List of paths, each (N, 2) array
        """
        # Find all skeleton points
        points = np.column_stack(np.where(skeleton > 0))

        if len(points) == 0:
            return []

        # Simple connected components
        # (More sophisticated: use graph traversal)
        paths = []

        # For demo, just return points in row-major order
        # In production, would use proper path extraction
        if len(points) >= self.min_crack_length:
            # Sort by x-coordinate as simple approximation
            sorted_points = points[np.argsort(points[:, 1])]
            paths.append(sorted_points[:, [1, 0]])  # Swap to (x, y)

        return paths


class WaveformAnalyzer:
    """
    WAVEFORM ANALYSIS FOR FRACTURE CHARACTERIZATION

    Computes waveform metrics from crack path.
    """

    def __init__(self, sampling_points: int = 1000):
        self.sampling_points = sampling_points

    def path_to_waveform(self, crack_path: CrackPath) -> FractureWaveform:
        """
        Convert crack path to waveform

        Computes perpendicular deviation from best-fit line.

        Args:
            crack_path: CrackPath object

        Returns:
            FractureWaveform
        """
        points = crack_path.points

        # Fit line to path (principal axis)
        mean_point = np.mean(points, axis=0)
        centered = points - mean_point

        # PCA
        if centered.shape[1] == 2:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            principal_axis = eigenvectors[:, -1]  # Largest eigenvalue
        else:
            # 3D case
            _, _, Vt = np.linalg.svd(centered)
            principal_axis = Vt[0]

        # Project points onto principal axis
        projections = centered @ principal_axis

        # Compute perpendicular distances (amplitude)
        if centered.shape[1] == 2:
            # 2D perpendicular distance
            perpendicular_axis = np.array([-principal_axis[1], principal_axis[0]])
            amplitude = centered @ perpendicular_axis
        else:
            # 3D perpendicular distance
            proj_vectors = np.outer(projections, principal_axis)
            perp_vectors = centered - proj_vectors
            amplitude = np.linalg.norm(perp_vectors, axis=1)

        # Resample uniformly along arc length
        arc_length_uniform = np.linspace(0, crack_path.total_length, self.sampling_points)

        # Interpolate amplitude
        f_amp = interpolate.interp1d(crack_path.arc_lengths, amplitude, kind='cubic',
                                     fill_value='extrapolate')
        amplitude_uniform = f_amp(arc_length_uniform)

        # Interpolate curvature
        f_curv = interpolate.interp1d(crack_path.arc_lengths, crack_path.curvatures,
                                      kind='cubic', fill_value='extrapolate')
        curvature_uniform = f_curv(arc_length_uniform)

        # Compute tangent angles
        tangent_angles = np.zeros(self.sampling_points)
        for i in range(1, self.sampling_points):
            idx_lo = np.searchsorted(crack_path.arc_lengths, arc_length_uniform[i - 1])
            idx_hi = np.searchsorted(crack_path.arc_lengths, arc_length_uniform[i])

            if idx_hi < len(crack_path.tangents):
                tangent = crack_path.tangents[idx_hi]
                if len(tangent) == 2:
                    tangent_angles[i] = np.arctan2(tangent[1], tangent[0])

        waveform = FractureWaveform(
            arc_length=arc_length_uniform,
            amplitude=amplitude_uniform,
            angle=tangent_angles,
            curvature=curvature_uniform
        )

        return waveform

    def compute_tortuosity(self, crack_path: CrackPath) -> float:
        """
        Compute crack tortuosity

        DEFINITION:
        T = L_actual / L_straight

        where:
        - L_actual: actual path length
        - L_straight: straight-line distance

        Returns:
            tortuosity: T ≥ 1 (1 = perfectly straight)
        """
        straight_line_length = np.linalg.norm(
            crack_path.points[-1] - crack_path.points[0]
        )

        if straight_line_length < 1e-10:
            return 1.0

        tortuosity = crack_path.total_length / straight_line_length

        return tortuosity

    def compute_roughness(self, waveform: FractureWaveform) -> Dict[str, float]:
        """
        Compute crack surface roughness metrics

        Returns:
            Dictionary with roughness measures
        """
        # RMS roughness
        rms = np.sqrt(np.mean(waveform.amplitude ** 2))

        # Peak-to-valley
        peak_to_valley = np.max(waveform.amplitude) - np.min(waveform.amplitude)

        # Average absolute deviation
        mean_abs = np.mean(np.abs(waveform.amplitude))

        return {
            'rms_roughness': float(rms),
            'peak_to_valley': float(peak_to_valley),
            'mean_absolute_deviation': float(mean_abs)
        }


class PatternMatcher:
    """
    PATTERN MATCHING FOR FAILURE MODE IDENTIFICATION

    Matches crack patterns to known failure signatures.
    """

    def __init__(self):
        # Template patterns for different failure modes
        # (Would be learned from training data)
        self.templates = {
            'Point Impact': {
                'tortuosity_range': (1.0, 1.3),
                'dominant_freq_range': (0.5, 2.0),
                'rms_roughness_range': (0.5, 3.0)
            },
            'Thermal Shock': {
                'tortuosity_range': (1.0, 1.1),
                'dominant_freq_range': (0.0, 0.3),
                'rms_roughness_range': (0.1, 0.5)
            },
            'Mechanical Fatigue': {
                'tortuosity_range': (1.2, 1.8),
                'dominant_freq_range': (1.0, 5.0),
                'rms_roughness_range': (1.0, 5.0)
            }
        }

    def match_pattern(self,
                     tortuosity: float,
                     dominant_frequency: float,
                     rms_roughness: float) -> Dict[str, float]:
        """
        Match crack pattern to failure mode templates

        Args:
            tortuosity: Path tortuosity
            dominant_frequency: Dominant FFT frequency
            rms_roughness: RMS surface roughness

        Returns:
            Dictionary of match scores for each mode
        """
        scores = {}

        for mode, template in self.templates.items():
            # Score based on range membership
            score = 0.0

            # Tortuosity score
            t_range = template['tortuosity_range']
            if t_range[0] <= tortuosity <= t_range[1]:
                score += 1.0
            else:
                # Distance from range
                dist = min(abs(tortuosity - t_range[0]), abs(tortuosity - t_range[1]))
                score += max(0, 1.0 - dist / 2.0)

            # Frequency score
            f_range = template['dominant_freq_range']
            if f_range[0] <= dominant_frequency <= f_range[1]:
                score += 1.0
            else:
                dist = min(abs(dominant_frequency - f_range[0]),
                          abs(dominant_frequency - f_range[1]))
                score += max(0, 1.0 - dist / 5.0)

            # Roughness score
            r_range = template['rms_roughness_range']
            if r_range[0] <= rms_roughness <= r_range[1]:
                score += 1.0
            else:
                dist = min(abs(rms_roughness - r_range[0]),
                          abs(rms_roughness - r_range[1]))
                score += max(0, 1.0 - dist / 3.0)

            # Normalize to [0, 1]
            scores[mode] = score / 3.0

        return scores


# ============================================================================
# INTEGRATION
# ============================================================================

def analyze_fracture_from_mask(mask: np.ndarray) -> Dict[str, any]:
    """
    Complete waveform analysis from fracture mask

    Args:
        mask: Binary fracture mask

    Returns:
        Analysis results dictionary
    """
    # Extract crack paths
    extractor = CrackExtractor(min_crack_length=50)
    skeleton = extractor.extract_skeleton(mask)
    paths = extractor.extract_paths(skeleton)

    if len(paths) == 0:
        return {'valid': False, 'message': 'No crack paths extracted'}

    results = []

    analyzer = WaveformAnalyzer(sampling_points=500)
    matcher = PatternMatcher()

    for i, path_points in enumerate(paths):
        if len(path_points) < 10:
            continue

        # Create crack path
        try:
            crack_path = CrackPath.from_points(path_points)
        except:
            continue

        # Convert to waveform
        waveform = analyzer.path_to_waveform(crack_path)
        waveform.compute_fft()

        # Compute metrics
        tortuosity = analyzer.compute_tortuosity(crack_path)
        roughness = analyzer.compute_roughness(waveform)

        # Pattern matching
        pattern_scores = matcher.match_pattern(
            tortuosity,
            waveform.dominant_frequency,
            roughness['rms_roughness']
        )

        path_result = {
            'path_id': i,
            'num_points': len(path_points),
            'total_length': crack_path.total_length,
            'tortuosity': tortuosity,
            'roughness': roughness,
            'dominant_frequency': waveform.dominant_frequency,
            'pattern_match_scores': pattern_scores
        }

        results.append(path_result)

    return {
        'valid': True,
        'num_paths': len(results),
        'paths': results
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demo waveform analysis"""

    print("="*70)
    print("FRACTURE WAVEFORM ANALYSIS - DEMO")
    print("="*70)

    # Create synthetic crack path
    print("\n1. Synthetic Crack Path:")
    print("-" * 70)

    # Sinusoidal crack (simulated thermal shock)
    x = np.linspace(0, 10, 200)
    y = 5 + 0.3 * np.sin(2 * np.pi * x / 5)  # Low frequency
    points = np.column_stack([x, y])

    crack_path = CrackPath.from_points(points)

    print(f"  Total length: {crack_path.total_length:.2f}")
    print(f"  Mean curvature: {np.mean(crack_path.curvatures):.4f}")

    # Waveform analysis
    print("\n2. Waveform Analysis:")
    print("-" * 70)

    analyzer = WaveformAnalyzer(sampling_points=500)
    waveform = analyzer.path_to_waveform(crack_path)
    waveform.compute_fft()

    tortuosity = analyzer.compute_tortuosity(crack_path)
    roughness = analyzer.compute_roughness(waveform)

    print(f"  Tortuosity: {tortuosity:.3f}")
    print(f"  RMS roughness: {roughness['rms_roughness']:.3f}")
    print(f"  Dominant frequency: {waveform.dominant_frequency:.3f}")

    # Pattern matching
    print("\n3. Pattern Matching:")
    print("-" * 70)

    matcher = PatternMatcher()
    scores = matcher.match_pattern(
        tortuosity,
        waveform.dominant_frequency,
        roughness['rms_roughness']
    )

    for mode, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {score:.3f}")

    print("\n" + "="*70)
    print("Waveform analysis complete")
    print("="*70)
