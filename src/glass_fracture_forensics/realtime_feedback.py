#!/usr/bin/env python3
"""
REAL-TIME SCAN COVERAGE FEEDBACK SYSTEM
========================================

Provides live visual feedback during AR capture to ensure complete coverage
and high-quality reconstruction for forensic analysis.

FEATURES:
- Voxel-based 3D space tracking
- Real-time coverage visualization
- Quality heatmap generation
- Rescan area detection
- AR overlay guidance

VISUALIZATION:
- Green: Well-scanned areas (sufficient coverage)
- Yellow: Partially scanned (needs more views)
- Red: Unscanned or poor quality (requires scanning)
- Blue: Optimal quality (redundant coverage)

Author: Forensic Engineering Team
Version: 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import cv2


class ScanQuality(Enum):
    """Scan quality levels for each region"""
    UNSCANNED = 0      # Red - No data
    POOR = 1           # Red - Insufficient data
    PARTIAL = 2        # Yellow - Needs more views
    GOOD = 3           # Green - Sufficient coverage
    EXCELLENT = 4      # Blue - Redundant coverage


@dataclass
class VoxelGrid:
    """
    3D VOXEL GRID FOR COVERAGE TRACKING

    Divides the scan volume into cubic voxels and tracks:
    - Number of observations per voxel
    - View angles (for multi-view coverage)
    - Point density
    - Reconstruction quality
    """
    bounds_min: np.ndarray      # (3,) minimum bounds [x, y, z]
    bounds_max: np.ndarray      # (3,) maximum bounds [x, y, z]
    resolution: float           # Voxel size [meters]

    # Internal state
    grid_size: Tuple[int, int, int] = field(init=False)
    observation_count: np.ndarray = field(init=False)
    point_density: np.ndarray = field(init=False)
    view_directions: List[List[np.ndarray]] = field(init=False)
    quality_scores: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize voxel grid arrays"""
        # Calculate grid dimensions
        extent = self.bounds_max - self.bounds_min
        self.grid_size = tuple(np.ceil(extent / self.resolution).astype(int))

        # Initialize tracking arrays
        self.observation_count = np.zeros(self.grid_size, dtype=np.int32)
        self.point_density = np.zeros(self.grid_size, dtype=np.float32)
        self.quality_scores = np.zeros(self.grid_size, dtype=np.float32)

        # View directions per voxel (for multi-view checking)
        self.view_directions = [
            [[] for _ in range(self.grid_size[1])]
            for _ in range(self.grid_size[0])
        ]
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.view_directions[i][j] = [[] for _ in range(self.grid_size[2])]

    def world_to_voxel(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Convert world coordinates to voxel indices

        Args:
            point: (3,) world coordinates [x, y, z]

        Returns:
            (i, j, k) voxel indices or None if out of bounds
        """
        if np.any(point < self.bounds_min) or np.any(point > self.bounds_max):
            return None

        relative = point - self.bounds_min
        indices = (relative / self.resolution).astype(int)

        # Clamp to grid bounds
        indices = np.clip(indices, 0, np.array(self.grid_size) - 1)

        return tuple(indices)

    def voxel_to_world(self, indices: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert voxel indices to world coordinates (voxel center)

        Args:
            indices: (i, j, k) voxel indices

        Returns:
            (3,) world coordinates of voxel center
        """
        return self.bounds_min + (np.array(indices) + 0.5) * self.resolution

    def update_voxel(self, point: np.ndarray, view_direction: np.ndarray):
        """
        Update voxel with new observation

        Args:
            point: (3,) 3D point in world coordinates
            view_direction: (3,) normalized camera viewing direction
        """
        voxel_idx = self.world_to_voxel(point)
        if voxel_idx is None:
            return

        i, j, k = voxel_idx

        # Increment observation count
        self.observation_count[i, j, k] += 1

        # Add view direction (for multi-view coverage check)
        self.view_directions[i][j][k].append(view_direction)

        # Update point density (running average)
        self.point_density[i, j, k] += 1.0

    def compute_quality(self,
                       min_observations: int = 3,
                       min_view_diversity: float = 30.0) -> None:
        """
        Compute quality score for each voxel

        Quality criteria:
        1. Observation count >= min_observations
        2. View diversity >= min_view_diversity (degrees)
        3. Point density

        Args:
            min_observations: Minimum views per voxel for GOOD quality
            min_view_diversity: Minimum angle between views (degrees)
        """
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    obs_count = self.observation_count[i, j, k]

                    if obs_count == 0:
                        self.quality_scores[i, j, k] = ScanQuality.UNSCANNED.value
                        continue

                    # Check view diversity
                    views = self.view_directions[i][j][k]
                    if len(views) < 2:
                        view_diversity = 0.0
                    else:
                        # Compute maximum angle between views
                        max_angle = 0.0
                        for v1_idx in range(len(views)):
                            for v2_idx in range(v1_idx + 1, len(views)):
                                v1 = views[v1_idx]
                                v2 = views[v2_idx]
                                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                                angle = np.degrees(np.arccos(dot))
                                max_angle = max(max_angle, angle)
                        view_diversity = max_angle

                    # Assign quality level
                    if obs_count < min_observations:
                        if obs_count == 1:
                            quality = ScanQuality.POOR.value
                        else:
                            quality = ScanQuality.PARTIAL.value
                    elif view_diversity < min_view_diversity:
                        quality = ScanQuality.PARTIAL.value
                    elif obs_count >= 2 * min_observations:
                        quality = ScanQuality.EXCELLENT.value
                    else:
                        quality = ScanQuality.GOOD.value

                    self.quality_scores[i, j, k] = quality

    def get_coverage_stats(self) -> Dict[str, float]:
        """
        Compute overall coverage statistics

        Returns:
            Dictionary with coverage metrics
        """
        total_voxels = np.prod(self.grid_size)
        observed_voxels = np.sum(self.observation_count > 0)

        quality_counts = {
            'unscanned': np.sum(self.quality_scores == ScanQuality.UNSCANNED.value),
            'poor': np.sum(self.quality_scores == ScanQuality.POOR.value),
            'partial': np.sum(self.quality_scores == ScanQuality.PARTIAL.value),
            'good': np.sum(self.quality_scores == ScanQuality.GOOD.value),
            'excellent': np.sum(self.quality_scores == ScanQuality.EXCELLENT.value),
        }

        return {
            'total_voxels': int(total_voxels),
            'observed_voxels': int(observed_voxels),
            'coverage_ratio': float(observed_voxels / total_voxels),
            'good_quality_ratio': float(
                (quality_counts['good'] + quality_counts['excellent']) / total_voxels
            ),
            'needs_rescan_ratio': float(
                (quality_counts['unscanned'] + quality_counts['poor'] +
                 quality_counts['partial']) / total_voxels
            ),
            'quality_distribution': quality_counts
        }


@dataclass
class ScanCoverageTracker:
    """
    REAL-TIME SCAN COVERAGE TRACKER

    Tracks scan progress and provides visual feedback during AR capture.
    """
    voxel_grid: VoxelGrid
    camera_matrix: np.ndarray      # (3, 3) camera intrinsics

    # Quality thresholds
    min_observations: int = 3
    min_view_diversity: float = 30.0  # degrees
    min_coverage_ratio: float = 0.8
    min_good_quality_ratio: float = 0.7

    def update_from_points(self,
                          points_3d: np.ndarray,
                          camera_pose: np.ndarray) -> None:
        """
        Update coverage from new 3D points

        Args:
            points_3d: (N, 3) 3D points in world coordinates
            camera_pose: (4, 4) camera pose matrix [R|t]
        """
        # Extract camera position and viewing direction
        camera_pos = camera_pose[:3, 3]
        camera_forward = camera_pose[:3, 2]  # Z-axis points forward

        # Update each point
        for point in points_3d:
            # Compute view direction from camera to point
            view_dir = point - camera_pos
            view_dir_norm = view_dir / (np.linalg.norm(view_dir) + 1e-8)

            self.voxel_grid.update_voxel(point, view_dir_norm)

    def compute_coverage_quality(self) -> None:
        """Compute quality scores for all voxels"""
        self.voxel_grid.compute_quality(
            self.min_observations,
            self.min_view_diversity
        )

    def get_rescan_regions(self) -> List[np.ndarray]:
        """
        Get list of regions that need rescanning

        Returns:
            List of voxel center positions that need more coverage
        """
        rescan_voxels = []

        quality = self.voxel_grid.quality_scores

        for i in range(self.voxel_grid.grid_size[0]):
            for j in range(self.voxel_grid.grid_size[1]):
                for k in range(self.voxel_grid.grid_size[2]):
                    if quality[i, j, k] <= ScanQuality.PARTIAL.value:
                        # Convert to world coordinates
                        pos = self.voxel_grid.voxel_to_world((i, j, k))
                        rescan_voxels.append(pos)

        return rescan_voxels

    def is_scan_complete(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if scan meets quality requirements

        Returns:
            (is_complete, statistics)
        """
        stats = self.voxel_grid.get_coverage_stats()

        is_complete = (
            stats['coverage_ratio'] >= self.min_coverage_ratio and
            stats['good_quality_ratio'] >= self.min_good_quality_ratio
        )

        return is_complete, stats

    def generate_heatmap_2d(self,
                           camera_pose: np.ndarray,
                           image_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate 2D quality heatmap overlay for current camera view

        Projects voxel grid onto image plane and colors by quality.

        Args:
            camera_pose: (4, 4) current camera pose
            image_size: (width, height) of output image

        Returns:
            (H, W, 3) RGB heatmap image
        """
        width, height = image_size
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        depth_buffer = np.full((height, width), np.inf)

        # Quality colors (BGR format for OpenCV)
        quality_colors = {
            ScanQuality.UNSCANNED.value: (0, 0, 255),      # Red
            ScanQuality.POOR.value: (0, 0, 200),           # Dark red
            ScanQuality.PARTIAL.value: (0, 255, 255),      # Yellow
            ScanQuality.GOOD.value: (0, 255, 0),           # Green
            ScanQuality.EXCELLENT.value: (255, 0, 0),      # Blue
        }

        # Camera extrinsics
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

        # Project voxels
        for i in range(self.voxel_grid.grid_size[0]):
            for j in range(self.voxel_grid.grid_size[1]):
                for k in range(self.voxel_grid.grid_size[2]):
                    # Skip unobserved voxels for clarity
                    if self.voxel_grid.observation_count[i, j, k] == 0:
                        continue

                    # Get voxel center in world coords
                    world_pos = self.voxel_grid.voxel_to_world((i, j, k))

                    # Transform to camera coords
                    cam_pos = R.T @ (world_pos - t)

                    # Skip if behind camera
                    if cam_pos[2] <= 0:
                        continue

                    # Project to image
                    pixel = self.camera_matrix @ cam_pos
                    pixel = pixel[:2] / pixel[2]

                    px, py = int(pixel[0]), int(pixel[1])

                    # Check bounds
                    if 0 <= px < width and 0 <= py < height:
                        # Depth test
                        depth = cam_pos[2]
                        if depth < depth_buffer[py, px]:
                            depth_buffer[py, px] = depth

                            # Get quality color
                            quality_level = int(self.voxel_grid.quality_scores[i, j, k])
                            color = quality_colors.get(quality_level, (128, 128, 128))

                            # Draw voxel projection (small circle)
                            cv2.circle(heatmap, (px, py), 3, color, -1)

        return heatmap


class ARFeedbackOverlay:
    """
    AR OVERLAY FOR REAL-TIME GUIDANCE

    Provides visual guidance during capture:
    - Coverage heatmap
    - Progress indicators
    - Direction arrows to uncovered areas
    - Quality warnings
    """

    def __init__(self, image_size: Tuple[int, int]):
        self.width, self.height = image_size

    def create_overlay(self,
                      heatmap: np.ndarray,
                      coverage_stats: Dict[str, float],
                      rescan_hints: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Create complete AR overlay

        Args:
            heatmap: (H, W, 3) quality heatmap
            coverage_stats: Coverage statistics
            rescan_hints: List of (x, y) pixel coordinates to highlight

        Returns:
            (H, W, 4) RGBA overlay image with transparency
        """
        # Create RGBA overlay
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Add semi-transparent heatmap
        overlay[:, :, :3] = heatmap
        overlay[:, :, 3] = (heatmap.sum(axis=2) > 0).astype(np.uint8) * 128  # 50% transparency

        # Add progress bar
        self._draw_progress_bar(overlay, coverage_stats)

        # Add quality indicator
        self._draw_quality_indicator(overlay, coverage_stats)

        # Add rescan hints
        if rescan_hints:
            self._draw_rescan_hints(overlay, rescan_hints)

        return overlay

    def _draw_progress_bar(self, overlay: np.ndarray, stats: Dict[str, float]) -> None:
        """Draw coverage progress bar at top of screen"""
        bar_height = 30
        bar_margin = 20
        bar_width = self.width - 2 * bar_margin

        coverage = stats['coverage_ratio']
        good_quality = stats['good_quality_ratio']

        # Background
        cv2.rectangle(overlay,
                     (bar_margin, 10),
                     (bar_margin + bar_width, 10 + bar_height),
                     (50, 50, 50, 200), -1)

        # Coverage bar (yellow)
        coverage_width = int(bar_width * coverage)
        cv2.rectangle(overlay,
                     (bar_margin, 10),
                     (bar_margin + coverage_width, 10 + bar_height),
                     (0, 255, 255, 200), -1)

        # Good quality bar (green)
        good_width = int(bar_width * good_quality)
        cv2.rectangle(overlay,
                     (bar_margin, 10),
                     (bar_margin + good_width, 10 + bar_height),
                     (0, 255, 0, 200), -1)

        # Text
        text = f"Coverage: {coverage*100:.0f}% | Quality: {good_quality*100:.0f}%"
        cv2.putText(overlay, text, (bar_margin + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 2)

    def _draw_quality_indicator(self, overlay: np.ndarray, stats: Dict[str, float]) -> None:
        """Draw overall quality indicator"""
        x, y = self.width - 150, self.height - 100

        good_ratio = stats['good_quality_ratio']

        if good_ratio >= 0.9:
            color = (0, 255, 0, 200)  # Green - excellent
            status = "EXCELLENT"
        elif good_ratio >= 0.7:
            color = (0, 255, 0, 200)  # Green - good
            status = "GOOD"
        elif good_ratio >= 0.5:
            color = (0, 255, 255, 200)  # Yellow - needs more
            status = "PARTIAL"
        else:
            color = (0, 0, 255, 200)  # Red - insufficient
            status = "INSUFFICIENT"

        # Status box
        cv2.rectangle(overlay, (x, y), (x + 140, y + 80), color, -1)
        cv2.rectangle(overlay, (x, y), (x + 140, y + 80), (255, 255, 255, 255), 2)

        # Text
        cv2.putText(overlay, "SCAN QUALITY", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 255), 1)
        cv2.putText(overlay, status, (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255, 255), 2)

    def _draw_rescan_hints(self, overlay: np.ndarray, hints: List[Tuple[int, int]]) -> None:
        """Draw arrows pointing to areas needing rescanning"""
        for x, y in hints[:5]:  # Limit to 5 hints for clarity
            # Draw pulsing circle
            cv2.circle(overlay, (x, y), 15, (0, 0, 255, 200), 2)
            cv2.circle(overlay, (x, y), 10, (0, 0, 255, 150), 2)

            # Draw "RESCAN" text
            cv2.putText(overlay, "!", (x - 5, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255, 255), 2)


# ============================================================================
# INTEGRATION WITH EXISTING SYSTEM
# ============================================================================

def integrate_realtime_feedback(forensic_system,
                               scan_bounds: Tuple[np.ndarray, np.ndarray],
                               voxel_resolution: float = 0.05) -> ScanCoverageTracker:
    """
    Integrate real-time feedback into existing forensic system

    Args:
        forensic_system: GlassFractureForensicSystem instance
        scan_bounds: (min_bounds, max_bounds) in world coordinates
        voxel_resolution: Voxel size in meters

    Returns:
        ScanCoverageTracker instance
    """
    min_bounds, max_bounds = scan_bounds

    voxel_grid = VoxelGrid(
        bounds_min=min_bounds,
        bounds_max=max_bounds,
        resolution=voxel_resolution
    )

    # Get camera matrix from forensic system (would be passed during capture)
    # For now, use a placeholder
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)

    tracker = ScanCoverageTracker(
        voxel_grid=voxel_grid,
        camera_matrix=camera_matrix
    )

    return tracker


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example real-time feedback demo"""

    print("="*70)
    print("REAL-TIME SCAN COVERAGE FEEDBACK SYSTEM")
    print("="*70)

    # Define scan volume (e.g., 1m x 1m x 0.5m)
    scan_bounds = (
        np.array([-0.5, -0.5, 0.0]),  # min
        np.array([0.5, 0.5, 0.5])     # max
    )

    # Create voxel grid (2cm resolution)
    voxel_grid = VoxelGrid(
        bounds_min=scan_bounds[0],
        bounds_max=scan_bounds[1],
        resolution=0.02  # 2cm voxels
    )

    print(f"\nVoxel Grid:")
    print(f"  Bounds: {scan_bounds[0]} to {scan_bounds[1]}")
    print(f"  Resolution: {voxel_grid.resolution}m")
    print(f"  Grid size: {voxel_grid.grid_size}")
    print(f"  Total voxels: {np.prod(voxel_grid.grid_size)}")

    # Simulate scanning with random points
    print("\nSimulating scan...")
    np.random.seed(42)

    for frame in range(10):
        # Generate random 3D points
        points = np.random.uniform(
            scan_bounds[0],
            scan_bounds[1],
            (50, 3)
        )

        # Random view direction
        view_dir = np.array([0, 0, -1])  # Looking down
        view_dir += np.random.randn(3) * 0.3
        view_dir /= np.linalg.norm(view_dir)

        # Update grid
        for point in points:
            voxel_grid.update_voxel(point, view_dir)

    # Compute quality
    voxel_grid.compute_quality()

    # Get statistics
    stats = voxel_grid.get_coverage_stats()

    print(f"\nCoverage Statistics:")
    print(f"  Total voxels: {stats['total_voxels']}")
    print(f"  Observed voxels: {stats['observed_voxels']}")
    print(f"  Coverage ratio: {stats['coverage_ratio']:.1%}")
    print(f"  Good quality ratio: {stats['good_quality_ratio']:.1%}")
    print(f"  Needs rescan: {stats['needs_rescan_ratio']:.1%}")

    print(f"\nQuality Distribution:")
    for quality, count in stats['quality_distribution'].items():
        print(f"  {quality}: {count} voxels ({count/stats['total_voxels']*100:.1f}%)")

    print("\n" + "="*70)
    print("Real-time feedback system ready for integration")
    print("="*70)
