#!/usr/bin/env python3
"""
Unit tests for real-time feedback system
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glass_fracture_forensics import (
    VoxelGrid,
    ScanCoverageTracker,
    ScanQuality,
    ARFeedbackOverlay,
)


class TestVoxelGrid:
    """Test voxel grid functionality"""

    def test_initialization(self):
        """Test voxel grid initialization"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([1, 1, 1])
        resolution = 0.1

        grid = VoxelGrid(bounds_min, bounds_max, resolution)

        # Grid should be 10x10x10
        assert grid.grid_size == (10, 10, 10)
        assert grid.observation_count.shape == (10, 10, 10)

    def test_world_to_voxel(self):
        """Test coordinate conversion"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([1, 1, 1])
        resolution = 0.1

        grid = VoxelGrid(bounds_min, bounds_max, resolution)

        # Test center of grid
        point = np.array([0.5, 0.5, 0.5])
        voxel_idx = grid.world_to_voxel(point)
        assert voxel_idx == (5, 5, 5)

        # Test corner
        point = np.array([0.05, 0.05, 0.05])
        voxel_idx = grid.world_to_voxel(point)
        assert voxel_idx == (0, 0, 0)

        # Test out of bounds
        point = np.array([2.0, 2.0, 2.0])
        voxel_idx = grid.world_to_voxel(point)
        assert voxel_idx is None

    def test_voxel_to_world(self):
        """Test reverse coordinate conversion"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([1, 1, 1])
        resolution = 0.1

        grid = VoxelGrid(bounds_min, bounds_max, resolution)

        # Voxel (5, 5, 5) center should be at (0.55, 0.55, 0.55)
        world_pos = grid.voxel_to_world((5, 5, 5))
        expected = np.array([0.55, 0.55, 0.55])
        np.testing.assert_allclose(world_pos, expected, rtol=1e-5)

    def test_update_voxel(self):
        """Test voxel update with observations"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([1, 1, 1])
        resolution = 0.1

        grid = VoxelGrid(bounds_min, bounds_max, resolution)

        # Add observation
        point = np.array([0.5, 0.5, 0.5])
        view_dir = np.array([0, 0, -1])

        grid.update_voxel(point, view_dir)

        # Check observation count
        assert grid.observation_count[5, 5, 5] == 1
        assert grid.point_density[5, 5, 5] == 1.0

    def test_quality_computation(self):
        """Test quality score computation"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([0.3, 0.3, 0.3])
        resolution = 0.1

        grid = VoxelGrid(bounds_min, bounds_max, resolution)

        # Single observation - should be POOR
        point = np.array([0.15, 0.15, 0.15])
        view1 = np.array([0, 0, -1])
        grid.update_voxel(point, view1)

        grid.compute_quality(min_observations=3)
        assert grid.quality_scores[1, 1, 1] == ScanQuality.POOR.value

        # Add more observations from same direction - still PARTIAL
        grid.update_voxel(point, view1)
        grid.compute_quality(min_observations=3)
        assert grid.quality_scores[1, 1, 1] == ScanQuality.PARTIAL.value

        # Add observations from different direction - should improve
        view2 = np.array([1, 0, 0])  # 90 degrees different
        grid.update_voxel(point, view2)
        grid.update_voxel(point, view2)
        grid.update_voxel(point, view2)

        grid.compute_quality(min_observations=3, min_view_diversity=30.0)
        # Should be GOOD or EXCELLENT now
        assert grid.quality_scores[1, 1, 1] >= ScanQuality.GOOD.value

    def test_coverage_stats(self):
        """Test coverage statistics computation"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([0.2, 0.2, 0.2])
        resolution = 0.1

        grid = VoxelGrid(bounds_min, bounds_max, resolution)

        # Add some observations
        points = [
            np.array([0.05, 0.05, 0.05]),
            np.array([0.15, 0.15, 0.15]),
        ]
        view = np.array([0, 0, -1])

        for point in points:
            grid.update_voxel(point, view)

        grid.compute_quality()
        stats = grid.get_coverage_stats()

        assert stats['total_voxels'] == 8  # 2x2x2
        assert stats['observed_voxels'] == 2
        assert stats['coverage_ratio'] == 0.25  # 2/8


class TestScanCoverageTracker:
    """Test scan coverage tracker"""

    def test_initialization(self):
        """Test tracker initialization"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([1, 1, 1])

        grid = VoxelGrid(bounds_min, bounds_max, 0.1)
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)

        tracker = ScanCoverageTracker(grid, camera_matrix)

        assert tracker.voxel_grid is not None
        assert tracker.min_observations == 3

    def test_update_from_points(self):
        """Test updating tracker from 3D points"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([1, 1, 1])

        grid = VoxelGrid(bounds_min, bounds_max, 0.1)
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)

        tracker = ScanCoverageTracker(grid, camera_matrix)

        # Create camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 2])  # 2m away

        # Create points
        points = np.array([
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
        ])

        tracker.update_from_points(points, camera_pose)

        # Check that voxels were updated
        assert grid.observation_count[5, 5, 5] > 0

    def test_is_scan_complete(self):
        """Test scan completion check"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([0.2, 0.2, 0.2])  # Small volume

        grid = VoxelGrid(bounds_min, bounds_max, 0.1)
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)

        tracker = ScanCoverageTracker(
            grid, camera_matrix,
            min_coverage_ratio=0.5,
            min_good_quality_ratio=0.3
        )

        # Initially incomplete
        tracker.compute_coverage_quality()
        is_complete, stats = tracker.is_scan_complete()
        assert not is_complete

        # Add sufficient observations
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 1])

        # Generate points covering the volume
        x = np.linspace(0.05, 0.15, 5)
        y = np.linspace(0.05, 0.15, 5)
        z = np.linspace(0.05, 0.15, 5)

        for xi in x:
            for yi in y:
                for zi in z:
                    points = np.array([[xi, yi, zi]])
                    for _ in range(5):  # Multiple observations
                        tracker.update_from_points(points, camera_pose)

        tracker.compute_coverage_quality()
        is_complete, stats = tracker.is_scan_complete()

        # Should have good coverage now
        assert stats['coverage_ratio'] > 0.5

    def test_rescan_regions(self):
        """Test rescan region detection"""
        bounds_min = np.array([0, 0, 0])
        bounds_max = np.array([0.3, 0.3, 0.3])

        grid = VoxelGrid(bounds_min, bounds_max, 0.1)
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)

        tracker = ScanCoverageTracker(grid, camera_matrix)

        # Add partial coverage
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 1])

        points = np.array([[0.15, 0.15, 0.15]])  # Only one point
        tracker.update_from_points(points, camera_pose)

        tracker.compute_coverage_quality()
        rescan_regions = tracker.get_rescan_regions()

        # Should have many regions needing rescan
        assert len(rescan_regions) > 0


class TestARFeedbackOverlay:
    """Test AR feedback overlay"""

    def test_initialization(self):
        """Test AR overlay initialization"""
        overlay = ARFeedbackOverlay((640, 480))
        assert overlay.width == 640
        assert overlay.height == 480

    def test_create_overlay(self):
        """Test overlay creation"""
        overlay = ARFeedbackOverlay((640, 480))

        # Create dummy heatmap
        heatmap = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create dummy stats
        stats = {
            'coverage_ratio': 0.75,
            'good_quality_ratio': 0.60,
            'needs_rescan_ratio': 0.25,
        }

        # Create overlay
        result = overlay.create_overlay(heatmap, stats)

        # Check output
        assert result.shape == (480, 640, 4)  # RGBA
        assert result.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
