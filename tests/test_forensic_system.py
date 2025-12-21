#!/usr/bin/env python3
"""
Unit tests for the main forensic system
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glass_fracture_forensics import (
    GlassFractureForensicSystem,
    GlassMaterialProperties,
    SystemThresholds,
    Trajectory3D,
    OriginEstimate,
    FailureMode,
)


class TestGlassMaterialProperties:
    """Test material properties validation"""

    def test_default_properties(self):
        """Test default soda-lime glass properties"""
        mat = GlassMaterialProperties()
        assert mat.E == 72.0e9
        assert mat.nu == 0.23
        assert mat.K_Ic == 0.75e6
        assert mat.validate()

    def test_invalid_youngs_modulus(self):
        """Test validation fails for invalid Young's modulus"""
        with pytest.raises(AssertionError):
            mat = GlassMaterialProperties(E=-1.0)
            mat.validate()

    def test_invalid_poisson_ratio(self):
        """Test validation fails for invalid Poisson's ratio"""
        with pytest.raises(AssertionError):
            mat = GlassMaterialProperties(nu=0.6)
            mat.validate()


class TestTrajectory3D:
    """Test 3D trajectory fitting"""

    def test_fit_line(self):
        """Test fitting a line to 3D points"""
        # Create points along a line
        t = np.linspace(0, 1, 10)
        direction = np.array([1, 0, 0])
        origin = np.array([0, 0, 0])
        points = origin + np.outer(t, direction)

        traj = Trajectory3D(
            points=points,
            origin=np.zeros(3),
            direction=np.zeros(3)
        )
        traj.fit()

        # Check direction is unit vector
        assert np.isclose(np.linalg.norm(traj.direction), 1.0)

        # Check direction matches (allow sign flip)
        assert np.isclose(np.abs(np.dot(traj.direction, direction)), 1.0)

    def test_insufficient_points(self):
        """Test error with insufficient points"""
        points = np.array([[0, 0, 0]])
        traj = Trajectory3D(
            points=points,
            origin=np.zeros(3),
            direction=np.zeros(3)
        )
        with pytest.raises(ValueError):
            traj.fit()


class TestOriginEstimate:
    """Test origin estimation"""

    def test_ellipsoid_computation(self):
        """Test confidence ellipsoid computation"""
        origin = OriginEstimate(
            position=np.zeros(3),
            covariance=np.eye(3),
            confidence=0.95
        )

        ellipsoid = origin.compute_ellipsoid()

        assert 'center' in ellipsoid
        assert 'radii' in ellipsoid
        assert 'volume' in ellipsoid
        assert len(ellipsoid['radii']) == 3


class TestGlassFractureForensicSystem:
    """Test main forensic system"""

    def test_initialization(self):
        """Test system initialization"""
        system = GlassFractureForensicSystem()
        assert system.material is not None
        assert system.thresholds is not None

    def test_custom_configuration(self):
        """Test initialization with custom config"""
        material = GlassMaterialProperties(E=70.0e9)
        thresholds = SystemThresholds(min_parallax_angle=10.0)

        system = GlassFractureForensicSystem(
            material=material,
            thresholds=thresholds
        )

        assert system.material.E == 70.0e9
        assert system.thresholds.min_parallax_angle == 10.0

    def test_analysis_pipeline(self):
        """Test full analysis pipeline with dummy data"""
        system = GlassFractureForensicSystem()

        # Create dummy data
        images = [
            np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            for _ in range(5)
        ]
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        masks = [np.zeros((480, 640), dtype=np.uint8) for _ in range(5)]

        # Run analysis
        report = system.analyze(images, K, masks)

        # Check report structure
        assert report.origin is not None
        assert report.failure_mode in FailureMode
        assert len(report.trajectories) > 0
        assert report.evidence_hash != ""

    def test_report_generation(self):
        """Test forensic report generation"""
        system = GlassFractureForensicSystem()

        images = [np.random.randint(0, 255, (480, 640), dtype=np.uint8) for _ in range(5)]
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        masks = [np.zeros((480, 640), dtype=np.uint8) for _ in range(5)]

        report = system.analyze(images, K, masks)

        # Check hash generation
        hash1 = report.generate_hash()
        hash2 = report.generate_hash()
        assert hash1 == hash2  # Should be deterministic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
