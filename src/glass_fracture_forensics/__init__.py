"""
Glass Fracture Forensic System
================================

Production-grade deterministic fracture analysis for brittle isotropic glass.

Author: Forensic Engineering Team
Version: 2.0
License: Proprietary - For Expert Testimony Use
"""

from .forensic_system import (
    GlassFractureForensicSystem,
    ForensicReport,
    FailureMode,
    GlassMaterialProperties,
    SystemThresholds,
    OriginEstimate,
    Trajectory3D,
    StressIntensityFactors,
)

__version__ = "2.0.0"
__all__ = [
    "GlassFractureForensicSystem",
    "ForensicReport",
    "FailureMode",
    "GlassMaterialProperties",
    "SystemThresholds",
    "OriginEstimate",
    "Trajectory3D",
    "StressIntensityFactors",
]
