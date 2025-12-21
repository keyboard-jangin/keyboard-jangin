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

from .realtime_feedback import (
    ScanCoverageTracker,
    VoxelGrid,
    ScanQuality,
    ARFeedbackOverlay,
    integrate_realtime_feedback,
)

from .accuracy_improvements import (
    compute_mean_parallax,
    compute_spatial_coverage,
    compute_reprojection_error,
    propagate_triangulation_uncertainty,
    AccuracyEnhancedCaptureValidator,
    SpatialCoverageGrid,
)

from .statistical_validation import (
    bootstrap_origin_estimation,
    generate_validation_report,
    robust_outlier_detection,
    OutlierMethod,
    BootstrapResult,
)

from .video_processing import (
    VideoProcessor,
    CaptureSession,
    VideoFrame,
    FractureDetector,
    FrameSelector,
)

from .fracture_waveform_analysis import (
    CrackPath,
    FractureWaveform,
    WaveformAnalyzer,
    analyze_fracture_from_mask,
)

from .visualization_engine import (
    ForensicVisualizer,
)

__version__ = "2.2.0"
__all__ = [
    # Core forensic system
    "GlassFractureForensicSystem",
    "ForensicReport",
    "FailureMode",
    "GlassMaterialProperties",
    "SystemThresholds",
    "OriginEstimate",
    "Trajectory3D",
    "StressIntensityFactors",
    # Real-time feedback
    "ScanCoverageTracker",
    "VoxelGrid",
    "ScanQuality",
    "ARFeedbackOverlay",
    "integrate_realtime_feedback",
    # Accuracy improvements
    "compute_mean_parallax",
    "compute_spatial_coverage",
    "compute_reprojection_error",
    "propagate_triangulation_uncertainty",
    "AccuracyEnhancedCaptureValidator",
    "SpatialCoverageGrid",
    # Statistical validation
    "bootstrap_origin_estimation",
    "generate_validation_report",
    "robust_outlier_detection",
    "OutlierMethod",
    "BootstrapResult",
    # Video processing
    "VideoProcessor",
    "CaptureSession",
    "VideoFrame",
    "FractureDetector",
    "FrameSelector",
    # Waveform analysis
    "CrackPath",
    "FractureWaveform",
    "WaveformAnalyzer",
    "analyze_fracture_from_mask",
    # Visualization
    "ForensicVisualizer",
]
