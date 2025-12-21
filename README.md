# Glass Fracture Forensic System

Production-grade deterministic fracture analysis for brittle isotropic glass. Suitable for industrial root-cause analysis, patent defense, and legal testimony.

## Overview

The Glass Fracture Forensic System is a **deterministic, physics-based** framework for analyzing glass fractures. It uses computer vision and linear elastic fracture mechanics (LEFM) to:

1. Reconstruct 3D fracture trajectories from 2D images
2. Estimate the fracture origin point with uncertainty quantification
3. Compute stress intensity factors
4. Classify failure modes (point impact, thermal shock, mechanical fatigue)

### Key Features

- **100% Deterministic**: NO machine learning, NO probabilistic classifiers
- **Mathematically Rigorous**: All equations explicitly stated and documented
- **Legally Defensible**: Every result can be reproduced with whiteboard + equations
- **Uncertainty Quantified**: 95% confidence ellipsoids for origin estimates
- **Production Ready**: Designed for expert testimony and forensic use

## Mathematical Foundation

The system is based on the following mathematical principles:

1. **Essential Matrix**: `x₂ᵀ E x₁ = 0`
2. **Origin Estimation**: `min_x Σᵢ ||(I − dᵢdᵢᵀ)(x − pᵢ)||²`
3. **Covariance**: `Σ = (Σᵢ (I − dᵢdᵢᵀ))⁻¹`
4. **Stress Intensity**:
   - `K_I = K₀·cos³(θ/2)`
   - `K_II = K₀·sin(θ/2)·cos²(θ/2)`
5. **95% Ellipsoid**: `(x − μ)ᵀ Σ⁻¹ (x − μ) ≤ χ²₍₃,₀.₉₅₎`

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- SciPy >= 1.7.0

### Install from source

```bash
# Clone the repository
git clone https://github.com/keyboard-jangin/keyboard-jangin.git
cd keyboard-jangin

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from glass_fracture_forensics import (
    GlassFractureForensicSystem,
    GlassMaterialProperties,
    SystemThresholds,
)
import numpy as np

# Initialize system
system = GlassFractureForensicSystem()

# Prepare your data
images = [...]  # List of grayscale images
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Camera intrinsics
masks = [...]   # Binary masks indicating fractures

# Run analysis
report = system.analyze(
    image_sequence=images,
    camera_matrix=K,
    fracture_masks=masks
)

# Access results
print(f"Origin: {report.origin.position}")
print(f"Confidence: {report.origin.confidence}")
print(f"Failure mode: {report.failure_mode.value}")

# Save report
system.save_report(report, output_dir="output")
```

## Real-Time Scan Feedback (NEW!)

The system now includes AR-guided real-time scan feedback for optimal capture quality:

```python
from glass_fracture_forensics import (
    ScanCoverageTracker,
    VoxelGrid,
    ARFeedbackOverlay,
)

# Define scan volume
scan_bounds = (np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 0.5]))

# Create voxel-based coverage tracker
voxel_grid = VoxelGrid(
    bounds_min=scan_bounds[0],
    bounds_max=scan_bounds[1],
    resolution=0.02  # 2cm voxels
)

tracker = ScanCoverageTracker(voxel_grid, camera_matrix)

# During AR capture loop:
for frame in capture_session:
    points_3d, camera_pose = process_frame(frame)

    # Update coverage
    tracker.update_from_points(points_3d, camera_pose)
    tracker.compute_coverage_quality()

    # Generate AR overlay
    heatmap = tracker.generate_heatmap_2d(camera_pose, image_size)
    is_complete, stats = tracker.is_scan_complete()

    # Display guidance to user
    if not is_complete:
        rescan_regions = tracker.get_rescan_regions()
        show_rescan_hints(rescan_regions)
```

**Visual Feedback:**
- 🔴 Red: Unscanned or poor quality - SCAN HERE
- 🟡 Yellow: Partial coverage - NEEDS MORE VIEWS
- 🟢 Green: Good coverage - WELL SCANNED
- 🔵 Blue: Excellent coverage - OPTIMAL

## Accuracy Enhancements (NEW!)

Version 2.1.0 introduces comprehensive accuracy improvements:

```python
from glass_fracture_forensics import (
    AccuracyEnhancedCaptureValidator,
    bootstrap_origin_estimation,
    generate_validation_report,
)

# Enhanced capture validation with accurate metrics
validator = AccuracyEnhancedCaptureValidator(thresholds, camera_matrix, image_size)
quality = validator.validate_tracks_accurate(tracks)

# Statistical validation with bootstrap
bootstrap_result = bootstrap_origin_estimation(
    trajectories, origin_estimator, n_bootstrap=1000
)

# Comprehensive validation report
validation_report = generate_validation_report(
    origin_estimate, trajectories, stress_factors,
    failure_mode, origin_estimator, mechanics_analyzer
)
```

**Improvements:**
- ✅ Accurate parallax computation from track motion
- ✅ Grid-based spatial coverage assessment
- ✅ Uncertainty propagation through pipeline
- ✅ Reprojection error validation
- ✅ Bootstrap confidence intervals
- ✅ Monte Carlo error propagation
- ✅ Outlier detection (Z-score, IQR, Mahalanobis)
- ✅ Statistical hypothesis testing
- ✅ Cross-validation for robustness

## Video Processing & Advanced Analysis (NEW in 2.2!)

Complete video-based capture and analysis pipeline:

```python
from glass_fracture_forensics import (
    VideoProcessor,
    FractureDetector,
    WaveformAnalyzer,
    ForensicVisualizer,
)

# Video capture and processing
processor = VideoProcessor(
    source=CaptureSource.CAMERA,
    target_fps=10
)

session = processor.capture_session(camera_matrix)

# Fracture waveform analysis
waveform_analyzer = WaveformAnalyzer()
waveform = waveform_analyzer.path_to_waveform(crack_path)
waveform.compute_fft()

# Advanced visualization
visualizer = ForensicVisualizer(dpi=150)
fig = visualizer.plot_3d_trajectories(trajectories, origin, covariance)
visualizer.create_summary_figure(report)
```

**New Capabilities:**
- 🎥 Real-time video processing and frame extraction
- 🔍 Automatic fracture detection and segmentation
- 📊 Waveform analysis with FFT
- 🌊 Crack pattern characterization (tortuosity, roughness)
- 📈 Publication-quality visualizations
- 🎨 3D rendering with uncertainty ellipsoids

## Examples

See the `examples/` directory for complete usage examples:

- `basic_analysis.py`: Basic forensic analysis workflow
- `realtime_scan_feedback.py`: AR-guided scan coverage demo with live quality visualization
- `accuracy_enhanced_analysis.py`: Complete pipeline with accuracy enhancements and statistical validation

## Project Structure

```
keyboard-jangin/
├── src/
│   └── glass_fracture_forensics/
│       ├── __init__.py
│       ├── forensic_system.py                 # Main forensic pipeline
│       ├── realtime_feedback.py               # AR scan feedback system
│       ├── accuracy_improvements.py           # Accuracy enhancements
│       ├── statistical_validation.py          # Statistical validation
│       ├── video_processing.py                # Video capture & processing
│       ├── fracture_waveform_analysis.py      # Waveform analysis
│       └── visualization_engine.py            # Advanced visualization
├── tests/                                      # Unit tests
│   ├── test_forensic_system.py
│   └── test_realtime_feedback.py
├── examples/                                   # Example scripts
│   ├── basic_analysis.py
│   ├── realtime_scan_feedback.py
│   └── accuracy_enhanced_analysis.py
├── config/                                     # Configuration files
│   └── default_config.yaml
├── docs/                                       # Documentation
│   └── improvement_analysis.md
├── output/                                     # Output directory (reports, viz)
├── requirements.txt                            # Python dependencies
├── setup.py                                    # Package setup
└── README.md                                   # This file
```

## Pipeline

The analysis pipeline consists of 8 steps:

1. **Feature Tracking**: KLT optical flow with Forward-Backward validation
2. **Capture Validation**: Verify sufficient parallax and coverage
3. **3D Reconstruction**: Essential matrix estimation and triangulation
4. **Trajectory Fitting**: PCA-based line fitting
5. **Origin Estimation**: Multi-trajectory intersection with uncertainty
6. **Fracture Mechanics**: LEFM-based stress intensity analysis
7. **Classification**: Deterministic failure mode classification
8. **Report Generation**: Immutable evidence with SHA-256 hash

## Configuration

All system parameters are configurable via YAML files or direct API:

```python
# Material properties (soda-lime glass)
material = GlassMaterialProperties(
    E=72.0e9,           # Young's Modulus [Pa]
    nu=0.23,            # Poisson's Ratio
    K_Ic=0.75e6,        # Fracture Toughness [Pa·√m]
    rho=2500.0,         # Density [kg/m³]
)

# System thresholds
thresholds = SystemThresholds(
    min_parallax_angle=5.0,
    ransac_confidence=0.999,
    # ... see config/default_config.yaml for all options
)

# Initialize with custom config
system = GlassFractureForensicSystem(
    material=material,
    thresholds=thresholds
)
```

## Validation

The system includes built-in validation to ensure all results are reproducible:

```python
from glass_fracture_forensics.forensic_system import validate_system

# Run validation
is_valid = validate_system()
```

**Validation Checklist:**
- Essential Matrix equation stated ✓
- Origin estimation equation stated ✓
- Covariance equation stated ✓
- Stress intensity equations stated ✓
- Confidence ellipsoid equation stated ✓
- All constants physically justified ✓
- No machine learning ✓
- No probabilistic classifiers ✓
- Deterministic only ✓
- Assumptions explicitly listed ✓

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src/glass_fracture_forensics
```

### Code Formatting

```bash
black src/ tests/ examples/
flake8 src/ tests/ examples/
```

### Type Checking

```bash
mypy src/glass_fracture_forensics
```

## References

- Anderson, T.L. (2017). *Fracture Mechanics: Fundamentals and Applications*, 4th Ed.
- ASTM C1036: Standard Specification for Flat Glass
- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*

## License

Proprietary - For Expert Testimony Use

## Contact

**Daniel**
Email: daniel@absolicsinc.com

## Contributing

This is a specialized forensic system. For collaboration inquiries, please contact the author.

## Acknowledgments

Developed by the Forensic Engineering Team for production-grade glass fracture analysis.

---

**Note**: This system is designed for forensic applications where deterministic, reproducible results are critical. All results can be defended in legal proceedings using only the stated equations and recorded data.
