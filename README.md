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

## Examples

See the `examples/` directory for complete usage examples:

- `basic_analysis.py`: Basic forensic analysis workflow
- More examples coming soon...

## Project Structure

```
keyboard-jangin/
├── src/
│   └── glass_fracture_forensics/
│       ├── __init__.py
│       └── forensic_system.py      # Main system implementation
├── tests/                          # Unit tests
├── examples/                       # Example scripts
│   └── basic_analysis.py
├── config/                         # Configuration files
│   └── default_config.yaml
├── docs/                           # Documentation
├── output/                         # Output directory (reports, viz)
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
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
