#!/usr/bin/env python3
"""
Setup script for Glass Fracture Forensic System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Glass Fracture Forensic System - Production-grade deterministic fracture analysis"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "numpy>=1.21.0,<2.0.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
    ]

setup(
    name="glass-fracture-forensics",
    version="2.0.0",
    author="Forensic Engineering Team",
    author_email="daniel@absolicsinc.com",
    description="Production-grade deterministic fracture analysis for brittle isotropic glass",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keyboard-jangin/keyboard-jangin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "glass-forensics=glass_fracture_forensics.forensic_system:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
