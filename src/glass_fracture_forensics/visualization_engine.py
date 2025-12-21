#!/usr/bin/env python3
"""
VISUALIZATION ENGINE
====================

Advanced visualization and rendering for forensic analysis results.

FEATURES:
- 3D trajectory visualization
- Origin uncertainty ellipsoid
- Heatmap generation
- AR overlay composition
- Publication-quality plots
- Interactive 3D views
- Forensic report figures

Author: Forensic Engineering Team
Version: 2.1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import cv2
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class ForensicVisualizer:
    """
    FORENSIC VISUALIZATION ENGINE

    Creates publication-quality figures for forensic reports.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', dpi: int = 150):
        self.style = style
        self.dpi = dpi

        # Apply style
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available

    def plot_3d_trajectories(self,
                            trajectories: List,
                            origin_position: np.ndarray,
                            origin_covariance: np.ndarray,
                            title: str = "3D Fracture Trajectories",
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Visualize 3D trajectories with origin uncertainty

        Args:
            trajectories: List of Trajectory3D objects
            origin_position: (3,) origin coordinates
            origin_covariance: (3, 3) covariance matrix
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))

        for i, traj in enumerate(trajectories):
            ax.plot(traj.points[:, 0],
                   traj.points[:, 1],
                   traj.points[:, 2],
                   color=colors[i],
                   linewidth=2,
                   label=f'Trajectory {i+1}')

            # Plot trajectory direction
            arrow = Arrow3D(
                [traj.origin[0], traj.origin[0] + traj.direction[0] * 0.1],
                [traj.origin[1], traj.origin[1] + traj.direction[1] * 0.1],
                [traj.origin[2], traj.origin[2] + traj.direction[2] * 0.1],
                mutation_scale=20,
                lw=2,
                arrowstyle='-|>',
                color=colors[i],
                alpha=0.6
            )
            ax.add_artist(arrow)

        # Plot origin
        ax.scatter(*origin_position, color='red', s=200, marker='*',
                  label='Estimated Origin', zorder=10)

        # Plot uncertainty ellipsoid
        self._plot_3d_ellipsoid(ax, origin_position, origin_covariance)

        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.set_zlabel('Z [m]', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def _plot_3d_ellipsoid(self, ax, center, covariance, n_std=2.447):
        """
        Plot 3D uncertainty ellipsoid (95% confidence)

        n_std = 2.447 for 95% confidence in 3D (chi-square with 3 DOF)
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Ellipsoid radii
        radii = n_std * np.sqrt(eigenvalues)

        # Generate ellipsoid surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)

        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Rotate by eigenvectors
        for i in range(len(x)):
            for j in range(len(x[0])):
                point = np.array([x[i, j], y[i, j], z[i, j]])
                rotated = eigenvectors @ point
                x[i, j], y[i, j], z[i, j] = rotated

        # Translate to center
        x += center[0]
        y += center[1]
        z += center[2]

        # Plot surface
        ax.plot_surface(x, y, z, alpha=0.2, color='red', label='95% CI Ellipsoid')

    def plot_coverage_heatmap(self,
                             coverage_map: np.ndarray,
                             title: str = "Spatial Coverage Heatmap",
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot spatial coverage heatmap

        Args:
            coverage_map: (H, W) coverage grid
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)

        im = ax.imshow(coverage_map, cmap='YlOrRd', interpolation='nearest')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Observations', fontsize=12)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid Column', fontsize=12)
        ax.set_ylabel('Grid Row', fontsize=12)

        # Grid lines
        ax.set_xticks(np.arange(coverage_map.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(coverage_map.shape[0]) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_stress_intensity(self,
                             stress_factors: List,
                             material_K_Ic: float,
                             title: str = "Stress Intensity Factors",
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot stress intensity factors

        Args:
            stress_factors: List of StressIntensityFactors
            material_K_Ic: Material fracture toughness
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)

        indices = np.arange(len(stress_factors))
        K_I_values = np.array([sf.K_I for sf in stress_factors]) / 1e6  # MPa·√m
        K_II_values = np.array([sf.K_II for sf in stress_factors]) / 1e6
        angles = np.array([np.degrees(sf.theta) for sf in stress_factors])

        # K_I and K_II bar chart
        width = 0.35
        ax1.bar(indices - width/2, K_I_values, width, label='K_I (Mode I)', color='royalblue')
        ax1.bar(indices + width/2, K_II_values, width, label='K_II (Mode II)', color='darkorange')

        ax1.axhline(material_K_Ic / 1e6, color='red', linestyle='--',
                   linewidth=2, label=f'K_Ic = {material_K_Ic/1e6:.2f}')

        ax1.set_xlabel('Trajectory Index', fontsize=12)
        ax1.set_ylabel('Stress Intensity [MPa·√m]', fontsize=12)
        ax1.set_title('Stress Intensity Factors', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Branching angles
        ax2.bar(indices, angles, color='green', alpha=0.7)
        ax2.set_xlabel('Trajectory Index', fontsize=12)
        ax2.set_ylabel('Branching Angle [degrees]', fontsize=12)
        ax2.set_title('Branching Angles from Mode I', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_waveform_analysis(self,
                              waveform,
                              title: str = "Fracture Waveform Analysis",
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot waveform analysis results

        Args:
            waveform: FractureWaveform object
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # 1. Amplitude waveform
        ax1.plot(waveform.arc_length, waveform.amplitude, 'b-', linewidth=1.5)
        ax1.set_xlabel('Arc Length [mm]', fontsize=11)
        ax1.set_ylabel('Amplitude [mm]', fontsize=11)
        ax1.set_title('Crack Deviation from Straight Line', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Power spectrum
        if waveform.power_spectrum is not None:
            ax2.semilogy(waveform.frequencies, waveform.power_spectrum, 'r-', linewidth=1.5)
            ax2.axvline(waveform.dominant_frequency, color='green', linestyle='--',
                       linewidth=2, label=f'Dominant: {waveform.dominant_frequency:.3f}')
            ax2.set_xlabel('Frequency [1/mm]', fontsize=11)
            ax2.set_ylabel('Power', fontsize=11)
            ax2.set_title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # 3. Curvature
        ax3.plot(waveform.arc_length, waveform.curvature, 'g-', linewidth=1.5)
        ax3.set_xlabel('Arc Length [mm]', fontsize=11)
        ax3.set_ylabel('Curvature [1/mm]', fontsize=11)
        ax3.set_title('Local Curvature Along Crack', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Tangent angle
        ax4.plot(waveform.arc_length, np.degrees(waveform.angle), 'm-', linewidth=1.5)
        ax4.set_xlabel('Arc Length [mm]', fontsize=11)
        ax4.set_ylabel('Tangent Angle [degrees]', fontsize=11)
        ax4.set_title('Crack Propagation Direction', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def create_summary_figure(self,
                             report,
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive summary figure

        Args:
            report: ForensicReport object
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)

        # Title
        fig.suptitle('Glass Fracture Forensic Analysis Summary',
                    fontsize=16, fontweight='bold')

        # 1. 3D Trajectories (top left, large)
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')

        colors = plt.cm.rainbow(np.linspace(0, 1, len(report.trajectories)))
        for i, traj in enumerate(report.trajectories):
            ax1.plot(traj.points[:, 0], traj.points[:, 1], traj.points[:, 2],
                    color=colors[i], linewidth=2, label=f'Traj {i+1}')

        ax1.scatter(*report.origin.position, color='red', s=200, marker='*',
                   label='Origin', zorder=10)

        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title('3D Reconstruction', fontweight='bold')
        ax1.legend(fontsize=8)

        # 2. Origin uncertainty (top right)
        ax2 = fig.add_subplot(2, 3, 2)

        eigenvalues, _ = np.linalg.eigh(report.origin.covariance)
        radii_mm = np.sqrt(7.815 * eigenvalues) * 1000  # 95% CI in mm

        ax2.bar(['X', 'Y', 'Z'], radii_mm, color=['red', 'green', 'blue'], alpha=0.7)
        ax2.set_ylabel('95% CI Radius [mm]')
        ax2.set_title('Origin Uncertainty', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Stress intensity (top right 2)
        ax3 = fig.add_subplot(2, 3, 3)

        mean_K_I = np.mean([sf.K_I for sf in report.stress_factors]) / 1e6
        mean_K_II = np.mean([sf.K_II for sf in report.stress_factors]) / 1e6

        ax3.bar(['K_I', 'K_II'], [mean_K_I, mean_K_II],
               color=['royalblue', 'darkorange'], alpha=0.7)
        ax3.set_ylabel('Stress Intensity [MPa·√m]')
        ax3.set_title('Mean Stress Intensities', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Quality metrics (bottom left)
        ax4 = fig.add_subplot(2, 3, 5)

        quality = report.capture_quality
        metrics = ['Coverage', 'Parallax\n(norm)', 'Quality\nScore']
        values = [
            quality.get('coverage_fraction', 0),
            min(1.0, quality.get('mean_parallax_degrees', 0) / 20.0),
            quality.get('quality_score', 0)
        ]

        bars = ax4.barh(metrics, values, color=['green', 'blue', 'purple'], alpha=0.7)
        ax4.set_xlim(0, 1)
        ax4.set_xlabel('Score [0-1]')
        ax4.set_title('Capture Quality', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # Color bars based on value
        for bar, val in zip(bars, values):
            if val >= 0.8:
                bar.set_color('green')
            elif val >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # 5. Classification (bottom middle)
        ax5 = fig.add_subplot(2, 3, 6)
        ax5.axis('off')

        # Text summary
        summary_text = f"""
FAILURE MODE:
{report.failure_mode.value}

ORIGIN POSITION:
X: {report.origin.position[0]:.4f} m
Y: {report.origin.position[1]:.4f} m
Z: {report.origin.position[2]:.4f} m

CONFIDENCE:
{report.origin.confidence:.3f}

TRAJECTORIES:
{len(report.trajectories)} detected

EVIDENCE HASH:
{report.evidence_hash[:16]}...
        """

        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demo visualization"""

    print("="*70)
    print("VISUALIZATION ENGINE - DEMO")
    print("="*70)

    # Create dummy data
    from glass_fracture_forensics import Trajectory3D, OriginEstimate

    # Trajectories
    trajectories = []
    for i in range(3):
        angle = i * 2 * np.pi / 3
        t = np.linspace(0, 0.3, 20)
        points = np.column_stack([
            t * np.cos(angle),
            t * np.sin(angle),
            np.ones(20) * 0.1
        ])

        traj = Trajectory3D(
            points=points,
            origin=np.zeros(3),
            direction=np.array([np.cos(angle), np.sin(angle), 0])
        )
        trajectories.append(traj)

    # Origin
    origin = OriginEstimate(
        position=np.array([0.01, -0.01, 0.12]),
        covariance=np.diag([0.001, 0.001, 0.002]) ** 2,
        confidence=0.92
    )

    # Visualize
    visualizer = ForensicVisualizer(dpi=100)

    print("\nGenerating 3D visualization...")
    fig = visualizer.plot_3d_trajectories(
        trajectories,
        origin.position,
        origin.covariance,
        title="Demo: 3D Fracture Trajectories"
    )
    plt.show()

    print("\n" + "="*70)
    print("Visualization engine ready")
    print("="*70)
