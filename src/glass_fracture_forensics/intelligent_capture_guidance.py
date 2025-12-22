#!/usr/bin/env python3
"""
INTELLIGENT CAPTURE GUIDANCE SYSTEM
====================================

Real-time suggestions for optimal lighting and camera angles during video capture.
Based on optical physics and glass material properties.

FEATURES:
1. Wavelength-specific lighting recommendations
2. Angle optimization (Brewster angle, glancing angle)
3. Polarization guidance for stress visualization
4. Real-time quality assessment with specific feedback

PHYSICAL PRINCIPLES:
1. Brewster's Angle: θ_B = arctan(n₂/n₁) for minimum reflection
2. Fresnel Equations: Reflection/transmission at interfaces
3. Photoelasticity: Stress-induced birefringence in glass
4. Raking Light: Grazing incidence for surface detail

REFERENCES:
- Hecht, E. (2017): Optics (5th ed.)
- Born & Wolf (1999): Principles of Optics
- Dally & Riley (1991): Experimental Stress Analysis

Author: Forensic Engineering Team
Version: 2.3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class LightingMode(Enum):
    """Lighting modes for glass fracture analysis"""
    DIFFUSE = "diffuse"              # General illumination
    RAKING = "raking"                # Low-angle surface detail
    POLARIZED = "polarized"          # Stress visualization
    CROSSED_POLARIZED = "crossed"    # Maximum stress contrast
    OBLIQUE = "oblique"              # 45° oblique lighting
    TRANSMITTED = "transmitted"      # Backlit (if accessible)


@dataclass
class OpticalProperties:
    """
    Optical properties of glass

    REFERENCE: ASTM C1036-16, Hecht (2017)
    """
    refractive_index: float = 1.52      # Soda-lime glass (n_D at 589nm)
    abbe_number: float = 58.5           # Dispersion parameter
    stress_optic_coefficient: float = 2.77e-12  # C [Pa⁻¹]

    def wavelength_dependent_n(self, wavelength_nm: float) -> float:
        """
        Wavelength-dependent refractive index

        EQUATION (Sellmeier):
        n²(λ) = 1 + Σᵢ (Bᵢλ²)/(λ² - Cᵢ)

        Simplified for soda-lime glass
        """
        # Simplified Cauchy equation
        lambda_um = wavelength_nm / 1000.0
        n = self.refractive_index + 0.003 / lambda_um**2
        return n

    def compute_brewster_angle(self, n_external: float = 1.0) -> float:
        """
        Compute Brewster's angle for minimum reflection

        EQUATION:
        θ_B = arctan(n₂/n₁)

        At Brewster's angle, p-polarized light has zero reflection

        Args:
            n_external: Refractive index of external medium (1.0 for air)

        Returns:
            θ_B in degrees
        """
        theta_b_rad = np.arctan(self.refractive_index / n_external)
        return np.degrees(theta_b_rad)

    def fresnel_reflectance(self,
                           incident_angle_deg: float,
                           polarization: str = "unpolarized") -> float:
        """
        Compute Fresnel reflectance at air-glass interface

        EQUATIONS:
        R_s = |sin(θ_i - θ_t)/sin(θ_i + θ_t)|²
        R_p = |tan(θ_i - θ_t)/tan(θ_i + θ_t)|²
        R_unpolarized = (R_s + R_p)/2

        Args:
            incident_angle_deg: Incident angle [degrees]
            polarization: 's', 'p', or 'unpolarized'

        Returns:
            Reflectance [0, 1]
        """
        theta_i = np.radians(incident_angle_deg)
        n1, n2 = 1.0, self.refractive_index

        # Snell's law for transmitted angle
        sin_theta_t = (n1 / n2) * np.sin(theta_i)

        if sin_theta_t > 1.0:
            return 1.0  # Total internal reflection (not typical for air→glass)

        theta_t = np.arcsin(sin_theta_t)

        # Fresnel equations
        if polarization == 's':
            numerator = np.sin(theta_i - theta_t)
            denominator = np.sin(theta_i + theta_t)
            if abs(denominator) < 1e-10:
                return 0.0
            R = (numerator / denominator) ** 2
        elif polarization == 'p':
            numerator = np.tan(theta_i - theta_t)
            denominator = np.tan(theta_i + theta_t)
            if abs(denominator) < 1e-10:
                return 0.0
            R = (numerator / denominator) ** 2
        else:  # unpolarized
            R_s = self.fresnel_reflectance(incident_angle_deg, 's')
            R_p = self.fresnel_reflectance(incident_angle_deg, 'p')
            R = (R_s + R_p) / 2

        return R


@dataclass
class CaptureRecommendation:
    """
    Single capture recommendation with rationale
    """
    priority: int           # 1=critical, 2=important, 3=optional
    category: str           # 'lighting', 'angle', 'polarization', 'wavelength'
    recommendation: str     # What to do
    rationale: str          # Why (physics-based)
    current_value: Optional[float] = None
    optimal_value: Optional[float] = None

    def __str__(self) -> str:
        priority_symbols = {1: "🔴 CRITICAL", 2: "🟡 IMPORTANT", 3: "⚪ OPTIONAL"}

        msg = f"{priority_symbols.get(self.priority, '•')} [{self.category.upper()}]\n"
        msg += f"  → {self.recommendation}\n"
        msg += f"  ℹ {self.rationale}"

        if self.current_value is not None and self.optimal_value is not None:
            msg += f"\n  Current: {self.current_value:.1f}° → Optimal: {self.optimal_value:.1f}°"

        return msg


class IntelligentCaptureGuidance:
    """
    Real-time capture guidance based on optical physics

    Provides specific recommendations for:
    - Camera angles (minimize reflection)
    - Lighting setup (enhance fracture features)
    - Wavelength selection (stress analysis)
    - Polarization (stress visualization)
    """

    def __init__(self):
        self.optical_props = OpticalProperties()
        self.recommendations: List[CaptureRecommendation] = []

    def analyze_camera_angle(self,
                            current_angle: float,
                            surface_normal: np.ndarray = np.array([0, 0, 1])) -> List[CaptureRecommendation]:
        """
        Analyze camera angle and provide recommendations

        Args:
            current_angle: Current incident angle [degrees]
            surface_normal: Surface normal vector

        Returns:
            List of recommendations
        """
        recommendations = []

        # Brewster's angle for minimum reflection
        brewster_angle = self.optical_props.compute_brewster_angle()

        # Normal incidence reflectance
        R_normal = self.optical_props.fresnel_reflectance(0)
        R_current = self.optical_props.fresnel_reflectance(current_angle)

        # Check if near Brewster's angle (for polarized light)
        if abs(current_angle - brewster_angle) < 5:
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="angle",
                recommendation=f"Use POLARIZED lighting at current angle ({current_angle:.1f}°)",
                rationale=f"Near Brewster's angle ({brewster_angle:.1f}°). "
                         f"P-polarized light will have minimal reflection, revealing surface detail.",
                current_value=current_angle,
                optimal_value=brewster_angle
            ))

        # Check if angle too steep (high reflection)
        if R_current > 0.15:  # >15% reflection
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="angle",
                recommendation=f"Reduce camera angle to 30-45° from surface",
                rationale=f"Current angle ({current_angle:.1f}°) gives {R_current*100:.1f}% reflection. "
                         f"Lower angles reduce glare and improve feature visibility.",
                current_value=current_angle,
                optimal_value=40.0
            ))

        # Check if too shallow for good parallax
        if current_angle < 15:
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="angle",
                recommendation="Increase camera elevation to 20-30°",
                rationale=f"Angle too shallow ({current_angle:.1f}°). "
                         f"Insufficient parallax for accurate 3D reconstruction.",
                current_value=current_angle,
                optimal_value=25.0
            ))

        return recommendations

    def recommend_lighting_mode(self,
                               fracture_characteristics: Dict[str, any]) -> List[CaptureRecommendation]:
        """
        Recommend optimal lighting mode based on fracture characteristics

        Args:
            fracture_characteristics: Dict with keys:
                - 'surface_roughness': 'smooth' | 'rough'
                - 'crack_width': float [mm]
                - 'stress_visible': bool
                - 'transparency': 'clear' | 'frosted'
        """
        recommendations = []

        surface = fracture_characteristics.get('surface_roughness', 'smooth')
        crack_width = fracture_characteristics.get('crack_width', 1.0)
        stress_visible = fracture_characteristics.get('stress_visible', False)
        transparency = fracture_characteristics.get('transparency', 'clear')

        # Fine cracks require raking light
        if crack_width < 0.5:  # < 0.5mm
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="lighting",
                recommendation="Use RAKING LIGHT (10-15° grazing angle)",
                rationale=f"Crack width {crack_width:.2f}mm requires low-angle illumination. "
                         f"Raking light creates shadows that reveal fine surface features.",
                optimal_value=12.0
            ))

        # Stress visualization
        if not stress_visible and transparency == 'clear':
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="polarization",
                recommendation="Enable CROSSED POLARIZERS for stress visualization",
                rationale="Glass stress creates birefringence (Δn ≈ C·σ). "
                         "Crossed polarizers reveal stress patterns as colorful fringes.",
            ))

        # Rough surfaces need diffuse lighting
        if surface == 'rough':
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="lighting",
                recommendation="Use DIFFUSE LIGHTING (dome or softbox)",
                rationale="Rough surfaces scatter light. Diffuse illumination prevents hotspots "
                         "and provides even coverage for feature tracking."
            ))

        # Transparent glass benefits from transmitted light
        if transparency == 'clear':
            recommendations.append(CaptureRecommendation(
                priority=3,
                category="lighting",
                recommendation="Add TRANSMITTED BACKLIGHT (if accessible)",
                rationale="Transmitted light enhances internal features: stress patterns, "
                         "crack branching, and inclusions. Provides complementary information."
            ))

        return recommendations

    def recommend_wavelength(self,
                            analysis_goal: str = "general") -> List[CaptureRecommendation]:
        """
        Recommend optimal wavelength(s) for specific analysis goals

        Args:
            analysis_goal: 'general', 'stress', 'surface', 'depth'

        Returns:
            Wavelength recommendations
        """
        recommendations = []

        if analysis_goal == "stress":
            # Photoelastic stress analysis
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="wavelength",
                recommendation="Use MONOCHROMATIC light (589nm sodium-D line)",
                rationale="Monochromatic light produces sharp isochromatic fringes in "
                         "photoelastic stress analysis. White light creates overlapping "
                         "orders that obscure stress magnitude.",
                optimal_value=589.0  # nm
            ))

            recommendations.append(CaptureRecommendation(
                priority=2,
                category="wavelength",
                recommendation="Alternative: 532nm (green laser pointer)",
                rationale="Green wavelength (532nm) maximizes human eye sensitivity and "
                         "provides good contrast in stress patterns.",
                optimal_value=532.0
            ))

        elif analysis_goal == "surface":
            # Surface detail visualization
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="wavelength",
                recommendation="Use SHORT WAVELENGTH (400-450nm blue LED)",
                rationale="Shorter wavelengths (λ ≈ 420nm) provide better resolution "
                         "for fine surface features (Rayleigh criterion: Δr ∝ λ).",
                optimal_value=420.0
            ))

        elif analysis_goal == "depth":
            # Depth penetration
            recommendations.append(CaptureRecommendation(
                priority=3,
                category="wavelength",
                recommendation="Use LONG WAVELENGTH (600-700nm red LED)",
                rationale="Longer wavelengths penetrate deeper with less scattering. "
                         "Useful for analyzing internal fracture patterns.",
                optimal_value=650.0
            ))

        else:  # general
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="wavelength",
                recommendation="Use DAYLIGHT (5500-6500K CCT) or white LED",
                rationale="Daylight-balanced illumination provides natural color rendering "
                         "and covers full visible spectrum for general analysis.",
                optimal_value=5500.0  # Kelvin
            ))

        return recommendations

    def assess_capture_quality_realtime(self,
                                       frame: np.ndarray,
                                       camera_angle: float,
                                       lighting_info: Dict) -> Tuple[float, List[CaptureRecommendation]]:
        """
        Real-time quality assessment with specific feedback

        Args:
            frame: Current video frame (RGB)
            camera_angle: Current camera angle [degrees]
            lighting_info: Dict with 'mode', 'intensity', 'angle'

        Returns:
            (quality_score, recommendations)
        """
        recommendations = []

        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = 0.299*frame[:,:,0] + 0.587*frame[:,:,1] + 0.114*frame[:,:,2]
        else:
            gray = frame

        # 1. Exposure assessment
        mean_intensity = np.mean(gray)
        if mean_intensity < 60:  # Underexposed
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="lighting",
                recommendation="INCREASE LIGHTING INTENSITY by 50-100%",
                rationale=f"Image underexposed (mean={mean_intensity:.1f}/255). "
                         f"Feature tracking requires intensity 80-180 for reliability.",
                current_value=mean_intensity,
                optimal_value=120.0
            ))
        elif mean_intensity > 200:  # Overexposed
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="lighting",
                recommendation="REDUCE LIGHTING INTENSITY or use neutral density filter",
                rationale=f"Image overexposed (mean={mean_intensity:.1f}/255). "
                         f"Clipping loses detail in bright regions (cracks, reflections).",
                current_value=mean_intensity,
                optimal_value=120.0
            ))

        # 2. Contrast assessment
        std_intensity = np.std(gray)
        if std_intensity < 30:  # Low contrast
            recommendations.append(CaptureRecommendation(
                priority=2,
                category="lighting",
                recommendation="INCREASE CONTRAST: Use directional/raking light",
                rationale=f"Low contrast (σ={std_intensity:.1f}). Diffuse lighting may be "
                         f"too even. Add oblique lighting to create shadows that reveal features.",
                current_value=std_intensity,
                optimal_value=50.0
            ))

        # 3. Reflection/glare detection
        saturated_pixels = np.sum(gray > 250)
        total_pixels = gray.size
        saturation_ratio = saturated_pixels / total_pixels

        if saturation_ratio > 0.05:  # >5% saturated
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="angle",
                recommendation="REDUCE GLARE: Adjust camera angle or use polarizer",
                rationale=f"Excessive glare ({saturation_ratio*100:.1f}% saturated pixels). "
                         f"Specular reflection obscures fracture features. "
                         f"Try angle = {self.optical_props.compute_brewster_angle():.1f}° with polarizer.",
                optimal_value=self.optical_props.compute_brewster_angle()
            ))

        # 4. Sharpness/blur assessment
        laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        from scipy.ndimage import convolve
        edges = convolve(gray, laplacian)
        sharpness = np.var(edges)

        if sharpness < 100:  # Blurry
            recommendations.append(CaptureRecommendation(
                priority=1,
                category="motion",
                recommendation="REDUCE MOTION: Move camera slower (< 10 cm/s)",
                rationale=f"Image blurry (sharpness={sharpness:.1f}). "
                         f"Motion blur degrades feature tracking. Slow, steady movements required.",
                current_value=sharpness,
                optimal_value=200.0
            ))

        # Compute overall quality score
        exposure_score = 1.0 - abs(mean_intensity - 120) / 120
        contrast_score = min(std_intensity / 50, 1.0)
        glare_score = 1.0 - min(saturation_ratio / 0.05, 1.0)
        sharpness_score = min(sharpness / 200, 1.0)

        quality_score = (exposure_score + contrast_score + glare_score + sharpness_score) / 4.0
        quality_score = max(0.0, min(1.0, quality_score))

        return quality_score, recommendations

    def generate_guidance_overlay(self,
                                 frame: np.ndarray,
                                 recommendations: List[CaptureRecommendation],
                                 quality_score: float) -> str:
        """
        Generate text overlay for AR display

        Returns:
            Text string for on-screen display
        """
        lines = []
        lines.append(f"═══ CAPTURE QUALITY: {quality_score*100:.0f}% ═══")
        lines.append("")

        # Group by priority
        critical = [r for r in recommendations if r.priority == 1]
        important = [r for r in recommendations if r.priority == 2]

        if critical:
            lines.append("🔴 CRITICAL ADJUSTMENTS:")
            for rec in critical:
                lines.append(f"  • {rec.recommendation}")
            lines.append("")

        if important:
            lines.append("🟡 RECOMMENDED IMPROVEMENTS:")
            for rec in important:
                lines.append(f"  • {rec.recommendation}")
            lines.append("")

        if not critical and not important:
            lines.append("✓ Capture settings optimal")
            lines.append("")

        # Optical parameters
        lines.append(f"Brewster angle: {self.optical_props.compute_brewster_angle():.1f}°")
        lines.append(f"Optimal for polarized light")

        return "\n".join(lines)


# ============================================================================
# WAVELENGTH-SPECIFIC LIGHTING PROFILES
# ============================================================================

@dataclass
class LightingProfile:
    """
    Complete lighting profile for specific analysis type
    """
    name: str
    wavelength_nm: Optional[float]
    color_temp_K: Optional[float]
    polarization: bool
    angle_degrees: float
    mode: LightingMode

    physics_rationale: str
    expected_results: str

    def __str__(self) -> str:
        profile = f"═══ {self.name} PROFILE ═══\n"

        if self.wavelength_nm:
            profile += f"Wavelength: {self.wavelength_nm:.0f} nm\n"
        if self.color_temp_K:
            profile += f"Color Temp: {self.color_temp_K:.0f} K\n"

        profile += f"Polarization: {'YES (crossed)' if self.polarization else 'NO'}\n"
        profile += f"Light Angle: {self.angle_degrees:.0f}°\n"
        profile += f"Mode: {self.mode.value}\n"
        profile += f"\nPhysics: {self.physics_rationale}\n"
        profile += f"Expected: {expected_results}\n"

        return profile


# Predefined lighting profiles
LIGHTING_PROFILES = {
    "stress_analysis": LightingProfile(
        name="STRESS PATTERN ANALYSIS",
        wavelength_nm=589.0,  # Sodium D-line
        color_temp_K=None,
        polarization=True,
        angle_degrees=90.0,  # Transmitted
        mode=LightingMode.CROSSED_POLARIZED,
        physics_rationale="Photoelastic birefringence (Δn = C·σ) creates optical path "
                         "difference. Crossed polarizers convert stress to intensity variation.",
        expected_results="Colorful isochromatic fringes indicating stress magnitude and direction"
    ),

    "surface_detail": LightingProfile(
        name="FINE CRACK DETAIL",
        wavelength_nm=420.0,  # Blue
        color_temp_K=None,
        polarization=False,
        angle_degrees=12.0,  # Raking
        mode=LightingMode.RAKING,
        physics_rationale="Low grazing angle (10-15°) creates long shadows from surface "
                         "irregularities. Short wavelength improves resolution (Δr ∝ λ).",
        expected_results="Enhanced contrast of fine cracks, hackle marks, and surface texture"
    ),

    "general_documentation": LightingProfile(
        name="GENERAL FORENSIC DOCUMENTATION",
        wavelength_nm=None,
        color_temp_K=5500.0,  # Daylight
        polarization=False,
        angle_degrees=45.0,  # Oblique
        mode=LightingMode.OBLIQUE,
        physics_rationale="Daylight spectrum provides natural color rendering. 45° oblique "
                         "lighting balances detail visibility with minimal glare.",
        expected_results="Natural appearance with good overall contrast for photogrammetry"
    ),

    "reflection_minimized": LightingProfile(
        name="MINIMUM REFLECTION",
        wavelength_nm=None,
        color_temp_K=5500.0,
        polarization=True,  # P-polarized
        angle_degrees=56.7,  # Brewster's angle for n=1.52
        mode=LightingMode.POLARIZED,
        physics_rationale="At Brewster's angle θ_B = arctan(n), p-polarized light has zero "
                         "reflection (Fresnel equations). Surface becomes transparent to p-pol.",
        expected_results="Minimal surface glare, maximum transmission, reveals internal features"
    ),
}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demo intelligent capture guidance"""

    print("="*70)
    print("INTELLIGENT CAPTURE GUIDANCE - DEMO")
    print("="*70)

    guidance = IntelligentCaptureGuidance()

    # 1. Camera angle analysis
    print("\n1. Camera Angle Analysis:")
    print("-" * 70)

    current_angle = 60.0  # degrees
    recs = guidance.analyze_camera_angle(current_angle)

    for rec in recs:
        print(rec)
        print()

    # 2. Lighting mode recommendations
    print("\n2. Lighting Mode Recommendations:")
    print("-" * 70)

    fracture_chars = {
        'surface_roughness': 'smooth',
        'crack_width': 0.3,  # mm - fine crack
        'stress_visible': False,
        'transparency': 'clear'
    }

    recs = guidance.recommend_lighting_mode(fracture_chars)
    for rec in recs:
        print(rec)
        print()

    # 3. Wavelength recommendations
    print("\n3. Wavelength Recommendations (Stress Analysis):")
    print("-" * 70)

    recs = guidance.recommend_wavelength(analysis_goal="stress")
    for rec in recs:
        print(rec)
        print()

    # 4. Predefined lighting profiles
    print("\n4. Predefined Lighting Profiles:")
    print("-" * 70)

    for profile_name, profile in LIGHTING_PROFILES.items():
        print(f"\n{profile}")

    # 5. Simulated real-time quality assessment
    print("\n5. Real-Time Quality Assessment:")
    print("-" * 70)

    # Simulate frame (underexposed with glare)
    test_frame = np.random.randint(0, 60, size=(480, 640), dtype=np.uint8)
    test_frame[100:150, 200:250] = 255  # Glare spot

    quality, recs = guidance.assess_capture_quality_realtime(
        frame=test_frame,
        camera_angle=current_angle,
        lighting_info={'mode': 'diffuse', 'intensity': 0.5, 'angle': 45}
    )

    print(f"Quality Score: {quality*100:.1f}%\n")
    for rec in recs:
        print(rec)
        print()

    # 6. AR overlay
    print("\n6. AR Guidance Overlay:")
    print("-" * 70)
    overlay = guidance.generate_guidance_overlay(test_frame, recs, quality)
    print(overlay)

    print("\n" + "="*70)
    print("Intelligent capture guidance ready")
    print("="*70)
