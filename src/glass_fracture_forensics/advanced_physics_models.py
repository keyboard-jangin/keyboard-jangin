#!/usr/bin/env python3
"""
ADVANCED PHYSICS MODELS
=======================

Extended physical models for glass fracture forensics.
Includes dynamic effects, mixed-mode fracture, and energy balance.

PHYSICAL MODELS:
1. Dynamic stress intensity factors
2. Mixed-mode fracture criteria
3. Crack propagation energy balance
4. Thermal stress analysis
5. Residual stress estimation

REFERENCES:
- Freund (1990): Dynamic Fracture Mechanics
- Anderson (2017): Fracture Mechanics
- Lawn (1993): Fracture of Brittle Solids

Author: Forensic Engineering Team
Version: 2.2
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from scipy.optimize import fsolve


@dataclass
class DynamicStressIntensity:
    """
    Dynamic stress intensity factors

    Accounts for crack velocity effects on stress field.

    THEORY (Freund 1990):
    K_dyn = K_static · g(v/v_R)

    where:
    - v: crack velocity
    - v_R: Rayleigh wave speed
    - g: universal function

    CONSTRAINT:
    v < v_R ≈ 0.92·v_s (v_s = shear wave speed)
    """
    K_static: float         # Static SIF [Pa·√m]
    crack_velocity: float   # Crack speed [m/s]
    material_density: float # ρ [kg/m³]
    shear_modulus: float    # G [Pa]
    poisson_ratio: float    # ν

    def compute_rayleigh_speed(self) -> float:
        """
        Compute Rayleigh wave speed

        EQUATION:
        v_s = √(G/ρ) (shear wave speed)
        v_R ≈ 0.92·v_s for typical Poisson ratios
        """
        v_s = np.sqrt(self.shear_modulus / self.material_density)
        v_R = 0.92 * v_s  # Approximate
        return v_R

    def universal_function(self, v_normalized: float) -> float:
        """
        Universal function g(v/v_R)

        APPROXIMATION (Freund 1990):
        g(β) ≈ √(1 - β) for mode I

        where β = v/v_R
        """
        if v_normalized >= 1.0:
            raise ValueError("Crack speed exceeds Rayleigh wave speed - physically impossible")

        return np.sqrt(1.0 - v_normalized)

    def compute_dynamic_K(self) -> float:
        """
        Compute dynamic stress intensity factor

        Returns:
            K_dyn [Pa·√m]
        """
        v_R = self.compute_rayleigh_speed()
        v_normalized = self.crack_velocity / v_R

        if v_normalized >= 1.0:
            # Crack branching regime
            return self.K_static * 0.5  # Approximate branching condition

        g = self.universal_function(v_normalized)
        K_dyn = self.K_static * g

        return K_dyn


@dataclass
class MixedModeFracture:
    """
    Mixed-mode fracture analysis (Modes I + II + III)

    THEORY:
    Multiple fracture criteria exist:
    1. Maximum tangential stress (MTS)
    2. Strain energy density (SED)
    3. Maximum energy release rate

    For brittle materials, MTS criterion most common.
    """
    K_I: float    # Mode I SIF [Pa·√m]
    K_II: float   # Mode II SIF [Pa·√m]
    K_III: float  # Mode III SIF [Pa·√m]
    K_Ic: float   # Mode I fracture toughness [Pa·√m]

    def compute_equivalent_K(self, criterion: str = "mts") -> float:
        """
        Compute equivalent stress intensity factor

        Args:
            criterion: Fracture criterion ('mts', 'sed', 'mer')

        Returns:
            K_eq [Pa·√m]
        """
        if criterion == "mts":
            # Maximum Tangential Stress criterion
            # K_eq = K_I·cos³(θ/2) + 3/2·K_II·sin(θ)·cos²(θ/2)
            # Simplified for θ=0 (crack growth direction):
            K_eq = np.sqrt(K_I**2 + K_II**2)

        elif criterion == "sed":
            # Strain Energy Density criterion
            # More complex - requires full calculation
            K_eq = np.sqrt(K_I**2 + K_II**2 + K_III**2)

        elif criterion == "mer":
            # Maximum Energy Release Rate
            # G = (K_I² + K_II²)/E' + K_III²/(2G)
            # Simplified:
            K_eq = np.sqrt(K_I**2 + K_II**2 + 0.5*K_III**2)

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return K_eq

    def predict_crack_angle(self) -> float:
        """
        Predict crack propagation angle using MTS criterion

        EQUATION (Anderson 2017):
        tan(θ/2) = (K_I ± √(K_I² + 8K_II²))/(4K_II)

        Returns:
            θ: crack angle [radians]
        """
        if abs(self.K_II) < 1e-6:
            return 0.0  # Pure mode I

        # Solve for angle
        discriminant = self.K_I**2 + 8*self.K_II**2
        tan_half_theta = (self.K_I + np.sqrt(discriminant)) / (4*self.K_II)

        half_theta = np.arctan(tan_half_theta)
        theta = 2 * half_theta

        return theta

    def check_fracture_criterion(self, criterion: str = "mts") -> Tuple[bool, float]:
        """
        Check if fracture criterion is satisfied

        Returns:
            (will_fracture, safety_factor)
        """
        K_eq = self.compute_equivalent_K(criterion)
        safety_factor = self.K_Ic / K_eq

        will_fracture = K_eq >= self.K_Ic

        return will_fracture, safety_factor


@dataclass
class CrackEnergyBalance:
    """
    Energy balance during crack propagation

    THEORY (Griffith 1920, Irwin 1956):
    Energy release rate: G = dU/dA

    For linear elastic:
    G = K²/E'

    where E' = E for plane stress
          E' = E/(1-ν²) for plane strain
    """
    stress_intensity: float  # K [Pa·√m]
    youngs_modulus: float    # E [Pa]
    poisson_ratio: float     # ν
    stress_state: str        # 'plane_stress' or 'plane_strain'

    def compute_effective_modulus(self) -> float:
        """Compute effective Young's modulus E'"""
        if self.stress_state == 'plane_stress':
            E_prime = self.youngs_modulus
        elif self.stress_state == 'plane_strain':
            E_prime = self.youngs_modulus / (1 - self.poisson_ratio**2)
        else:
            raise ValueError(f"Unknown stress state: {self.stress_state}")

        return E_prime

    def compute_energy_release_rate(self) -> float:
        """
        Compute energy release rate G

        EQUATION:
        G = K²/E'

        Returns:
            G [J/m²]
        """
        E_prime = self.compute_effective_modulus()
        G = self.stress_intensity**2 / E_prime

        return G

    def compute_surface_energy(self, K_Ic: float) -> float:
        """
        Compute critical surface energy γ_c

        EQUATION:
        γ_c = G_c/2 = K_Ic²/(2E')

        Returns:
            γ_c [J/m²]
        """
        E_prime = self.compute_effective_modulus()
        gamma_c = K_Ic**2 / (2 * E_prime)

        return gamma_c


@dataclass
class ThermalStressAnalysis:
    """
    Thermal stress analysis for glass

    THEORY:
    Thermal shock occurs when temperature gradient creates stress:
    σ_thermal = α·E·ΔT/(1-ν)

    where:
    - α: coefficient of thermal expansion
    - ΔT: temperature difference
    """
    alpha: float = 9e-6      # CTE [1/K] for soda-lime glass
    E: float = 72e9          # Young's modulus [Pa]
    nu: float = 0.23         # Poisson's ratio
    T_ref: float = 293.15    # Reference temperature [K]

    def compute_thermal_stress(self, T_surface: float, T_interior: float) -> float:
        """
        Compute thermal stress from temperature gradient

        Args:
            T_surface: Surface temperature [K]
            T_interior: Interior temperature [K]

        Returns:
            σ_thermal [Pa] (tensile positive)
        """
        delta_T = T_surface - T_interior

        # Thermal stress (constrained expansion)
        sigma_thermal = self.alpha * self.E * delta_T / (1 - self.nu)

        return sigma_thermal

    def estimate_critical_temperature_diff(self, K_Ic: float, flaw_size: float) -> float:
        """
        Estimate critical ΔT for fracture initiation

        EQUATION:
        K_I = σ·√(πa)
        σ_crit = K_Ic/√(πa)
        ΔT_crit = σ_crit·(1-ν)/(α·E)

        Args:
            K_Ic: Fracture toughness [Pa·√m]
            flaw_size: Initial flaw size [m]

        Returns:
            ΔT_crit [K]
        """
        sigma_crit = K_Ic / np.sqrt(np.pi * flaw_size)
        delta_T_crit = sigma_crit * (1 - self.nu) / (self.alpha * self.E)

        return delta_T_crit


@dataclass
class ResidualStressEstimator:
    """
    Estimate residual stresses from fracture pattern

    THEORY:
    Residual stresses affect fracture patterns:
    - Compressive: crack arrest, branching
    - Tensile: accelerated propagation
    """

    def estimate_from_crack_speed(self,
                                  observed_K: float,
                                  expected_K_static: float,
                                  crack_velocity: float,
                                  material) -> float:
        """
        Estimate residual stress from crack dynamics

        If observed K > expected K (accounting for dynamics),
        residual tensile stress present.

        Args:
            observed_K: Measured stress intensity [Pa·√m]
            expected_K_static: Expected static K [Pa·√m]
            crack_velocity: Crack speed [m/s]
            material: Material properties

        Returns:
            σ_residual [Pa] (tensile positive)
        """
        # Account for dynamic effects
        dyn = DynamicStressIntensity(
            K_static=expected_K_static,
            crack_velocity=crack_velocity,
            material_density=material.rho,
            shear_modulus=material.E / (2 * (1 + material.nu)),
            poisson_ratio=material.nu
        )

        expected_K_dynamic = dyn.compute_dynamic_K()

        # Difference indicates residual stress
        delta_K = observed_K - expected_K_dynamic

        # Approximate conversion (requires flaw size)
        # σ_residual ≈ ΔK / √(πa)
        # For order of magnitude, assume a ~ 1mm
        a_assumed = 1e-3  # 1mm
        sigma_residual = delta_K / np.sqrt(np.pi * a_assumed)

        return sigma_residual


# ============================================================================
# INTEGRATED PHYSICAL VALIDATION
# ============================================================================

class PhysicsValidator:
    """
    Validate forensic results against physical laws
    """

    def __init__(self, material):
        self.material = material

    def validate_energy_balance(self,
                               stress_intensity: float,
                               crack_length: float,
                               applied_stress: float) -> Dict[str, any]:
        """
        Check energy balance

        Total energy = Elastic energy - Surface energy

        Returns validation results
        """
        # Energy release rate
        energy_balance = CrackEnergyBalance(
            stress_intensity=stress_intensity,
            youngs_modulus=self.material.E,
            poisson_ratio=self.material.nu,
            stress_state=self.material.stress_state
        )

        G = energy_balance.compute_energy_release_rate()
        gamma_c = energy_balance.compute_surface_energy(self.material.K_Ic)

        # Critical condition: G ≥ 2γ
        is_valid = G >= 2 * gamma_c * 0.9  # 10% tolerance

        return {
            'valid': is_valid,
            'energy_release_rate': G,
            'critical_surface_energy': gamma_c,
            'energy_ratio': G / (2 * gamma_c),
            'interpretation': (
                f"Energy balance {'SATISFIED' if is_valid else 'VIOLATED'}. "
                f"G/{(2*gamma_c)} = {G/(2*gamma_c):.2f}"
            )
        }

    def validate_crack_speed(self, crack_velocity: float) -> Dict[str, any]:
        """
        Validate crack speed is below theoretical limit

        v < v_R ≈ 0.92·√(G/ρ)
        """
        G = self.material.E / (2 * (1 + self.material.nu))
        v_s = np.sqrt(G / self.material.rho)
        v_R = 0.92 * v_s

        is_valid = crack_velocity < v_R

        return {
            'valid': is_valid,
            'crack_velocity': crack_velocity,
            'rayleigh_speed': v_R,
            'velocity_ratio': crack_velocity / v_R,
            'interpretation': (
                f"Crack speed {'PLAUSIBLE' if is_valid else 'EXCEEDS PHYSICAL LIMIT'}. "
                f"v/v_R = {crack_velocity/v_R:.2f}"
            )
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demo advanced physics models"""

    print("="*70)
    print("ADVANCED PHYSICS MODELS - DEMO")
    print("="*70)

    # Material properties (soda-lime glass)
    from dataclasses import dataclass as dc

    @dc
    class Material:
        E: float = 72e9
        nu: float = 0.23
        rho: float = 2500.0
        K_Ic: float = 0.75e6
        stress_state: str = 'plane_stress'

    material = Material()

    # 1. Dynamic stress intensity
    print("\n1. Dynamic Stress Intensity:")
    print("-" * 70)

    K_static = 0.5e6  # 50% of K_Ic
    v_crack = 500.0   # 500 m/s (fast crack)

    dyn = DynamicStressIntensity(
        K_static=K_static,
        crack_velocity=v_crack,
        material_density=material.rho,
        shear_modulus=material.E / (2 * (1 + material.nu)),
        poisson_ratio=material.nu
    )

    K_dyn = dyn.compute_dynamic_K()
    v_R = dyn.compute_rayleigh_speed()

    print(f"  Static K: {K_static/1e6:.3f} MPa·√m")
    print(f"  Crack velocity: {v_crack:.0f} m/s")
    print(f"  Rayleigh speed: {v_R:.0f} m/s")
    print(f"  v/v_R: {v_crack/v_R:.2f}")
    print(f"  Dynamic K: {K_dyn/1e6:.3f} MPa·√m")
    print(f"  Reduction: {(1 - K_dyn/K_static)*100:.1f}%")

    # 2. Mixed-mode fracture
    print("\n2. Mixed-Mode Fracture:")
    print("-" * 70)

    mixed = MixedModeFracture(
        K_I=0.5e6,
        K_II=0.2e6,
        K_III=0.1e6,
        K_Ic=material.K_Ic
    )

    K_eq = mixed.compute_equivalent_K("mts")
    theta = mixed.predict_crack_angle()
    will_fracture, sf = mixed.check_fracture_criterion()

    print(f"  K_I: {mixed.K_I/1e6:.3f} MPa·√m")
    print(f"  K_II: {mixed.K_II/1e6:.3f} MPa·√m")
    print(f"  K_eq (MTS): {K_eq/1e6:.3f} MPa·√m")
    print(f"  Predicted angle: {np.degrees(theta):.1f}°")
    print(f"  Will fracture: {will_fracture}")
    print(f"  Safety factor: {sf:.2f}")

    # 3. Energy balance
    print("\n3. Energy Balance:")
    print("-" * 70)

    energy = CrackEnergyBalance(
        stress_intensity=0.6e6,
        youngs_modulus=material.E,
        poisson_ratio=material.nu,
        stress_state=material.stress_state
    )

    G = energy.compute_energy_release_rate()
    gamma_c = energy.compute_surface_energy(material.K_Ic)

    print(f"  Energy release rate G: {G:.2f} J/m²")
    print(f"  Critical surface energy γ_c: {gamma_c:.2f} J/m²")
    print(f"  2γ_c: {2*gamma_c:.2f} J/m²")
    print(f"  Griffith criterion: G >= 2γ: {G >= 2*gamma_c}")

    # 4. Thermal stress
    print("\n4. Thermal Stress Analysis:")
    print("-" * 70)

    thermal = ThermalStressAnalysis()

    T_hot = 373.15    # 100°C
    T_cold = 273.15   # 0°C

    sigma_thermal = thermal.compute_thermal_stress(T_hot, T_cold)
    delta_T_crit = thermal.estimate_critical_temperature_diff(material.K_Ic, 1e-3)

    print(f"  Temperature difference: {T_hot - T_cold:.1f} K")
    print(f"  Thermal stress: {sigma_thermal/1e6:.1f} MPa")
    print(f"  Critical ΔT (1mm flaw): {delta_T_crit:.1f} K")

    # 5. Physics validation
    print("\n5. Physics Validation:")
    print("-" * 70)

    validator = PhysicsValidator(material)

    energy_check = validator.validate_energy_balance(0.6e6, 0.01, 10e6)
    speed_check = validator.validate_crack_speed(500.0)

    print(f"  Energy balance: {energy_check['interpretation']}")
    print(f"  Crack speed: {speed_check['interpretation']}")

    print("\n" + "="*70)
    print("Advanced physics models ready")
    print("="*70)
