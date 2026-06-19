"""Canonical inputs for the tidal-deformation benchmark.

This module contains data only. The complete numerical pipeline lives in
``run_benchmark.py`` and imports the single ``CONFIG`` object defined here.

All quantities use SI units unless a field name states otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PhysicalConstants:
    gravitational_constant_m3_kg_s2: float = 6.67430e-11
    earth_mass_kg: float = 5.9722e24
    earth_mean_radius_m: float = 6.3710e6

    @property
    def earth_gravitational_parameter_m3_s2(self) -> float:
        return self.gravitational_constant_m3_kg_s2 * self.earth_mass_kg

    @property
    def earth_mean_density_kg_m3(self) -> float:
        volume = (4.0 / 3.0) * 3.141592653589793 * self.earth_mean_radius_m**3
        return self.earth_mass_kg / volume


@dataclass(frozen=True)
class AsteroidInputs:
    diameter_m: float = 340.0
    density_kg_m3: float = 3200.0
    young_modulus_pa: float = 1.0e7
    poisson_ratio: float = 0.25
    rotation_period_s: float = 30.52 * 3600.0
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    @property
    def radius_m(self) -> float:
        """Reference spherical radius, derived to prevent radius/diameter drift."""
        return 0.5 * self.diameter_m

    @property
    def shear_modulus_pa(self) -> float:
        return self.young_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_lambda_pa(self) -> float:
        nu = self.poisson_ratio
        return self.young_modulus_pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def spin_rate_rad_s(self) -> float:
        return 2.0 * 3.141592653589793 / self.rotation_period_s


@dataclass(frozen=True)
class EncounterInputs:
    periapsis_earth_radii: float = 5.96
    asymptotic_speed_m_s: float = 5954.7444


@dataclass(frozen=True)
class NumericalInputs:
    trajectory_max_periapsis_radii: float = 8.0
    trajectory_points: int = 60001
    periapsis_half_window_h: float = 1.5
    periapsis_points: int = 40001
    modal_max_periapsis_radii: float = 50.0
    modal_points: int = 120001
    figure_dpi: int = 240
    csv_float_format: str = "%.15e"
    modal_convergence_relative_tolerance: float = 2.0e-4
    validation_relative_tolerance: float = 2.0e-10


@dataclass(frozen=True)
class SweepInputs:
    reduced_stiffness_factors: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001)
    modulus_min_pa: float = 1.0e3
    modulus_max_pa: float = 1.0e9
    modulus_points: int = 121
    periapsis_min_earth_radii: float = 1.5
    periapsis_max_earth_radii: float = 15.0
    periapsis_points: int = 100
    speed_min_m_s: float = 2.0e3
    speed_max_m_s: float = 20.0e3
    speed_points: int = 100


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str = "tidal_deformation_reference_v1"
    modal_forcing_projection_factor: float = 1.0
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    asteroid: AsteroidInputs = field(default_factory=AsteroidInputs)
    encounter: EncounterInputs = field(default_factory=EncounterInputs)
    numerics: NumericalInputs = field(default_factory=NumericalInputs)
    sweeps: SweepInputs = field(default_factory=SweepInputs)

    def validate(self) -> None:
        c = self.constants
        a = self.asteroid
        e = self.encounter
        n = self.numerics
        s = self.sweeps

        positive = {
            "G": c.gravitational_constant_m3_kg_s2,
            "Earth mass": c.earth_mass_kg,
            "Earth radius": c.earth_mean_radius_m,
            "asteroid diameter": a.diameter_m,
            "asteroid density": a.density_kg_m3,
            "Young modulus": a.young_modulus_pa,
            "rotation period": a.rotation_period_s,
            "periapsis": e.periapsis_earth_radii,
            "asymptotic speed": e.asymptotic_speed_m_s,
            "modal forcing factor": self.modal_forcing_projection_factor,
        }
        for name, value in positive.items():
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")

        if not -1.0 < a.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must lie in (-1, 0.5).")

        axis_norm_sq = sum(component * component for component in a.rotation_axis)
        if axis_norm_sq <= 0.0:
            raise ValueError("Rotation axis must be non-zero.")

        for name, points in {
            "trajectory_points": n.trajectory_points,
            "periapsis_points": n.periapsis_points,
            "modal_points": n.modal_points,
        }.items():
            if points < 5 or points % 2 == 0:
                raise ValueError(f"{name} must be an odd integer >= 5.")

        if not 1.0 < n.trajectory_max_periapsis_radii:
            raise ValueError("The trajectory extent must exceed periapsis.")
        if not n.modal_max_periapsis_radii >= n.trajectory_max_periapsis_radii:
            raise ValueError("The modal trajectory must cover the plotted trajectory.")
        if not s.modulus_min_pa < s.modulus_max_pa:
            raise ValueError("Invalid Young-modulus sweep interval.")
        if any(factor <= 0.0 for factor in s.reduced_stiffness_factors):
            raise ValueError("All stiffness factors must be positive.")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["asteroid"]["radius_m"] = self.asteroid.radius_m
        data["asteroid"]["shear_modulus_pa"] = self.asteroid.shear_modulus_pa
        data["asteroid"]["lame_lambda_pa"] = self.asteroid.lame_lambda_pa
        data["asteroid"]["spin_rate_rad_s"] = self.asteroid.spin_rate_rad_s
        data["constants"]["earth_gravitational_parameter_m3_s2"] = (
            self.constants.earth_gravitational_parameter_m3_s2
        )
        data["constants"]["earth_mean_density_kg_m3"] = (
            self.constants.earth_mean_density_kg_m3
        )
        return data


CONFIG = BenchmarkConfig()

