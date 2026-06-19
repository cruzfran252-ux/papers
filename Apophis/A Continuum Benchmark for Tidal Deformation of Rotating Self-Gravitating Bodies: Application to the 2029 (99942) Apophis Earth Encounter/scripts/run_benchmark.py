"""Self-contained tidal-deformation benchmark pipeline.

Run from the paper directory:

    python run_benchmark.py

The script imports only ``inputs.py`` from this repository. It computes the
nominal hyperbolic encounter, dimensional and nondimensional groups, the
quadrupolar modal response, reduced-stiffness cases, regime maps, tables,
figures, validation checks, and a SHA-256 provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
WORKSPACE_SITE_PACKAGES = (
    ROOT
    / "TFM"
    / "codes"
    / "python_def"
    / "proyecto"
    / ".venv"
    / "Lib"
    / "site-packages"
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    if WORKSPACE_SITE_PACKAGES.is_dir():
        sys.path.insert(0, str(WORKSPACE_SITE_PACKAGES))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
        except ImportError:
            raise SystemExit(
                "Missing dependency. Install numpy, pandas, and matplotlib "
                "before running run_benchmark.py."
            ) from exc
    else:
        raise SystemExit(
            "Missing dependency. Install numpy, pandas, and matplotlib before "
            "running run_benchmark.py."
        ) from exc

from inputs import AsteroidInputs, BenchmarkConfig, CONFIG


@dataclass(frozen=True)
class Derived:
    earth_mu_m3_s2: float
    earth_density_kg_m3: float
    asteroid_radius_m: float
    shear_modulus_pa: float
    lame_lambda_pa: float
    spin_rate_rad_s: float
    periapsis_m: float
    periapsis_speed_m_s: float
    hyperbolic_alpha_m: float
    eccentricity: float
    semilatus_rectum_m: float
    specific_angular_momentum_m2_s: float
    hyperbolic_mean_motion_rad_s: float
    elastic_time_s: float
    flyby_time_s: float
    modal_encounter_time_s: float
    displacement_scale_m: float
    tidal_acceleration_scale_m_s2: float
    deformation_ratio_eta: float
    chi: float
    xi_self_gravity: float
    lambda_centrifugal: float
    coriolis_group: float
    coriolis_assembled_coefficient: float
    euler_group: float
    pi_lambda: float
    omega_elastic_rad_s: float
    omega_self_gravity_rad_s: float
    omega_rotation_rad_s: float
    omega_quadrupole_rad_s: float
    quadrupole_period_s: float
    omega_love_rad_s: float
    zeta: float
    modal_mass_kg: float
    roche_rigid_m: float
    roche_fluid_m: float


@dataclass
class Trajectory:
    time_s: np.ndarray
    hyperbolic_anomaly: np.ndarray
    true_anomaly_rad: np.ndarray
    earth_position_m: np.ndarray
    asteroid_position_m: np.ndarray
    distance_m: np.ndarray
    distance_over_periapsis: np.ndarray
    tidal_modulation: np.ndarray
    direction_body: np.ndarray
    tidal_tensor_body: np.ndarray


@dataclass
class ModalResponse:
    time_s: np.ndarray
    distance_m: np.ndarray
    forcing_m_s2: np.ndarray
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    quasi_static_m: np.ndarray
    adiabatic_second_order_m: np.ndarray
    amplitude_m: np.ndarray
    specific_energy_m2_s2: np.ndarray
    modal_energy_j: np.ndarray


def style_plots(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": dpi,
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "CMU Serif",
                "Computer Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.38,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.05,
            "figure.constrained_layout.use": False,
        }
    )


def compute_derived(config: BenchmarkConfig, asteroid: AsteroidInputs | None = None) -> Derived:
    c = config.constants
    a = asteroid or config.asteroid
    encounter = config.encounter

    earth_mu = c.earth_gravitational_parameter_m3_s2
    earth_density = c.earth_mean_density_kg_m3
    radius = a.radius_m
    shear = a.shear_modulus_pa
    lame = a.lame_lambda_pa
    spin = a.spin_rate_rad_s
    periapsis = encounter.periapsis_earth_radii * c.earth_mean_radius_m
    v_inf = encounter.asymptotic_speed_m_s
    v_periapsis = math.sqrt(v_inf**2 + 2.0 * earth_mu / periapsis)

    alpha = earth_mu / v_inf**2
    eccentricity = 1.0 + periapsis / alpha
    semilatus = alpha * (eccentricity**2 - 1.0)
    angular_momentum = math.sqrt(earth_mu * semilatus)
    mean_motion = math.sqrt(earth_mu / alpha**3)

    elastic_time = radius * math.sqrt(a.density_kg_m3 / shear)
    modal_encounter_time = periapsis / v_periapsis
    flyby_time = modal_encounter_time
    tidal_acceleration = earth_mu * radius / periapsis**3
    displacement_scale = (
        a.density_kg_m3 * tidal_acceleration * radius**2 / shear
    )
    eta = displacement_scale / radius
    chi = elastic_time / flyby_time
    xi = 4.0 * math.pi * c.gravitational_constant_m3_kg_s2 * a.density_kg_m3 * elastic_time**2
    centrifugal = spin**2 * periapsis**3 / earth_mu
    coriolis_group = spin * elastic_time * chi
    coriolis_assembled = 2.0 * coriolis_group
    euler_group = spin / flyby_time * periapsis**3 / earth_mu
    pi_lambda = lame / shear

    omega_elastic_sq = 5.0 * (4.0 * shear + 3.0 * lame) / (
        a.density_kg_m3 * radius**2
    )
    omega_gravity_sq = (
        72.0
        * math.pi
        / 25.0
        * c.gravitational_constant_m3_kg_s2
        * a.density_kg_m3
    )
    omega_rotation_sq = (10.0 / 21.0) * spin**2
    omega_quadrupole = math.sqrt(
        omega_elastic_sq + omega_gravity_sq + omega_rotation_sq
    )
    omega_love = math.sqrt(
        19.0 * shear / (2.0 * a.density_kg_m3 * radius**2)
    )
    modal_mass = (4.0 * math.pi / 25.0) * a.density_kg_m3 * radius**3

    roche_rigid = (
        1.26
        * c.earth_mean_radius_m
        * (earth_density / a.density_kg_m3) ** (1.0 / 3.0)
    )
    roche_fluid = (
        2.44
        * c.earth_mean_radius_m
        * (earth_density / a.density_kg_m3) ** (1.0 / 3.0)
    )

    return Derived(
        earth_mu_m3_s2=earth_mu,
        earth_density_kg_m3=earth_density,
        asteroid_radius_m=radius,
        shear_modulus_pa=shear,
        lame_lambda_pa=lame,
        spin_rate_rad_s=spin,
        periapsis_m=periapsis,
        periapsis_speed_m_s=v_periapsis,
        hyperbolic_alpha_m=alpha,
        eccentricity=eccentricity,
        semilatus_rectum_m=semilatus,
        specific_angular_momentum_m2_s=angular_momentum,
        hyperbolic_mean_motion_rad_s=mean_motion,
        elastic_time_s=elastic_time,
        flyby_time_s=flyby_time,
        modal_encounter_time_s=modal_encounter_time,
        displacement_scale_m=displacement_scale,
        tidal_acceleration_scale_m_s2=tidal_acceleration,
        deformation_ratio_eta=eta,
        chi=chi,
        xi_self_gravity=xi,
        lambda_centrifugal=centrifugal,
        coriolis_group=coriolis_group,
        coriolis_assembled_coefficient=coriolis_assembled,
        euler_group=euler_group,
        pi_lambda=pi_lambda,
        omega_elastic_rad_s=math.sqrt(omega_elastic_sq),
        omega_self_gravity_rad_s=math.sqrt(omega_gravity_sq),
        omega_rotation_rad_s=math.sqrt(omega_rotation_sq),
        omega_quadrupole_rad_s=omega_quadrupole,
        quadrupole_period_s=2.0 * math.pi / omega_quadrupole,
        omega_love_rad_s=omega_love,
        zeta=omega_quadrupole * modal_encounter_time,
        modal_mass_kg=modal_mass,
        roche_rigid_m=roche_rigid,
        roche_fluid_m=roche_fluid,
    )


def normalized_axis(axis: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(axis), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Rotation axis must be non-zero.")
    return vector / norm


def solve_hyperbolic_anomaly(
    time_s: np.ndarray, eccentricity: float, mean_motion: float
) -> np.ndarray:
    mean_anomaly = mean_motion * np.asarray(time_s, dtype=float)
    anomaly = np.arcsinh(mean_anomaly / eccentricity)
    for _ in range(20):
        residual = eccentricity * np.sinh(anomaly) - anomaly - mean_anomaly
        derivative = eccentricity * np.cosh(anomaly) - 1.0
        increment = residual / derivative
        anomaly -= increment
        if float(np.max(np.abs(increment))) < 2.0e-14:
            break
    return anomaly


def rodrigues_rotate(vectors: np.ndarray, angles: np.ndarray, axis: np.ndarray) -> np.ndarray:
    cross = np.cross(np.broadcast_to(axis, vectors.shape), vectors)
    dot = vectors @ axis
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    return (
        vectors * cosine
        + cross * sine
        + dot[:, None] * axis[None, :] * (1.0 - cosine)
    )


def trajectory_from_anomaly(
    anomaly: np.ndarray,
    derived: Derived,
    config: BenchmarkConfig,
) -> Trajectory:
    eccentricity = derived.eccentricity
    alpha = derived.hyperbolic_alpha_m
    cosh_h = np.cosh(anomaly)
    sinh_h = np.sinh(anomaly)
    denominator = eccentricity * cosh_h - 1.0
    distance = alpha * denominator
    cosine_true = (eccentricity - cosh_h) / denominator
    sine_true = (
        math.sqrt(eccentricity**2 - 1.0) * sinh_h / denominator
    )
    true_anomaly = np.arctan2(sine_true, cosine_true)
    earth_position = np.column_stack(
        (
            distance * cosine_true,
            distance * sine_true,
            np.zeros_like(distance),
        )
    )
    asteroid_position = -earth_position
    direction_inertial = asteroid_position / distance[:, None]
    time_s = (
        eccentricity * sinh_h - anomaly
    ) / derived.hyperbolic_mean_motion_rad_s

    axis = normalized_axis(config.asteroid.rotation_axis)
    direction_body = rodrigues_rotate(
        direction_inertial,
        derived.spin_rate_rad_s * time_s,
        axis,
    )
    identity = np.eye(3)
    shape_tensor = (
        3.0 * direction_body[:, :, None] * direction_body[:, None, :]
        - identity[None, :, :]
    )
    distance_ratio = distance / derived.periapsis_m
    modulation = distance_ratio**-3
    tidal_tensor = modulation[:, None, None] * shape_tensor

    return Trajectory(
        time_s=time_s,
        hyperbolic_anomaly=anomaly,
        true_anomaly_rad=true_anomaly,
        earth_position_m=earth_position,
        asteroid_position_m=asteroid_position,
        distance_m=distance,
        distance_over_periapsis=distance_ratio,
        tidal_modulation=modulation,
        direction_body=direction_body,
        tidal_tensor_body=tidal_tensor,
    )


def compute_trajectory(
    config: BenchmarkConfig,
    derived: Derived,
    *,
    max_periapsis_radii: float,
    points: int,
) -> Trajectory:
    argument = (
        max_periapsis_radii * derived.periapsis_m / derived.hyperbolic_alpha_m
        + 1.0
    ) / derived.eccentricity
    max_anomaly = math.acosh(argument)
    anomaly = np.linspace(-max_anomaly, max_anomaly, points)
    return trajectory_from_anomaly(anomaly, derived, config)


def compute_periapsis_trajectory(
    config: BenchmarkConfig, derived: Derived, *, points: int | None = None
) -> Trajectory:
    count = points or config.numerics.periapsis_points
    half_window_s = config.numerics.periapsis_half_window_h * 3600.0
    time_s = np.linspace(-half_window_s, half_window_s, count)
    anomaly = solve_hyperbolic_anomaly(
        time_s, derived.eccentricity, derived.hyperbolic_mean_motion_rad_s
    )
    return trajectory_from_anomaly(anomaly, derived, config)


def integrate_piecewise_linear_forcing(
    time_s: np.ndarray,
    forcing_m_s2: np.ndarray,
    omega_rad_s: float,
    initial_displacement_m: float = 0.0,
    initial_velocity_m_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact state propagation for linear forcing on each time interval."""
    time_s = np.asarray(time_s, dtype=float)
    forcing_m_s2 = np.asarray(forcing_m_s2, dtype=float)
    displacement = np.zeros_like(time_s)
    velocity = np.zeros_like(time_s)
    displacement[0] = initial_displacement_m
    velocity[0] = initial_velocity_m_s
    omega_sq = omega_rad_s**2
    omega_cu = omega_sq * omega_rad_s

    for index in range(len(time_s) - 1):
        step = float(time_s[index + 1] - time_s[index])
        f0 = float(forcing_m_s2[index])
        f1 = float(forcing_m_s2[index + 1])
        slope = (f1 - f0) / step
        phase = omega_rad_s * step
        cosine = math.cos(phase)
        sine = math.sin(phase)
        displacement[index + 1] = (
            displacement[index] * cosine
            + velocity[index] * sine / omega_rad_s
            + f0 * (1.0 - cosine) / omega_sq
            + slope * (step / omega_sq - sine / omega_cu)
        )
        velocity[index + 1] = (
            -displacement[index] * omega_rad_s * sine
            + velocity[index] * cosine
            + f0 * sine / omega_rad_s
            + slope * (1.0 - cosine) / omega_sq
        )
    return displacement, velocity


def second_derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    first = np.gradient(values, time_s, edge_order=2)
    return np.gradient(first, time_s, edge_order=2)


def compute_modal_response(
    config: BenchmarkConfig,
    derived: Derived,
    trajectory: Trajectory,
) -> ModalResponse:
    forcing = (
        config.modal_forcing_projection_factor
        * derived.earth_mu_m3_s2
        * derived.asteroid_radius_m
        / trajectory.distance_m**3
    )
    displacement, velocity = integrate_piecewise_linear_forcing(
        trajectory.time_s,
        forcing,
        derived.omega_quadrupole_rad_s,
    )
    omega_sq = derived.omega_quadrupole_rad_s**2
    quasi_static = forcing / omega_sq
    forcing_second = second_derivative(forcing, trajectory.time_s)
    adiabatic_second = quasi_static - forcing_second / omega_sq**2
    amplitude = np.sqrt(
        displacement**2
        + (velocity / derived.omega_quadrupole_rad_s) ** 2
    )
    specific_energy = 0.5 * (
        velocity**2 + omega_sq * displacement**2
    )
    modal_energy = derived.modal_mass_kg * specific_energy
    return ModalResponse(
        time_s=trajectory.time_s,
        distance_m=trajectory.distance_m,
        forcing_m_s2=forcing,
        displacement_m=displacement,
        velocity_m_s=velocity,
        quasi_static_m=quasi_static,
        adiabatic_second_order_m=adiabatic_second,
        amplitude_m=amplitude,
        specific_energy_m2_s2=specific_energy,
        modal_energy_j=modal_energy,
    )


def interpolate_modal_to_periapsis(
    modal: ModalResponse, periapsis: Trajectory, derived: Derived
) -> pd.DataFrame:
    time_s = periapsis.time_s
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "time_over_encounter": time_s / derived.flyby_time_s,
            "distance_m": periapsis.distance_m,
            "distance_over_periapsis": periapsis.distance_over_periapsis,
            "forcing_m_s2": np.interp(
                time_s, modal.time_s, modal.forcing_m_s2
            ),
            "displacement_m": np.interp(
                time_s, modal.time_s, modal.displacement_m
            ),
            "velocity_m_s": np.interp(
                time_s, modal.time_s, modal.velocity_m_s
            ),
            "quasi_static_m": np.interp(
                time_s, modal.time_s, modal.quasi_static_m
            ),
            "adiabatic_second_order_m": np.interp(
                time_s, modal.time_s, modal.adiabatic_second_order_m
            ),
            "amplitude_m": np.interp(
                time_s, modal.time_s, modal.amplitude_m
            ),
            "modal_energy_j": np.interp(
                time_s, modal.time_s, modal.modal_energy_j
            ),
        }
    )
    return add_modal_dimensionless_columns(frame, derived)


def modal_response_dataframe(
    modal: ModalResponse, derived: Derived
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time_s": modal.time_s,
            "time_over_encounter": modal.time_s / derived.flyby_time_s,
            "distance_m": modal.distance_m,
            "distance_over_periapsis": modal.distance_m / derived.periapsis_m,
            "forcing_m_s2": modal.forcing_m_s2,
            "displacement_m": modal.displacement_m,
            "velocity_m_s": modal.velocity_m_s,
            "quasi_static_m": modal.quasi_static_m,
            "adiabatic_second_order_m": modal.adiabatic_second_order_m,
            "amplitude_m": modal.amplitude_m,
            "specific_energy_m2_s2": modal.specific_energy_m2_s2,
            "modal_energy_j": modal.modal_energy_j,
        }
    )
    return add_modal_dimensionless_columns(frame, derived)


def add_modal_dimensionless_columns(
    frame: pd.DataFrame, derived: Derived
) -> pd.DataFrame:
    frame["forcing_over_a0"] = (
        frame["forcing_m_s2"] / derived.tidal_acceleration_scale_m_s2
    )
    frame["displacement_over_radius"] = (
        frame["displacement_m"] / derived.asteroid_radius_m
    )
    frame["displacement_over_Uc"] = (
        frame["displacement_m"] / derived.displacement_scale_m
    )
    frame["velocity_scaled_Uc_tenc"] = (
        frame["velocity_m_s"]
        * derived.flyby_time_s
        / derived.displacement_scale_m
    )
    frame["quasi_static_over_Uc"] = (
        frame["quasi_static_m"] / derived.displacement_scale_m
    )
    frame["amplitude_over_Uc"] = (
        frame["amplitude_m"] / derived.displacement_scale_m
    )
    return frame


def trajectory_dataframe(trajectory: Trajectory) -> pd.DataFrame:
    tensor = trajectory.tidal_tensor_body
    return pd.DataFrame(
        {
            "time_s": trajectory.time_s,
            "hyperbolic_anomaly_rad": trajectory.hyperbolic_anomaly,
            "true_anomaly_rad": trajectory.true_anomaly_rad,
            "x_earth_m": trajectory.earth_position_m[:, 0],
            "y_earth_m": trajectory.earth_position_m[:, 1],
            "x_asteroid_frame_m": trajectory.asteroid_position_m[:, 0],
            "y_asteroid_frame_m": trajectory.asteroid_position_m[:, 1],
            "distance_m": trajectory.distance_m,
            "distance_over_periapsis": trajectory.distance_over_periapsis,
            "tidal_modulation": trajectory.tidal_modulation,
            "rhat_body_x": trajectory.direction_body[:, 0],
            "rhat_body_y": trajectory.direction_body[:, 1],
            "rhat_body_z": trajectory.direction_body[:, 2],
            "tidal_tensor_xx": tensor[:, 0, 0],
            "tidal_tensor_xy": tensor[:, 0, 1],
            "tidal_tensor_xz": tensor[:, 0, 2],
            "tidal_tensor_yy": tensor[:, 1, 1],
            "tidal_tensor_yz": tensor[:, 1, 2],
            "tidal_tensor_zz": tensor[:, 2, 2],
        }
    )


def derived_dataframe(derived: Derived) -> pd.DataFrame:
    return pd.DataFrame(
        [{"quantity": key, "value": value} for key, value in vars(derived).items()]
    )


def flatten_mapping(
    mapping: dict[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten_mapping(value, name))
        elif isinstance(value, (tuple, list)):
            rows.append({"parameter": name, "value": json.dumps(value)})
        else:
            rows.append({"parameter": name, "value": value})
    return rows


def coefficient_dataframe(
    trajectory: Trajectory, derived: Derived
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": trajectory.time_s,
            "time_over_encounter": trajectory.time_s / derived.flyby_time_s,
            "signed_distance_over_periapsis": (
                np.sign(trajectory.time_s)
                * trajectory.distance_over_periapsis
            ),
            "tidal_M": trajectory.tidal_modulation,
            "elastic": np.ones_like(trajectory.time_s),
            "inertia_chi_squared": np.full_like(
                trajectory.time_s, derived.chi**2
            ),
            "self_gravity_Xi": np.full_like(
                trajectory.time_s, derived.xi_self_gravity
            ),
            "centrifugal_Lambda": np.full_like(
                trajectory.time_s, derived.lambda_centrifugal
            ),
            "coriolis_group_C": np.full_like(
                trajectory.time_s, derived.coriolis_group
            ),
            "coriolis_assembled_2C": np.full_like(
                trajectory.time_s,
                derived.coriolis_assembled_coefficient,
            ),
            "euler_group_E": np.full_like(
                trajectory.time_s, derived.euler_group
            ),
            "active_euler_term": np.zeros_like(trajectory.time_s),
        }
    )


def stiffness_sweep(
    config: BenchmarkConfig,
    modal_trajectory: Trajectory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nominal_young = config.asteroid.young_modulus_pa
    case_rows: list[dict[str, Any]] = []

    for factor in config.sweeps.reduced_stiffness_factors:
        asteroid = replace(
            config.asteroid, young_modulus_pa=nominal_young * factor
        )
        derived = compute_derived(config, asteroid)
        response = compute_modal_response(config, derived, modal_trajectory)
        peak_index = int(np.argmax(np.abs(response.displacement_m)))
        force_peak_index = int(np.argmax(response.forcing_m_s2))
        case_rows.append(
            {
                "case": "nominal" if factor == 1.0 else f"E_x_{factor:g}",
                "stiffness_factor": factor,
                "young_modulus_pa": asteroid.young_modulus_pa,
                "shear_modulus_pa": derived.shear_modulus_pa,
                "elastic_time_s": derived.elastic_time_s,
                "chi": derived.chi,
                "xi_self_gravity": derived.xi_self_gravity,
                "omega_quadrupole_rad_s": derived.omega_quadrupole_rad_s,
                "quadrupole_period_s": derived.quadrupole_period_s,
                "zeta": derived.zeta,
                "peak_dynamic_displacement_m": float(
                    np.max(np.abs(response.displacement_m))
                ),
                "peak_quasi_static_displacement_m": float(
                    np.max(np.abs(response.quasi_static_m))
                ),
                "dynamic_peak_time_s": float(response.time_s[peak_index]),
                "forcing_peak_time_s": float(response.time_s[force_peak_index]),
                "peak_modal_energy_j": float(np.max(response.modal_energy_j)),
            }
        )

    modulus = np.logspace(
        math.log10(config.sweeps.modulus_min_pa),
        math.log10(config.sweeps.modulus_max_pa),
        config.sweeps.modulus_points,
    )
    parametric_rows: list[dict[str, float]] = []
    force_peak = (
        config.modal_forcing_projection_factor
        * compute_derived(config).tidal_acceleration_scale_m_s2
    )
    for young in modulus:
        asteroid = replace(config.asteroid, young_modulus_pa=float(young))
        derived = compute_derived(config, asteroid)
        parametric_rows.append(
            {
                "young_modulus_pa": float(young),
                "shear_modulus_pa": derived.shear_modulus_pa,
                "elastic_time_s": derived.elastic_time_s,
                "chi": derived.chi,
                "xi_self_gravity": derived.xi_self_gravity,
                "omega_quadrupole_rad_s": derived.omega_quadrupole_rad_s,
                "quadrupole_period_s": derived.quadrupole_period_s,
                "zeta": derived.zeta,
                "peak_quasi_static_displacement_m": (
                    force_peak / derived.omega_quadrupole_rad_s**2
                ),
            }
        )
    return pd.DataFrame(case_rows), pd.DataFrame(parametric_rows)


def parametric_case_definitions(config: BenchmarkConfig) -> list[dict[str, Any]]:
    return [
        {
            "case": "Nominal",
            "description": "Nominal Apophis",
            "periapsis_earth_radii": config.encounter.periapsis_earth_radii,
            "asymptotic_speed_m_s": config.encounter.asymptotic_speed_m_s,
            "young_modulus_pa": config.asteroid.young_modulus_pa,
            "marker": "*",
            "size": 150,
            "facecolor": "#b21f35",
        },
        {
            "case": "P1",
            "description": "close-fast-compliant",
            "periapsis_earth_radii": 3.5,
            "asymptotic_speed_m_s": 18.0e3,
            "young_modulus_pa": 1.0e4,
            "marker": "s",
            "size": 64,
            "facecolor": "#f2f4f3",
        },
        {
            "case": "P2",
            "description": "close-slow-stiff",
            "periapsis_earth_radii": 3.5,
            "asymptotic_speed_m_s": 3.0e3,
            "young_modulus_pa": 1.0e8,
            "marker": "D",
            "size": 62,
            "facecolor": "#f2f4f3",
        },
        {
            "case": "P3",
            "description": "wide-slow-compliant",
            "periapsis_earth_radii": 10.0,
            "asymptotic_speed_m_s": 3.0e3,
            "young_modulus_pa": 1.0e4,
            "marker": "o",
            "size": 64,
            "facecolor": "#f2f4f3",
        },
        {
            "case": "P4",
            "description": "wide-fast-stiff",
            "periapsis_earth_radii": 10.0,
            "asymptotic_speed_m_s": 18.0e3,
            "young_modulus_pa": 1.0e8,
            "marker": "^",
            "size": 72,
            "facecolor": "#f2f4f3",
        },
    ]


def parametric_benchmark_cases(config: BenchmarkConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in parametric_case_definitions(config):
        asteroid = replace(
            config.asteroid,
            young_modulus_pa=float(definition["young_modulus_pa"]),
        )
        encounter = replace(
            config.encounter,
            periapsis_earth_radii=float(definition["periapsis_earth_radii"]),
            asymptotic_speed_m_s=float(definition["asymptotic_speed_m_s"]),
        )
        case_config = replace(config, asteroid=asteroid, encounter=encounter)
        derived = compute_derived(case_config, asteroid)
        modal_trajectory = compute_trajectory(
            case_config,
            derived,
            max_periapsis_radii=case_config.numerics.modal_max_periapsis_radii,
            points=case_config.numerics.modal_points,
        )
        response = compute_modal_response(case_config, derived, modal_trajectory)
        peak_index = int(np.argmax(np.abs(response.displacement_m)))
        rows.append(
            {
                **definition,
                "periapsis_m": derived.periapsis_m,
                "periapsis_speed_m_s": derived.periapsis_speed_m_s,
                "encounter_time_s": derived.flyby_time_s,
                "eta": derived.deformation_ratio_eta,
                "tau_e_squared": derived.chi**2,
                "xi_self_gravity": derived.xi_self_gravity,
                "lambda_centrifugal": derived.lambda_centrifugal,
                "zeta": derived.zeta,
                "omega_quadrupole_rad_s": derived.omega_quadrupole_rad_s,
                "peak_dynamic_displacement_m": float(
                    np.max(np.abs(response.displacement_m))
                ),
                "peak_quasi_static_displacement_m": float(
                    np.max(np.abs(response.quasi_static_m))
                ),
                "dynamic_peak_time_s": float(response.time_s[peak_index]),
            }
        )
    return pd.DataFrame(rows)


def encounter_regime_grid(
    config: BenchmarkConfig, derived: Derived
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    earth_radii = np.linspace(
        config.sweeps.periapsis_min_earth_radii,
        config.sweeps.periapsis_max_earth_radii,
        config.sweeps.periapsis_points,
    )
    speeds = np.linspace(
        config.sweeps.speed_min_m_s,
        config.sweeps.speed_max_m_s,
        config.sweeps.speed_points,
    )
    rr, vv = np.meshgrid(earth_radii, speeds)
    periapsis = rr * config.constants.earth_mean_radius_m
    periapsis_speed = np.sqrt(
        vv**2 + 2.0 * derived.earth_mu_m3_s2 / periapsis
    )
    encounter_time = periapsis / periapsis_speed
    zeta = derived.omega_quadrupole_rad_s * encounter_time
    return rr, vv, zeta


def stiffness_periapsis_regime_grid(
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    earth_radii = np.linspace(
        config.sweeps.periapsis_min_earth_radii,
        config.sweeps.periapsis_max_earth_radii,
        config.sweeps.periapsis_points,
    )
    modulus = np.logspace(
        math.log10(config.sweeps.modulus_min_pa),
        math.log10(config.sweeps.modulus_max_pa),
        config.sweeps.modulus_points,
    )
    rr, ee = np.meshgrid(earth_radii, modulus)
    c = config.constants
    a = config.asteroid
    nu = a.poisson_ratio
    shear = ee / (2.0 * (1.0 + nu))
    lame = ee * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    omega_sq = (
        5.0 * (4.0 * shear + 3.0 * lame) / (a.density_kg_m3 * a.radius_m**2)
        + (72.0 * math.pi / 25.0)
        * c.gravitational_constant_m3_kg_s2
        * a.density_kg_m3
        + (10.0 / 21.0) * a.spin_rate_rad_s**2
    )
    periapsis = rr * c.earth_mean_radius_m
    v_periapsis = np.sqrt(
        config.encounter.asymptotic_speed_m_s**2
        + 2.0 * c.earth_gravitational_parameter_m3_s2 / periapsis
    )
    encounter_time = periapsis / v_periapsis
    return rr, ee, np.sqrt(omega_sq) * encounter_time


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "run_benchmark.py"},
    )
    plt.close(fig)


def plot_trajectory(trajectory: Trajectory, derived: Derived, path: Path) -> None:
    mask = trajectory.distance_over_periapsis <= 1.75
    earth = trajectory.earth_position_m[mask, :2] / 1.0e6
    asteroid = trajectory.asteroid_position_m[mask, :2] / 1.0e6
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    axes[0].plot(earth[:, 0], earth[:, 1], color="#1769aa")
    axes[0].scatter(0.0, 0.0, s=70, color="#24445c", zorder=4)
    axes[0].text(1.5, -5.0, "Earth", color="#24445c", fontsize=9)
    axes[0].set_title("(a) Earth-centered frame")
    axes[1].plot(asteroid[:, 0], asteroid[:, 1], color="#1769aa")
    axes[1].scatter(0.0, 0.0, s=58, color="#d2601a", zorder=4)
    axes[1].text(-8.0, -5.0, "Asteroid", color="#9a3b00", fontsize=9, ha="right")
    axes[1].set_title("(b) Asteroid-centered frame")
    for axis in axes:
        axis.set_xlabel(r"$x$ [Mm]")
        axis.set_ylabel(r"$y$ [Mm]")
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-2.0, 46.0)
    axes[1].set_xlim(-46.0, 2.0)
    fig.tight_layout()
    save_figure(fig, path)


def plot_true_anomaly(periapsis: Trajectory, derived: Derived, path: Path) -> None:
    time_h = periapsis.time_s / 3600.0
    true_deg = np.degrees(periapsis.true_anomaly_rad)
    slope_rad_s = (
        derived.specific_angular_momentum_m2_s / derived.periapsis_m**2
    )
    linear_deg = np.degrees(slope_rad_s * periapsis.time_s)
    fig, ax = plt.subplots(figsize=(3.38, 2.55))
    ax.plot(
        time_h,
        true_deg,
        color="#0b5cad",
        linewidth=1.05,
        label="Hyperbolic solution",
    )
    ax.plot(
        time_h,
        linear_deg,
        "--",
        color="#5e7698",
        linewidth=0.95,
        label="Linear periapsis approximation",
    )
    ax.axvline(
        0.0,
        color="#6d1f2a",
        linestyle=":",
        linewidth=0.85,
        label="Periapsis",
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
    ax.set_xlabel("Time from periapsis [h]")
    ax.set_ylabel("True anomaly [deg]")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        ncol=1,
        loc="lower right",
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_rotation_components(periapsis: Trajectory, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.38, 2.55))
    time_h = periapsis.time_s / 3600.0
    labels = (r"$\hat r_x^{(b)}$", r"$\hat r_y^{(b)}$", r"$\hat r_z^{(b)}$")
    colors = ("#1b4965", "#7b2d26", "#5f6c7b")
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(
            time_h,
            periapsis.direction_body[:, index],
            label=label,
            color=color,
            linewidth=1.05,
        )
    ax.axvline(0.0, color="#1f2933", linestyle=":", alpha=0.7)
    ax.set_xlabel("Time from periapsis [h]")
    ax.set_ylabel("Earth direction in body frame [-]")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        ncol=3,
        loc="upper right",
        borderpad=0.25,
        handlelength=1.1,
        columnspacing=0.8,
    )
    ax.set_xlim(-1.5, 1.5)
    fig.tight_layout()
    save_figure(fig, path)


def plot_nondimensional_hierarchy(
    trajectory: Trajectory, derived: Derived, path: Path
) -> None:
    time_h = trajectory.time_s / 3600.0
    time_limit_h = min(8.0, float(np.nanmax(np.abs(time_h))))
    chi_sq = derived.chi**2
    values = (
        (r"Elastic, $1$", 1.0),
        (r"Tide, $\mathcal{M}(0)$", 1.0),
        (r"Centrifugal, $\Lambda$", derived.lambda_centrifugal),
        (r"Self-gravity, $\Xi$", derived.xi_self_gravity),
        (r"Inertia, $\tau_e^2$", chi_sq),
        (r"Coriolis, $2\mathscr{C}_\tau$", derived.coriolis_assembled_coefficient),
    )
    colors = {
        r"Elastic, $1$": "#6b6f74",
        r"Tide, $\mathcal{M}(0)$": "#0b5cad",
        r"Centrifugal, $\Lambda$": "#4477aa",
        r"Self-gravity, $\Xi$": "#66a61e",
        r"Inertia, $\tau_e^2$": "#aa3377",
        r"Coriolis, $2\mathscr{C}_\tau$": "#228833",
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.38, 4.85))
    left = axes[0]
    left.semilogy(
        time_h,
        trajectory.tidal_modulation,
        color=colors[r"Tide, $\mathcal{M}(0)$"],
        linewidth=1.0,
        label=r"Tide, $\mathcal{M}(t)=|r_T(t)/\mathcal{R}_p|^{-3}$",
    )
    for label, value in values:
        if label == r"Tide, $\mathcal{M}(0)$":
            continue
        left.axhline(
            value,
            color=colors[label],
            linewidth=0.78,
            label=label,
        )
    left.axvspan(
        -derived.flyby_time_s / 3600.0,
        derived.flyby_time_s / 3600.0,
        color="#d8dce2",
        alpha=0.28,
        zorder=0,
    )
    left.set_xlim(-time_limit_h, time_limit_h)
    left.set_ylim(1.0e-8, 3.0)
    left.set_xlabel("Time from periapsis [h]")
    left.set_ylabel("Coefficient magnitude [-]")
    left.set_title("(a) Evolution along the encounter", loc="left")
    left.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        fontsize=5.95,
        ncol=2,
        loc="lower left",
        columnspacing=0.65,
        handlelength=1.25,
        borderpad=0.25,
    )

    right = axes[1]
    names = [name for name, _ in values]
    heights = np.array([value for _, value in values])
    positions = np.arange(len(names))
    bar_colors = plt.get_cmap("cividis")(np.linspace(0.18, 0.86, len(names)))
    right.barh(
        positions,
        heights,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.45,
    )
    right.set_xscale("log")
    right.set_xlim(1.0e-8, 3.0)
    right.set_yticks(positions, names)
    right.invert_yaxis()
    right.set_xlabel("Coefficient magnitude [-]")
    right.set_title("(b) Ordering at periapsis", loc="left")
    fig.tight_layout(h_pad=1.0)
    save_figure(fig, path)


def plot_coefficient_evolution(
    trajectory: Trajectory, derived: Derived, path: Path
) -> None:
    signed_coordinate = np.sign(trajectory.time_s) * trajectory.distance_over_periapsis
    incoming = trajectory.time_s < 0.0
    outgoing = trajectory.time_s > 0.0
    fig, ax = plt.subplots(figsize=(3.38, 2.65))

    zones = (
        (derived.roche_fluid_m / derived.periapsis_m, "#d7ead7", "Roche limit (fluid)", 0.32),
        (derived.roche_rigid_m / derived.periapsis_m, "#eadfce", "Roche limit (rigid)", 0.38),
    )
    for width, color, label, alpha in zones:
        ax.axvspan(-width, width, color=color, alpha=alpha, label=label, zorder=0)

    ax.axvspan(
        -1.0,
        1.0,
        color="#eeeeee",
        alpha=0.55,
        label="Periapsis neighbourhood",
        zorder=0,
    )
    ax.semilogy(
        signed_coordinate[incoming],
        trajectory.tidal_modulation[incoming],
        color="#1769aa",
        label=r"Tidal coefficient, $\mathcal{M}$",
        zorder=3,
    )
    ax.semilogy(
        signed_coordinate[outgoing],
        trajectory.tidal_modulation[outgoing],
        color="#1769aa",
        zorder=3,
    )
    horizontal = (
        (1.0, "#d2601a", r"Elastic coefficient, $1$"),
        (derived.lambda_centrifugal, "#116149", r"Centrifugal coefficient, $\Lambda$"),
        (derived.xi_self_gravity, "#a32638", r"Self-gravity coefficient, $\Xi$"),
        (derived.chi**2, "#6b4c9a", r"Inertial coefficient, $\tau_e^2$"),
        (
            derived.coriolis_assembled_coefficient,
            "#4aa5d8",
            r"Coriolis coefficient, $\mathscr{C}_\tau$",
        ),
    )
    for value, color, label in horizontal:
        ax.axhline(value, color=color, linewidth=1.05, label=label, zorder=2)
    ax.axvline(0.0, color="#222222", linestyle=":", linewidth=1.1, zorder=4)
    ax.set_xlim(-config_value(trajectory.distance_over_periapsis), config_value(trajectory.distance_over_periapsis))
    ax.set_ylim(1.0e-8, 3.0)
    ax.set_xlabel(r"Signed encounter coordinate, $\mathrm{sgn}(t)r_T/\mathcal{R}_p$ [-]")
    ax.set_ylabel("Assembled coefficient magnitude [-]")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        fontsize=5.8,
        loc="lower left",
        ncol=1,
        borderpad=0.22,
        handlelength=1.0,
        labelspacing=0.18,
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_coefficient_bars(derived: Derived, path: Path) -> None:
    labels = (
        "Elastic",
        "Tide at periapsis",
        "Centrifugal",
        "Self-gravity",
        "Inertia",
        "Coriolis",
    )
    values = (
        1.0,
        1.0,
        derived.lambda_centrifugal,
        derived.xi_self_gravity,
        derived.chi**2,
        derived.coriolis_assembled_coefficient,
    )
    colors = ("#d2601a", "#1769aa", "#116149", "#a32638", "#6b4c9a", "#4aa5d8")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors, edgecolor="#333333", linewidth=0.7, width=0.72)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-8, 3.0)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("Assembled coefficient magnitude [-]")
    for position, value in zip(positions, values):
        ax.text(position, value * 1.25, f"{value:.2e}", ha="center", fontsize=8.2)
    fig.tight_layout()
    save_figure(fig, path)


def config_value(values: np.ndarray) -> float:
    return float(np.nanmax(np.abs(values)))


def plot_modal_diagnostics(
    modal_data: pd.DataFrame,
    derived: Derived,
    path: Path,
    time_limits_h: tuple[float, float],
) -> None:
    all_time_h = modal_data["time_s"].to_numpy() / 3600.0
    mask = (all_time_h >= time_limits_h[0]) & (all_time_h <= time_limits_h[1])
    time_h = all_time_h[mask]
    frame = modal_data.loc[mask]

    fig, axes = plt.subplots(4, 1, figsize=(3.38, 6.35), sharex=True)
    axes = axes.ravel()
    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#c8cdd2",
        "framealpha": 0.94,
        "fontsize": 6.25,
        "handlelength": 1.05,
        "handletextpad": 0.35,
        "borderpad": 0.24,
        "labelspacing": 0.2,
    }
    dense_legend_style = {
        **legend_style,
        "fontsize": 5.85,
        "labelspacing": 0.16,
    }

    def add_legend_headroom(axis, *series: np.ndarray, top_fraction: float) -> None:
        values = np.concatenate([np.asarray(item, dtype=float).ravel() for item in series])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        y_min = float(values.min())
        y_max = float(values.max())
        span = y_max - y_min
        if span <= 0.0:
            span = max(abs(y_max), 1.0)
        axis.set_ylim(y_min - 0.04 * span, y_max + top_fraction * span)

    axes[0].plot(
        time_h,
        frame["forcing_m_s2"],
        color="#0b5cad",
        linewidth=0.95,
        label=r"$\mathcal{F}_2(t)=GM_\oplus\ell_{\rm ast}/r_T(t)^3$",
    )
    add_legend_headroom(axes[0], frame["forcing_m_s2"], top_fraction=0.5)
    axes[0].set_ylabel(r"$\mathcal{F}_2$ [m s$^{-2}$]")
    axes[0].set_title("(a) Projected tidal forcing", loc="left")
    axes[0].legend(
        loc="upper right",
        **legend_style,
    )

    axes[1].plot(
        time_h,
        1.0e6 * frame["displacement_m"],
        color="#7b2d26",
        linewidth=0.95,
        label=r"$h_2:\ \ddot h_2+\omega_2^2h_2=\mathcal{F}_2$",
    )
    axes[1].plot(
        time_h,
        1.0e6 * frame["quasi_static_m"],
        "--",
        color="#2f3437",
        linewidth=0.85,
        label=r"$h_{2,\rm qs}=\mathcal{F}_2/\omega_2^2$",
    )
    axes[1].plot(
        time_h,
        1.0e6 * frame["adiabatic_second_order_m"],
        ":",
        color="#4d6f53",
        linewidth=0.9,
        label=r"$h_{2,\rm ad}^{(2)}=h_{2,\rm qs}-\ddot{\mathcal{F}}_2/\omega_2^4$",
    )
    add_legend_headroom(
        axes[1],
        1.0e6 * frame["displacement_m"],
        1.0e6 * frame["quasi_static_m"],
        1.0e6 * frame["adiabatic_second_order_m"],
        top_fraction=0.72,
    )
    axes[1].set_ylabel(r"$h_2$ [$\mu$m]")
    axes[1].set_title("(b) Quadrupolar displacement", loc="left")
    axes[1].legend(
        ncol=1,
        loc="upper right",
        **dense_legend_style,
    )

    axes[2].plot(
        time_h,
        1.0e6 * frame["amplitude_m"],
        color="#4f3d75",
        linewidth=0.95,
        label=r"$A_2=\sqrt{h_2^2+\dot h_2^2/\omega_2^2}$",
    )
    add_legend_headroom(axes[2], 1.0e6 * frame["amplitude_m"], top_fraction=0.48)
    axes[2].set_ylabel(r"Amplitude [$\mu$m]")
    axes[2].set_title("(c) Phase-space modal amplitude", loc="left")
    axes[2].legend(
        loc="upper right",
        **legend_style,
    )

    axes[3].plot(
        time_h,
        frame["modal_energy_j"],
        color="#8f6b32",
        linewidth=0.95,
        label=r"$\mathcal{E}_2=\frac{M_{\rm eff}}{2}(\dot h_2^2+\omega_2^2h_2^2)$",
    )
    add_legend_headroom(axes[3], frame["modal_energy_j"], top_fraction=0.5)
    axes[3].set_ylabel("Modal energy [J]")
    axes[3].set_title("(d) Physical modal energy", loc="left")
    axes[3].legend(
        loc="upper right",
        **legend_style,
    )

    for axis in axes:
        axis.axvline(0.0, color="#1f2933", linestyle=":", alpha=0.65, linewidth=0.85)
        axis.set_xlim(*time_limits_h)
    axes[3].set_xlabel("Time from periapsis [h]")
    if time_limits_h[1] <= 2.0:
        axes[3].set_xticks(np.arange(time_limits_h[0], time_limits_h[1] + 1.0e-9, 0.5))
    else:
        axes[3].set_xticks(np.arange(time_limits_h[0], time_limits_h[1] + 1.0e-9, 4.0))
    fig.tight_layout(h_pad=0.75)
    save_figure(fig, path)


def plot_encounter_regime(
    rr: np.ndarray,
    vv: np.ndarray,
    zeta: np.ndarray,
    config: BenchmarkConfig,
    parametric_cases: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    image = ax.pcolormesh(
        rr,
        vv / 1000.0,
        np.log10(zeta),
        shading="auto",
        cmap="viridis",
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}\zeta$")
    ax.axvline(
        fluid_roche_earth_radii(config),
        color="white",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
    )
    levels = [level for level in (0.1, 1.0, 10.0, 100.0, 1000.0) if zeta.min() <= level <= zeta.max()]
    if levels:
        contours = ax.contour(
            rr, vv / 1000.0, zeta, levels=levels, colors="white", linewidths=0.8
        )
        ax.clabel(contours, fmt=lambda value: f"{value:g}", fontsize=8)
    for row in parametric_cases.itertuples(index=False):
        label = "A" if row.case == "Nominal" else row.case
        ax.scatter(
            row.periapsis_earth_radii,
            row.asymptotic_speed_m_s / 1000.0,
            marker=row.marker,
            s=row.size,
            color=row.facecolor,
            edgecolor="#111827",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            label,
            (row.periapsis_earth_radii, row.asymptotic_speed_m_s / 1000.0),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.2,
            weight="bold",
        )
    ax.set_xlabel(r"Periapsis $\mathcal{R}_p/R_\oplus$ [-]")
    ax.set_ylabel(r"Asymptotic speed $v_\infty$ [km s$^{-1}$]")
    ax.set_title("(a) Encounter geometry", loc="left")
    fig.tight_layout()
    save_figure(fig, path)


def plot_stiffness_periapsis_regime(
    rr: np.ndarray,
    ee: np.ndarray,
    zeta: np.ndarray,
    config: BenchmarkConfig,
    parametric_cases: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    image = ax.pcolormesh(
        rr,
        ee,
        np.log10(zeta),
        shading="auto",
        cmap="viridis",
    )
    ax.set_yscale("log")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}\zeta$")
    ax.axvline(
        fluid_roche_earth_radii(config),
        color="white",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
    )
    levels = [level for level in (0.1, 1.0, 10.0, 100.0, 1000.0) if zeta.min() <= level <= zeta.max()]
    if levels:
        contours = ax.contour(rr, ee, zeta, levels=levels, colors="white", linewidths=0.8)
        ax.clabel(contours, fmt=lambda value: f"{value:g}", fontsize=8)
    for row in parametric_cases.itertuples(index=False):
        label = "A" if row.case == "Nominal" else row.case
        ax.scatter(
            row.periapsis_earth_radii,
            row.young_modulus_pa,
            marker=row.marker,
            s=row.size,
            color=row.facecolor,
            edgecolor="#111827",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            label,
            (row.periapsis_earth_radii, row.young_modulus_pa),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.2,
            weight="bold",
        )
    ax.set_xlabel(r"Periapsis $\mathcal{R}_p/R_\oplus$ [-]")
    ax.set_ylabel("Young's modulus [Pa]")
    ax.set_title("(b) Material stiffness", loc="left")
    fig.tight_layout()
    save_figure(fig, path)


def fluid_roche_earth_radii(config: BenchmarkConfig) -> float:
    density_ratio = (
        config.constants.earth_mean_density_kg_m3
        / config.asteroid.density_kg_m3
    )
    return 2.44 * density_ratio ** (1.0 / 3.0)


def plot_regime_maps(
    encounter_rr: np.ndarray,
    encounter_vv: np.ndarray,
    encounter_zeta: np.ndarray,
    stiffness_rr: np.ndarray,
    stiffness_ee: np.ndarray,
    stiffness_zeta: np.ndarray,
    config: BenchmarkConfig,
    parametric_cases: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.38, 6.05))
    roche = fluid_roche_earth_radii(config)

    image = axes[0].pcolormesh(
        encounter_rr,
        encounter_vv / 1000.0,
        np.log10(encounter_zeta),
        shading="auto",
        cmap="viridis",
    )
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.025)
    colorbar.set_label(r"$\log_{10}\zeta$")
    levels = [
        level
        for level in (10.0, 100.0, 1000.0, 10000.0)
        if encounter_zeta.min() <= level <= encounter_zeta.max()
    ]
    if levels:
        contours = axes[0].contour(
            encounter_rr,
            encounter_vv / 1000.0,
            encounter_zeta,
            levels=levels,
            colors="white",
            linewidths=0.65,
        )
        axes[0].clabel(contours, fmt=lambda value: f"{value:g}", fontsize=6.6)
    axes[0].axvline(roche, color="white", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel(r"Periapsis $\mathcal{R}_p/R_\oplus$ [-]")
    axes[0].set_ylabel(r"$v_\infty$ [km s$^{-1}$]")
    axes[0].set_title("(a) Encounter geometry", loc="left")

    image = axes[1].pcolormesh(
        stiffness_rr,
        stiffness_ee,
        np.log10(stiffness_zeta),
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_yscale("log")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.025)
    colorbar.set_label(r"$\log_{10}\zeta$")
    levels = [
        level
        for level in (10.0, 100.0, 1000.0, 10000.0)
        if stiffness_zeta.min() <= level <= stiffness_zeta.max()
    ]
    if levels:
        contours = axes[1].contour(
            stiffness_rr,
            stiffness_ee,
            stiffness_zeta,
            levels=levels,
            colors="white",
            linewidths=0.65,
        )
        axes[1].clabel(contours, fmt=lambda value: f"{value:g}", fontsize=6.6)
    axes[1].axvline(roche, color="white", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel(r"Periapsis $\mathcal{R}_p/R_\oplus$ [-]")
    axes[1].set_ylabel("Young's modulus [Pa]")
    axes[1].set_title("(b) Material stiffness", loc="left")

    for row in parametric_cases.itertuples(index=False):
        label = "A" if row.case == "Nominal" else row.case
        for axis, y_value in (
            (axes[0], row.asymptotic_speed_m_s / 1000.0),
            (axes[1], row.young_modulus_pa),
        ):
            axis.scatter(
                row.periapsis_earth_radii,
                y_value,
                marker=row.marker,
                s=0.62 * row.size,
                color=row.facecolor,
                edgecolor="#111827",
                linewidth=0.65,
                zorder=4,
            )
            axis.annotate(
                label,
                (row.periapsis_earth_radii, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.0,
                weight="bold",
            )

    fig.tight_layout(h_pad=1.0)
    save_figure(fig, path)


def plot_body_frame_forcing(periapsis: Trajectory, path: Path) -> None:
    time_h = periapsis.time_s / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.3))

    direction_labels = (
        r"$\hat r_x^{(b)}$",
        r"$\hat r_y^{(b)}$",
        r"$\hat r_z^{(b)}$",
    )
    direction_colors = ("#1b4965", "#7b2d26", "#5f6c7b")
    for index, (label, color) in enumerate(
        zip(direction_labels, direction_colors)
    ):
        axes[0].plot(
            time_h,
            periapsis.direction_body[:, index],
            label=label,
            color=color,
            linewidth=0.95,
        )
    axes[0].set_ylabel("Earth direction in body frame [-]")
    axes[0].set_title("(a) Perturber direction", loc="left")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.19),
        borderpad=0.25,
    )

    tensor = periapsis.tidal_tensor_body
    entries = (
        (0, 0, r"$T_{xx}$"),
        (0, 1, r"$T_{xy}$"),
        (0, 2, r"$T_{xz}$"),
        (1, 1, r"$T_{yy}$"),
        (1, 2, r"$T_{yz}$"),
        (2, 2, r"$T_{zz}$"),
    )
    colors = ("#183a59", "#8f6b32", "#5f6c7b", "#7b2d26", "#4d6f53", "#4f3d75")
    linestyles = ("-", "--", "-.", "-", "--", "-.")
    for (row, column, label), color, linestyle in zip(
        entries, colors, linestyles
    ):
        axes[1].plot(
            time_h,
            tensor[:, row, column],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=0.95,
        )
    axes[1].set_ylabel("Dimensionless tidal tensor")
    axes[1].set_title("(b) Independent tensor components", loc="left")
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.19),
        borderpad=0.25,
    )

    for axis in axes:
        axis.axvline(0.0, color="#1f2933", linestyle=":", alpha=0.7, linewidth=0.85)
        axis.set_xlabel("Time from periapsis [h]")
        axis.set_xlim(-1.5, 1.5)
    fig.tight_layout(w_pad=1.2, rect=(0.0, 0.0, 1.0, 0.82))
    save_figure(fig, path)


def plot_tidal_tensor(periapsis: Trajectory, path: Path) -> None:
    time_h = periapsis.time_s / 3600.0
    tensor = periapsis.tidal_tensor_body
    fig, ax = plt.subplots(figsize=(3.38, 2.55))
    entries = (
        (0, 0, r"$T_{xx}$"),
        (0, 1, r"$T_{xy}$"),
        (0, 2, r"$T_{xz}$"),
        (1, 1, r"$T_{yy}$"),
        (1, 2, r"$T_{yz}$"),
        (2, 2, r"$T_{zz}$"),
    )
    colors = ("#183a59", "#8f6b32", "#5f6c7b", "#7b2d26", "#4d6f53", "#4f3d75")
    linestyles = ("-", "--", "-.", "-", "--", "-.")
    for (row, column, label), color, linestyle in zip(entries, colors, linestyles):
        ax.plot(
            time_h,
            tensor[:, row, column],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.05,
        )
    ax.axvline(0.0, color="#1f2933", linestyle=":", alpha=0.7, linewidth=0.85)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("Time from periapsis [h]")
    ax.set_ylabel("Dimensionless tidal tensor")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#c8cdd2",
        framealpha=0.92,
        fontsize=6.0,
        ncol=3,
        loc="upper right",
        borderpad=0.22,
        handlelength=1.0,
        columnspacing=0.6,
        labelspacing=0.18,
    )
    fig.tight_layout()
    save_figure(fig, path)


def run_validation(
    config: BenchmarkConfig,
    derived: Derived,
    trajectory: Trajectory,
    periapsis: Trajectory,
    modal: ModalResponse,
    modal_coarse: ModalResponse,
) -> pd.DataFrame:
    tolerance = config.numerics.validation_relative_tolerance
    trace = np.trace(periapsis.tidal_tensor_body, axis1=1, axis2=2)
    asymmetry = periapsis.tidal_tensor_body - np.swapaxes(
        periapsis.tidal_tensor_body, 1, 2
    )
    center = len(periapsis.time_s) // 2
    modal_center = int(np.argmin(np.abs(modal.time_s)))
    coarse_center = int(np.argmin(np.abs(modal_coarse.time_s)))
    modal_convergence = abs(
        modal.displacement_m[modal_center]
        - modal_coarse.displacement_m[coarse_center]
    ) / max(abs(modal.displacement_m[modal_center]), 1.0e-30)
    symmetry_distance = float(
        np.max(np.abs(periapsis.distance_m - periapsis.distance_m[::-1]))
        / derived.periapsis_m
    )
    checks = [
        (
            "diameter_to_radius",
            abs(2.0 * derived.asteroid_radius_m - config.asteroid.diameter_m),
            tolerance * config.asteroid.diameter_m,
        ),
        (
            "periapsis_distance",
            abs(periapsis.distance_m[center] - derived.periapsis_m),
            tolerance * derived.periapsis_m,
        ),
        ("orbital_distance_symmetry", symmetry_distance, tolerance),
        ("tidal_tensor_trace", float(np.max(np.abs(trace))), 2.0e-12),
        (
            "tidal_tensor_symmetry",
            float(np.max(np.abs(asymmetry))),
            2.0e-12,
        ),
        (
            "modal_grid_convergence",
            modal_convergence,
            config.numerics.modal_convergence_relative_tolerance,
        ),
        (
            "positive_quadrupole_frequency",
            0.0 if derived.omega_quadrupole_rad_s > 0.0 else 1.0,
            0.0,
        ),
        (
            "nominal_outside_fluid_roche",
            0.0 if derived.periapsis_m > derived.roche_fluid_m else 1.0,
            0.0,
        ),
    ]
    rows = []
    for name, value, allowed in checks:
        passed = value <= allowed
        rows.append(
            {
                "check": name,
                "value": value,
                "allowed": allowed,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_tex_tables(
    path: Path,
    config: BenchmarkConfig,
    derived: Derived,
    stiffness_cases: pd.DataFrame,
) -> None:
    rows = [
        (
            "Mean diameter",
            r"$D_{\rm ast}$",
            f"{config.asteroid.diameter_m:.3f}",
            "m",
        ),
        (
            "Reference radius",
            r"$\ell_{\rm ast}=D_{\rm ast}/2$",
            f"{derived.asteroid_radius_m:.3f}",
            "m",
        ),
        ("Density", r"$\rho_0$", f"{config.asteroid.density_kg_m3:.6g}", r"kg\,m$^{-3}$"),
        ("Effective Young's modulus", "$E$", f"{config.asteroid.young_modulus_pa:.6e}", "Pa"),
        ("Effective Poisson's ratio", r"$\nu$", f"{config.asteroid.poisson_ratio:.6g}", "-"),
        ("Periapsis", r"$\mathcal{R}_p/R_\oplus$", f"{config.encounter.periapsis_earth_radii:.6g}", "-"),
        ("Asymptotic speed", r"$v_\infty$", f"{config.encounter.asymptotic_speed_m_s:.6g}", r"m\,s$^{-1}$"),
    ]
    lines = [
        "% Generated by run_benchmark.py. Do not edit numerical values manually.",
        r"\begin{tabular}{llll}",
        r"\hline",
        r"Input & Symbol & Value & Unit\\",
        r"\hline",
    ]
    lines.extend(f"{name} & {symbol} & {value} & {unit}\\\\" for name, symbol, value, unit in rows)
    lines.extend([r"\hline", r"\end{tabular}", "", r"\begin{tabular}{lrrrr}", r"\hline"])
    lines.append(r"Case & $E$ [Pa] & $\omega_2$ [rad/s] & $\zeta$ & $h_{2,\max}$ [m]\\")
    lines.append(r"\hline")
    for row in stiffness_cases.itertuples(index=False):
        lines.append(
            f"{row.case} & {row.young_modulus_pa:.6e} & "
            f"{row.omega_quadrupole_rad_s:.6e} & {row.zeta:.6e} & "
            f"{row.peak_dynamic_displacement_m:.6e}\\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(output_dir: Path, generated_files: list[Path]) -> None:
    source_files = [ROOT / "inputs.py", ROOT / "run_benchmark.py"]
    records = []
    for path in sorted(source_files + generated_files, key=lambda item: str(item)):
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "benchmark_id": CONFIG.benchmark_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "files": records,
    }
    write_json(output_dir / "manifest.json", manifest)


def export_paper_figures(
    source_figure_dir: Path,
    paper_root: Path,
    generated_files: list[Path],
) -> None:
    """Copy audited figures into the LaTeX figure tree with paper filenames."""
    destinations = {
        "coefficient_evolution.png": "figures/results/adim/admin_coeffs_evol_rt.png",
        "coefficient_bars.png": "figures/results/adim/barras_adim2.png",
        "nondimensional_hierarchy.png": "figures/results/adim/nondimensional_hierarchy.png",
        "trajectory_frames.png": "figures/results/trayectoria/trayectoria.png",
        "trajectory_true_anomaly.png": "figures/results/trayectoria/trajectory_periapsis_true_anomaly.png",
        "rotation_frame_components.png": "figures/results/trayectoria/rotation_frame_components.png",
        "body_frame_forcing.png": "figures/results/armonicos/body_frame_forcing.png",
        "modal_diagnostics_periapsis.png": "figures/results/armonicos/modal_diagnostics_periapsis.png",
        "modal_diagnostics_extended.png": "figures/results/armonicos/modal_diagnostics_extended.png",
        "tidal_tensor_body_frame.png": "figures/results/armonicos/tidal_tensor_body_frame.png",
        "regime_maps.png": "figures/results/armonicos/regime_maps.png",
        "encounter_regime_map.png": "figures/results/armonicos/encounter_regime_map.png",
        "stiffness_periapsis_regime_map.png": "figures/results/armonicos/stiffness_periapsis_regime_map.png",
    }
    for source_name, destination_name in destinations.items():
        source = source_figure_dir / source_name
        destination = paper_root / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        generated_files.append(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete tidal-deformation reference benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_outputs",
        help="Output directory. Default: benchmark_outputs beside the paper.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller grids for a fast smoke test.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compute and export numerical data without figures.",
    )
    return parser.parse_args(argv)


def run(config: BenchmarkConfig, args: argparse.Namespace) -> Path:
    config.validate()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in (output_dir, data_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.quick:
        numerics = replace(
            config.numerics,
            trajectory_points=10001,
            periapsis_points=8001,
            modal_points=20001,
        )
        config = replace(config, numerics=numerics)
        config.validate()

    print("[1/7] Computing canonical inputs and derived groups")
    derived = compute_derived(config)
    input_snapshot = config.as_dict()
    write_json(output_dir / "inputs_snapshot.json", input_snapshot)
    pd.DataFrame(flatten_mapping(input_snapshot)).to_csv(
        table_dir / "input_parameters.csv",
        index=False,
    )
    derived_dataframe(derived).to_csv(
        table_dir / "derived_quantities.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )

    print("[2/7] Computing hyperbolic trajectories and body-frame tensor")
    trajectory = compute_trajectory(
        config,
        derived,
        max_periapsis_radii=config.numerics.trajectory_max_periapsis_radii,
        points=config.numerics.trajectory_points,
    )
    periapsis = compute_periapsis_trajectory(config, derived)
    modal_trajectory = compute_trajectory(
        config,
        derived,
        max_periapsis_radii=config.numerics.modal_max_periapsis_radii,
        points=config.numerics.modal_points,
    )
    trajectory_dataframe(trajectory).to_csv(
        data_dir / "trajectory_full.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    trajectory_dataframe(periapsis).to_csv(
        data_dir / "trajectory_periapsis.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    coefficient_dataframe(trajectory, derived).to_csv(
        data_dir / "nondimensional_coefficients_full.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    coefficient_dataframe(periapsis, derived).to_csv(
        data_dir / "nondimensional_coefficients_periapsis.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )

    print("[3/7] Solving the dimensional quadrupolar response")
    modal = compute_modal_response(config, derived, modal_trajectory)
    modal_full = modal_response_dataframe(modal, derived)
    modal_peri = interpolate_modal_to_periapsis(modal, periapsis, derived)
    modal_full.to_csv(
        data_dir / "modal_response_full.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    modal_peri.to_csv(
        data_dir / "modal_response_periapsis.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )

    coarse_points = max(5001, (config.numerics.modal_points // 2) | 1)
    modal_coarse_trajectory = compute_trajectory(
        config,
        derived,
        max_periapsis_radii=config.numerics.modal_max_periapsis_radii,
        points=coarse_points,
    )
    modal_coarse = compute_modal_response(config, derived, modal_coarse_trajectory)

    print("[4/7] Computing reduced-stiffness cases and regime maps")
    stiffness_cases, stiffness_parametric = stiffness_sweep(
        config, modal_trajectory
    )
    parametric_cases = parametric_benchmark_cases(config)
    stiffness_cases.to_csv(
        table_dir / "stiffness_cases.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    parametric_cases.to_csv(
        table_dir / "parametric_cases.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    stiffness_parametric.to_csv(
        data_dir / "stiffness_sweep.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    encounter_rr, encounter_vv, encounter_zeta = encounter_regime_grid(
        config, derived
    )
    stiffness_rr, stiffness_ee, stiffness_zeta = (
        stiffness_periapsis_regime_grid(config)
    )
    pd.DataFrame(
        {
            "periapsis_earth_radii": encounter_rr.ravel(),
            "asymptotic_speed_m_s": encounter_vv.ravel(),
            "zeta": encounter_zeta.ravel(),
        }
    ).to_csv(
        data_dir / "encounter_regime_map.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    pd.DataFrame(
        {
            "periapsis_earth_radii": stiffness_rr.ravel(),
            "young_modulus_pa": stiffness_ee.ravel(),
            "zeta": stiffness_zeta.ravel(),
        }
    ).to_csv(
        data_dir / "stiffness_periapsis_regime_map.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )

    print("[5/7] Running numerical and physical validation checks")
    validation = run_validation(
        config, derived, trajectory, periapsis, modal, modal_coarse
    )
    validation.to_csv(
        table_dir / "validation_checks.csv",
        index=False,
        float_format=config.numerics.csv_float_format,
    )
    failed = validation.loc[~validation["passed"]]

    nominal_peak = float(np.max(np.abs(modal.displacement_m)))
    nominal_qs_peak = float(np.max(np.abs(modal.quasi_static_m)))
    summary = {
        "benchmark_id": config.benchmark_id,
        "geometry": {
            "diameter_m": config.asteroid.diameter_m,
            "radius_m": derived.asteroid_radius_m,
            "radius_definition": "diameter_m / 2",
        },
        "encounter": {
            "periapsis_m": derived.periapsis_m,
            "periapsis_earth_radii": config.encounter.periapsis_earth_radii,
            "asymptotic_speed_m_s": config.encounter.asymptotic_speed_m_s,
            "periapsis_speed_m_s": derived.periapsis_speed_m_s,
            "eccentricity": derived.eccentricity,
            "encounter_time_s": derived.modal_encounter_time_s,
        },
        "dimensionless": {
            "eta": derived.deformation_ratio_eta,
            "chi": derived.chi,
            "xi": derived.xi_self_gravity,
            "lambda": derived.lambda_centrifugal,
            "coriolis_group": derived.coriolis_group,
            "coriolis_assembled_coefficient": derived.coriolis_assembled_coefficient,
            "euler_group": derived.euler_group,
            "zeta": derived.zeta,
        },
        "quadrupole": {
            "omega_elastic_rad_s": derived.omega_elastic_rad_s,
            "omega_self_gravity_rad_s": derived.omega_self_gravity_rad_s,
            "omega_rotation_rad_s": derived.omega_rotation_rad_s,
            "omega_total_rad_s": derived.omega_quadrupole_rad_s,
            "period_s": derived.quadrupole_period_s,
            "forcing_projection_factor": config.modal_forcing_projection_factor,
            "peak_dynamic_displacement_m": nominal_peak,
            "peak_dynamic_displacement_over_radius": (
                nominal_peak / derived.asteroid_radius_m
            ),
            "peak_dynamic_displacement_over_Uc": (
                nominal_peak / derived.displacement_scale_m
            ),
            "peak_quasi_static_displacement_m": nominal_qs_peak,
            "peak_modal_energy_j": float(np.max(modal.modal_energy_j)),
        },
        "validation": {
            "passed": bool(failed.empty),
            "failed_checks": failed["check"].tolist(),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_tex_tables(
        table_dir / "benchmark_tables.tex",
        config,
        derived,
        stiffness_cases,
    )

    if not args.no_plots:
        print("[6/7] Rendering reproducible benchmark figures")
        style_plots(config.numerics.figure_dpi)
        plot_coefficient_evolution(
            trajectory, derived, figure_dir / "coefficient_evolution.png"
        )
        plot_coefficient_bars(derived, figure_dir / "coefficient_bars.png")
        plot_trajectory(trajectory, derived, figure_dir / "trajectory_frames.png")
        plot_true_anomaly(
            periapsis, derived, figure_dir / "trajectory_true_anomaly.png"
        )
        plot_rotation_components(
            periapsis, figure_dir / "rotation_frame_components.png"
        )
        plot_nondimensional_hierarchy(
            trajectory, derived, figure_dir / "nondimensional_hierarchy.png"
        )
        plot_body_frame_forcing(
            periapsis, figure_dir / "body_frame_forcing.png"
        )
        plot_modal_diagnostics(
            modal_peri,
            derived,
            figure_dir / "modal_diagnostics_periapsis.png",
            (-1.5, 1.5),
        )
        plot_modal_diagnostics(
            modal_full,
            derived,
            figure_dir / "modal_diagnostics_extended.png",
            (-8.0, 8.0),
        )
        plot_regime_maps(
            encounter_rr,
            encounter_vv,
            encounter_zeta,
            stiffness_rr,
            stiffness_ee,
            stiffness_zeta,
            config,
            parametric_cases,
            figure_dir / "regime_maps.png",
        )
        plot_encounter_regime(
            encounter_rr,
            encounter_vv,
            encounter_zeta,
            config,
            parametric_cases,
            figure_dir / "encounter_regime_map.png",
        )
        plot_stiffness_periapsis_regime(
            stiffness_rr,
            stiffness_ee,
            stiffness_zeta,
            config,
            parametric_cases,
            figure_dir / "stiffness_periapsis_regime_map.png",
        )
        plot_tidal_tensor(
            periapsis, figure_dir / "tidal_tensor_body_frame.png"
        )

    print("[7/7] Writing provenance manifest")
    generated_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    if not args.no_plots:
        export_paper_figures(figure_dir, ROOT, generated_files)
    write_manifest(output_dir, generated_files)

    if not failed.empty:
        names = ", ".join(failed["check"].tolist())
        raise RuntimeError(f"Benchmark validation failed: {names}")

    print()
    print("Benchmark completed successfully.")
    print(f"Output directory: {output_dir}")
    print(f"Reference radius: {derived.asteroid_radius_m:.3f} m")
    print(f"Quadrupole frequency: {derived.omega_quadrupole_rad_s:.9e} rad/s")
    print(f"Quadrupole period: {derived.quadrupole_period_s:.9e} s")
    print(f"Peak dynamic displacement: {nominal_peak:.9e} m")
    print(f"All {len(validation)} validation checks passed.")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(CONFIG, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
