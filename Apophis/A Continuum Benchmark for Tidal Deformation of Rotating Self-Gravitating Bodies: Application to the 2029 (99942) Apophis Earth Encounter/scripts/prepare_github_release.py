"""Build a clean GitHub-ready supplementary repository snapshot.

The script collects the manuscript figures, benchmark data, current scripts,
and generated supplementary visual material into ``github_release/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASE = ROOT / "github_release"

TITLE = (
    "A Continuum Benchmark for Tidal Deformation of Rotating "
    "Self-Gravitating Bodies: Application to the 2029 (99942) Apophis "
    "Earth Encounter"
)
AUTHORS = [
    "Francisco Cruz Perez",
    "Isabel Herreros Cid",
    "Fernando Veiga Lopez",
]
AFFILIATIONS = [
    "Universidad Carlos III de Madrid",
    "Centro de Astrobiologia, CSIC-INTA",
    "Universidad de Vigo",
]
REPOSITORY_URL = (
    "https://github.com/cruzfran252-ux/papers/tree/"
    "0ba7ea234b0d28a1c58c644ab1dc7460d5756a06/Apophis/"
    "A%20Continuum%20Benchmark%20for%20Tidal%20Deformation%20of%20Rotating"
    "%20Self-Gravitating%20Bodies%3A%20Application%20to%20the%202029%20"
    "(99942)%20Apophis%20Earth%20Encounter"
)
ABSTRACT = (
    "Close planetary flybys provide natural loading experiments for weak "
    "small bodies, but comparisons between deformation models remain "
    "ambiguous without a common mathematical reference problem. The 2029 "
    "Earth encounter of (99942) Apophis, together with the observational "
    "context provided by RAMSES and OSIRIS-APEX, motivates a reproducible "
    "benchmark for tidal deformation during the 2029 close approach. This "
    "paper defines a continuum benchmark for a rotating, self-gravitating "
    "deformable body subjected to a prescribed external tidal field and then "
    "specializes it to the Apophis encounter. The general statement admits "
    "arbitrary body geometry, density distribution, rotation axis, "
    "constitutive closure, perturber, and prescribed relative-position "
    "history. The reference realization fixes small-strain isotropic "
    "elasticity, a traction-free homogeneous sphere, a hyperbolic "
    "trajectory, and the leading quadrupolar tide. Dimensional equations, "
    "nondimensional coefficients, ordered perturbations, body-frame forcing, "
    "a quadrupolar modal approximation, and reproducible numerical targets "
    "are reported together. The Apophis case illustrates a quasi-static "
    "elastic-tidal regime and a controlled set of stiffness and encounter "
    "variations for analytical, continuum, reduced-order, and granular "
    "implementations."
)


@dataclass
class Asset:
    label: str
    path: str
    description: str
    section: str
    source: str = "generated"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.05,
        }
    )


def ensure_clean_release(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if not str(resolved).startswith(str(root)):
        raise ValueError(f"Refusing to write outside the workspace: {resolved}")
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(
                f"{resolved} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_optional(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    copy_file(source, destination)
    return True


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_indices(length: int, frames: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, frames, dtype=int))


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "trajectory_full": pd.read_csv(ROOT / "benchmark_outputs/data/trajectory_full.csv"),
        "trajectory": pd.read_csv(ROOT / "benchmark_outputs/data/trajectory_periapsis.csv"),
        "modal": pd.read_csv(ROOT / "benchmark_outputs/data/modal_response_periapsis.csv"),
        "stiffness": pd.read_csv(ROOT / "benchmark_outputs/tables/stiffness_cases.csv"),
        "parametric": pd.read_csv(ROOT / "benchmark_outputs/tables/parametric_cases.csv"),
        "summary": pd.json_normalize(
            json.loads((ROOT / "benchmark_outputs/summary.json").read_text())
        ),
    }


def tensor_matrix(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [row["tidal_tensor_xx"], row["tidal_tensor_xy"], row["tidal_tensor_xz"]],
            [row["tidal_tensor_xy"], row["tidal_tensor_yy"], row["tidal_tensor_yz"]],
            [row["tidal_tensor_xz"], row["tidal_tensor_yz"], row["tidal_tensor_zz"]],
        ],
        dtype=float,
    )


def tensor_stack(df: pd.DataFrame) -> np.ndarray:
    return np.stack([tensor_matrix(row) for _, row in df.iterrows()])


def generate_quadrupole_gallery(out: Path) -> Asset:
    theta = np.linspace(0.0, np.pi, 120)
    phi = np.linspace(0.0, 2.0 * np.pi, 180)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x0 = np.sin(theta_grid) * np.cos(phi_grid)
    y0 = np.sin(theta_grid) * np.sin(phi_grid)
    z0 = np.cos(theta_grid)

    modes = [
        (r"(a) $Y_{20}$ axisymmetric", 0.5 * (3.0 * z0**2 - 1.0), 22, -45),
        (r"(b) $Y_{22}^{c}$ sectoral", (1.0 - z0**2) * np.cos(2.0 * phi_grid), 22, -38),
        (r"(c) $Y_{21}^{c}$ tesseral", x0 * z0, 21, -50),
        (r"(d) tide-aligned $P_2(\hat r_x)$", 0.5 * (3.0 * x0**2 - 1.0), 22, -38),
    ]

    fig = plt.figure(figsize=(7.2, 5.25))
    positions = [
        (0.01, 0.52, 0.48, 0.40),
        (0.51, 0.52, 0.48, 0.40),
        (0.01, 0.06, 0.48, 0.40),
        (0.51, 0.06, 0.48, 0.40),
    ]
    for idx, (name, field, elev, azim) in enumerate(modes, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.set_position(positions[idx - 1])
        norm = np.max(np.abs(field))
        radial = 1.0 + 0.22 * field / norm
        colors = plt.cm.RdBu_r((field / norm + 1.0) * 0.5)
        ax.plot_surface(
            radial * x0,
            radial * y0,
            radial * z0,
            facecolors=colors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        ax.set_title(name, pad=0, fontsize=9.6, loc="left")
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((1, 1, 1), zoom=1.72)
        except TypeError:
            ax.set_box_aspect((1, 1, 1))
        lim = 1.22
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
    fig.suptitle(
        r"Quadrupolar $\ell=2$ surface basis and tide-aligned forcing",
        y=0.985,
        fontsize=12.2,
    )
    save_figure(fig, out)
    return Asset(
        "Extended Figure E5",
        "figures/extended/figure_E5_quadrupole_mode_gallery.png",
        "Gallery of real l=2 deformation patterns and the tide-aligned quadrupole.",
        "Quadrupolar modal interpretation",
    )


def generate_tensor_eigenstructure(traj: pd.DataFrame, out: Path) -> Asset:
    hours = traj["time_s"].to_numpy() / 3600.0
    tensors = tensor_stack(traj)
    eigvals = np.linalg.eigvalsh(tensors)
    trace = np.trace(tensors, axis1=1, axis2=2)
    frob = np.linalg.norm(tensors, axis=(1, 2))
    center_idx = int(np.argmin(np.abs(traj["time_s"].to_numpy())))
    center = tensors[center_idx]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.5, 5.35),
        gridspec_kw={"width_ratios": [0.96, 1.08], "height_ratios": [1.0, 1.0]},
    )
    ax = axes[0, 0]
    vmax = np.max(np.abs(center))
    im = ax.imshow(center, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([0, 1, 2], labels=[r"$x$", r"$y$", r"$z$"])
    ax.set_yticks([0, 1, 2], labels=[r"$x$", r"$y$", r"$z$"])
    ax.set_title(r"(a) Periapsis tensor $T_{ij}$", loc="left")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{center[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, label=r"normalized $T_{ij}$")

    ax = axes[0, 1]
    ax.plot(hours, traj["tidal_tensor_xx"], label=r"$T_{xx}$")
    ax.plot(hours, traj["tidal_tensor_yy"], label=r"$T_{yy}$")
    ax.plot(hours, traj["tidal_tensor_zz"], label=r"$T_{zz}$")
    ax.plot(hours, traj["tidal_tensor_xy"], label=r"$T_{xy}$", alpha=0.82)
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_title(r"(b) Independent tensor components", loc="left")
    ax.set_xlabel(r"time from periapsis, $t-t_p$ [h]")
    ax.set_ylabel(r"component $T_{ij}$")
    ax.legend(ncol=2, frameon=False)

    ax = axes[1, 0]
    colors = ["#243b6b", "#7d8d9c", "#a22c29"]
    for i, color in enumerate(colors):
        ax.plot(hours, eigvals[:, i], color=color, label=rf"$\lambda_{i + 1}$")
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_title(r"(c) Principal tidal eigenvalues", loc="left")
    ax.set_xlabel(r"time from periapsis, $t-t_p$ [h]")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(hours, np.abs(trace), color="#a22c29", label=r"$|\mathrm{tr}\,\mathbf{T}|$")
    ax.plot(hours, frob, color="#244f46", label=r"$\|\mathbf{T}\|_F$")
    ax.set_yscale("log")
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_title(r"(d) Trace-free audit", loc="left")
    ax.set_xlabel(r"time from periapsis, $t-t_p$ [h]")
    ax.set_ylabel("diagnostic magnitude")
    ax.legend(frameon=False)

    fig.suptitle(r"Body-frame tidal tensor: eigenstructure and trace-free checks", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965], pad=0.45, w_pad=0.8, h_pad=0.85)
    save_figure(fig, out)
    return Asset(
        "Extended Figure E6",
        "figures/extended/figure_E6_tidal_tensor_eigenstructure.png",
        "Tensor matrix, component histories, eigenvalues, and trace-free validation.",
        "Tidal-tensor diagnostics",
    )


def generate_response_scaling(
    stiffness: pd.DataFrame, parametric: pd.DataFrame, out: Path
) -> Asset:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))

    ax = axes[0, 0]
    ax.loglog(
        stiffness["young_modulus_pa"],
        stiffness["peak_dynamic_displacement_m"],
        "o-",
        color="#a22c29",
        label="dynamic peak",
    )
    ax.loglog(
        stiffness["young_modulus_pa"],
        stiffness["peak_quasi_static_displacement_m"],
        "s--",
        color="#244f46",
        label="quasi-static peak",
    )
    ax.invert_xaxis()
    ax.set_title("Stiffness sweep displacement")
    ax.set_xlabel("Young modulus [Pa]")
    ax.set_ylabel("peak displacement [m]")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.loglog(
        stiffness["young_modulus_pa"],
        stiffness["zeta"],
        "o-",
        color="#243b6b",
    )
    ax.axhline(1.0, color="#a22c29", lw=0.9, ls="--")
    ax.invert_xaxis()
    ax.set_title("Regime parameter")
    ax.set_xlabel("Young modulus [Pa]")
    ax.set_ylabel(r"$\zeta=\omega_2 t_{\rm enc}$")

    ax = axes[1, 0]
    ratio = (
        stiffness["peak_dynamic_displacement_m"]
        / stiffness["peak_quasi_static_displacement_m"]
    )
    excess_ppm = 1.0e6 * (ratio - 1.0)
    ax.semilogx(stiffness["young_modulus_pa"], excess_ppm, "o-", color="#6f5e2e")
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.invert_xaxis()
    ax.set_title("Dynamic excess over quasi-static peak")
    ax.set_xlabel("Young modulus [Pa]")
    ax.set_ylabel("relative excess [ppm]")

    ax = axes[1, 1]
    labels = parametric["case"].to_list()
    peaks = parametric["peak_dynamic_displacement_m"].to_numpy()
    colors = ["#a22c29"] + ["#6a8f86", "#c38f3d", "#537aa5", "#8d6a9f"]
    ax.bar(labels, peaks, color=colors[: len(labels)], edgecolor="0.2", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_title("Nominal and factorial cases")
    ax.set_ylabel("peak dynamic displacement [m]")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Quadrupolar response scaling across benchmark extensions", y=0.99)
    fig.tight_layout()
    save_figure(fig, out)
    return Asset(
        "Extended Figure E7",
        "figures/extended/figure_E7_response_scaling.png",
        "Peak response, zeta, and quasi-static agreement for stiffness and factorial cases.",
        "Benchmark extension cases",
    )


def generate_body_frame_story(traj: pd.DataFrame, out: Path) -> Asset:
    hours = traj["time_s"].to_numpy() / 3600.0
    rh = traj[["rhat_body_x", "rhat_body_y", "rhat_body_z"]].to_numpy()
    tensors = tensor_stack(traj)
    angle = 0.5 * np.arctan2(
        2.0 * traj["tidal_tensor_xy"].to_numpy(),
        traj["tidal_tensor_xx"].to_numpy() - traj["tidal_tensor_yy"].to_numpy(),
    )
    norm = np.linalg.norm(tensors, axis=(1, 2))

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))

    ax = axes[0, 0]
    sc = ax.scatter(rh[:, 0], rh[:, 1], c=hours, s=2, cmap="viridis", rasterized=True)
    unit = plt.Circle((0, 0), 1.0, edgecolor="0.45", facecolor="none", lw=0.8)
    ax.add_patch(unit)
    for time_h in [-1.0, 0.0, 1.0]:
        idx = int(np.argmin(np.abs(hours - time_h)))
        ax.plot([0, rh[idx, 0]], [0, rh[idx, 1]], lw=1.0, label=f"{time_h:+.0f} h")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel(r"$\hat r_x$")
    ax.set_ylabel(r"$\hat r_y$")
    ax.set_title("Earth direction in the body xy plane")
    ax.legend(frameon=False, loc="lower left")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="time [h]")

    ax = axes[0, 1]
    ax.plot(hours, rh[:, 0], label=r"$\hat r_x$")
    ax.plot(hours, rh[:, 1], label=r"$\hat r_y$")
    ax.plot(hours, rh[:, 2], label=r"$\hat r_z$")
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_title("Rotating-frame direction cosines")
    ax.set_xlabel("time from periapsis [h]")
    ax.set_ylabel("component")
    ax.legend(ncol=3, frameon=False)

    ax = axes[1, 0]
    ax.plot(hours, np.rad2deg(angle), color="#a22c29")
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_title("Major principal-axis orientation")
    ax.set_xlabel("time from periapsis [h]")
    ax.set_ylabel("angle in xy plane [deg]")

    ax = axes[1, 1]
    ax.plot(hours, norm / np.max(norm), color="#244f46", label="tensor norm")
    ax.plot(hours, traj["tidal_modulation"], color="#243b6b", ls="--", label=r"$r_p^3/r^3$")
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_title("Tensor intensity")
    ax.set_xlabel("time from periapsis [h]")
    ax.set_ylabel("normalized magnitude")
    ax.legend(frameon=False)

    fig.suptitle("Geometry of the body-frame tidal forcing", y=0.99)
    fig.tight_layout()
    save_figure(fig, out)
    return Asset(
        "Extended Figure E8",
        "figures/extended/figure_E8_body_frame_tidal_geometry.png",
        "Direction cosines, tensor intensity, and principal-axis evolution.",
        "Body-frame forcing geometry",
    )


def animate_body_frame_direction(traj: pd.DataFrame, out: Path) -> Asset:
    frames = sample_indices(len(traj), 120)
    hours = traj["time_s"].to_numpy() / 3600.0
    rh = traj[["rhat_body_x", "rhat_body_y", "rhat_body_z"]].to_numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    circle = plt.Circle((0, 0), 1.0, edgecolor="0.45", facecolor="none", lw=0.8)
    ax0.add_patch(circle)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlim(-1.08, 1.08)
    ax0.set_ylim(-1.08, 1.08)
    ax0.set_xlabel(r"$\hat r_x$")
    ax0.set_ylabel(r"$\hat r_y$")
    ax0.set_title("Earth direction in rotating frame")
    (trail,) = ax0.plot([], [], color="#6a8f86", lw=1.0)
    (vec,) = ax0.plot([], [], color="#a22c29", lw=2.0)
    point = ax0.scatter([], [], s=28, color="#a22c29", zorder=5)

    ax1.plot(hours, rh[:, 0], color="#a22c29", lw=0.9, label=r"$\hat r_x$")
    ax1.plot(hours, rh[:, 1], color="#243b6b", lw=0.9, label=r"$\hat r_y$")
    ax1.plot(hours, rh[:, 2], color="#244f46", lw=0.9, label=r"$\hat r_z$")
    marker = ax1.axvline(hours[0], color="0.15", lw=1.0)
    ax1.set_xlabel("time from periapsis [h]")
    ax1.set_ylabel("component")
    ax1.set_title("Direction cosines")
    ax1.legend(ncol=3, frameon=False)
    fig.tight_layout()

    def update(frame_index: int) -> tuple:
        idx = frames[frame_index]
        start = max(0, idx - 500)
        trail.set_data(rh[start:idx + 1, 0], rh[start:idx + 1, 1])
        vec.set_data([0.0, rh[idx, 0]], [0.0, rh[idx, 1]])
        point.set_offsets([[rh[idx, 0], rh[idx, 1]]])
        marker.set_xdata([hours[idx], hours[idx]])
        return trail, vec, point, marker

    out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out, writer=PillowWriter(fps=18))
    plt.close(fig)
    return Asset(
        "Animation 1",
        "animations/animation_01_body_frame_direction.gif",
        "Evolution of the Earth direction vector in the rotating benchmark frame.",
        "GIF",
    )


def animate_modal_response(modal: pd.DataFrame, out: Path) -> Asset:
    frames = sample_indices(len(modal), 120)
    hours = modal["time_s"].to_numpy() / 3600.0
    forcing = modal["forcing_m_s2"].to_numpy()
    displacement = modal["displacement_m"].to_numpy()
    quasi = modal["quasi_static_m"].to_numpy()
    forcing_norm = forcing / np.max(np.abs(forcing))
    disp_norm = displacement / np.max(np.abs(quasi))
    quasi_norm = quasi / np.max(np.abs(quasi))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(hours, forcing_norm, color="#243b6b", lw=1.0, label="forcing")
    ax.plot(hours, disp_norm, color="#a22c29", lw=1.1, label="dynamic response")
    ax.plot(hours, quasi_norm, color="#244f46", lw=0.9, ls="--", label="quasi-static")
    marker = ax.axvline(hours[0], color="0.15", lw=1.0)
    dot1 = ax.scatter([], [], color="#243b6b", s=25, zorder=5)
    dot2 = ax.scatter([], [], color="#a22c29", s=25, zorder=5)
    ax.set_xlabel("time from periapsis [h]")
    ax.set_ylabel("normalized quantity")
    ax.set_title("Quadrupolar forcing and response through periapsis")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    def update(frame_index: int) -> tuple:
        idx = frames[frame_index]
        marker.set_xdata([hours[idx], hours[idx]])
        dot1.set_offsets([[hours[idx], forcing_norm[idx]]])
        dot2.set_offsets([[hours[idx], disp_norm[idx]]])
        return marker, dot1, dot2

    out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out, writer=PillowWriter(fps=18))
    plt.close(fig)
    return Asset(
        "Animation 2",
        "animations/animation_02_quadrupole_forcing_response.gif",
        "Synchronized marker over tidal forcing, dynamic response, and quasi-static response.",
        "GIF",
    )


def animate_tensor_principal_axes(traj: pd.DataFrame, out: Path) -> Asset:
    frames = sample_indices(len(traj), 110)
    hours = traj["time_s"].to_numpy() / 3600.0
    tensors = tensor_stack(traj)
    norm = np.linalg.norm(tensors, axis=(1, 2))
    max_norm = float(np.max(norm))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlim(-1.15, 1.15)
    ax0.set_ylim(-1.15, 1.15)
    ax0.set_xlabel("body x")
    ax0.set_ylabel("body y")
    ax0.set_title("Principal stretch/compression axes")
    circle = plt.Circle((0, 0), 1.0, edgecolor="0.65", facecolor="none", lw=0.8)
    ax0.add_patch(circle)
    stretch = ax0.quiver(
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#a22c29",
        width=0.012,
    )
    compress = ax0.quiver(
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#243b6b",
        width=0.012,
    )
    time_text = ax0.text(0.02, 0.95, "", transform=ax0.transAxes, va="top")

    ax1.plot(hours, norm / max_norm, color="#244f46", lw=1.0)
    marker = ax1.axvline(hours[0], color="0.15", lw=1.0)
    ax1.set_xlabel("time from periapsis [h]")
    ax1.set_ylabel("normalized norm")
    ax1.set_title("Tidal-tensor intensity")
    fig.tight_layout()

    def update(frame_index: int) -> tuple:
        idx = frames[frame_index]
        vals, vecs = np.linalg.eigh(tensors[idx, :2, :2])
        order = np.argsort(vals)
        vmin = vecs[:, order[0]]
        vmax = vecs[:, order[-1]]
        scale = 0.3 + 0.65 * norm[idx] / max_norm
        stretch.set_UVC([scale * vmax[0]], [scale * vmax[1]])
        compress.set_UVC([scale * vmin[0]], [scale * vmin[1]])
        marker.set_xdata([hours[idx], hours[idx]])
        time_text.set_text(f"t = {hours[idx]:+.2f} h")
        return stretch, compress, marker, time_text

    out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out, writer=PillowWriter(fps=18))
    plt.close(fig)
    return Asset(
        "Animation 3",
        "animations/animation_03_tidal_tensor_principal_axes.gif",
        "Animated principal stretching and compression axes of the body-frame tidal tensor.",
        "GIF",
    )


def animate_apophis_flyby_trajectory(traj: pd.DataFrame, out: Path) -> Asset:
    frames = sample_indices(len(traj), 130)
    hours = traj["time_s"].to_numpy() / 3600.0
    x_m = traj["x_earth_m"].to_numpy() / 1.0e6
    y_m = traj["y_earth_m"].to_numpy() / 1.0e6
    tidal = traj["tidal_modulation"].to_numpy()
    earth_radius_m = 6.371
    periapsis_idx = int(np.argmin(traj["distance_m"].to_numpy()))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.55), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax0.plot(x_m, y_m, color="#9fb5c9", lw=0.8)
    earth = plt.Circle((0.0, 0.0), earth_radius_m, facecolor="#406b8f", edgecolor="#1f3447", lw=0.7, alpha=0.95)
    ax0.add_patch(earth)
    ax0.plot(x_m[periapsis_idx], y_m[periapsis_idx], marker="*", ms=8, color="#c43b2f")
    ax0.annotate(
        "periapsis",
        xy=(x_m[periapsis_idx], y_m[periapsis_idx]),
        xytext=(7, -10),
        textcoords="offset points",
        fontsize=7,
        color="#8f2b25",
        ha="left",
        va="top",
    )
    trail, = ax0.plot([], [], color="#a22c29", lw=1.8)
    marker = ax0.scatter([], [], s=32, color="#a22c29", zorder=5)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel(r"$x$ [Mm]")
    ax0.set_ylabel(r"$y$ [Mm]")
    ax0.set_title(r"(a) Earth-centered Apophis flyby", loc="left")
    margin = 18.0
    ax0.set_xlim(float(np.min(x_m)) - margin, float(np.max(x_m)) + margin)
    ax0.set_ylim(float(np.min(y_m)) - margin, float(np.max(y_m)) + margin)

    ax1.plot(hours, tidal, color="#244f46", lw=1.0)
    time_line = ax1.axvline(hours[0], color="#a22c29", lw=1.1)
    dot = ax1.scatter([], [], s=26, color="#a22c29", zorder=4)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"time from periapsis, $t-t_p$ [h]")
    ax1.set_ylabel(r"$r_p^3/r^3$")
    ax1.set_title(r"(b) Tidal intensity", loc="left")
    fig.tight_layout()

    def update(frame_index: int) -> tuple:
        idx = frames[frame_index]
        start = max(0, idx - 900)
        trail.set_data(x_m[start:idx + 1], y_m[start:idx + 1])
        marker.set_offsets([[x_m[idx], y_m[idx]]])
        time_line.set_xdata([hours[idx], hours[idx]])
        dot.set_offsets([[hours[idx], tidal[idx]]])
        return trail, marker, time_line, dot

    out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out, writer=PillowWriter(fps=18))
    plt.close(fig)
    return Asset(
        "Animation 4",
        "animations/animation_04_apophis_flyby_trajectory.gif",
        "Earth-centered flyby of Apophis synchronized with the tidal-intensity history.",
        "GIF",
    )


def animate_body_frame_tidal_field(traj: pd.DataFrame, out: Path) -> Asset:
    frames = sample_indices(len(traj), 100)
    hours = traj["time_s"].to_numpy() / 3600.0
    rh = traj[["rhat_body_x", "rhat_body_y"]].to_numpy()
    rh_norm = np.linalg.norm(rh, axis=1)
    rh = rh / rh_norm[:, None]
    tidal = traj["tidal_modulation"].to_numpy()

    grid = np.linspace(-1.65, 1.65, 140)
    xx, yy = np.meshgrid(grid, grid)
    rr = np.sqrt(xx**2 + yy**2)
    mask = rr <= 1.0
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlim(-1.35, 1.35)
    ax0.set_ylim(-1.35, 1.35)
    ax0.set_xlabel(r"body $x/R$")
    ax0.set_ylabel(r"body $y/R$")
    ax0.set_title(r"(a) Rotating-frame quadrupolar tide", loc="left")
    image = ax0.imshow(
        np.ma.masked_where(~mask, np.zeros_like(xx)),
        extent=(grid.min(), grid.max(), grid.min(), grid.max()),
        origin="lower",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    circle = plt.Circle((0, 0), 1.0, edgecolor="0.15", facecolor="none", lw=0.8)
    ax0.add_patch(circle)
    arrow = ax0.quiver(
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#171717",
        width=0.009,
    )
    label = ax0.text(0.03, 0.96, "", transform=ax0.transAxes, va="top")
    ax1.plot(hours, tidal, color="#244f46", lw=1.0)
    time_line = ax1.axvline(hours[0], color="#a22c29", lw=1.1)
    ax1.set_xlabel(r"time from periapsis, $t-t_p$ [h]")
    ax1.set_ylabel(r"$r_p^3/r^3$")
    ax1.set_title(r"(b) Encounter envelope", loc="left")

    def update(frame_index: int) -> tuple:
        idx = frames[frame_index]
        direction = rh[idx]
        projection = direction[0] * xx + direction[1] * yy
        field = tidal[idx] * (3.0 * projection**2 - rr**2)
        field = np.where(mask, field / max(float(np.max(tidal)), 1.0e-12), np.nan)
        image.set_data(np.ma.masked_invalid(field))
        arrow.set_UVC([1.05 * direction[0]], [1.05 * direction[1]])
        label.set_text(rf"$t-t_p={hours[idx]:+.2f}\,\mathrm{{h}}$")
        time_line.set_xdata([hours[idx], hours[idx]])
        return image, arrow, label, time_line

    update(0)
    fig.colorbar(image, ax=ax0, fraction=0.046, pad=0.035, label=r"normalized quadrupolar tide")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out, writer=PillowWriter(fps=16))
    plt.close(fig)
    return Asset(
        "Animation 5",
        "animations/animation_05_body_frame_tidal_field.gif",
        "Rotating-frame quadrupolar tidal field across the asteroid cross-section.",
        "GIF",
    )


def animate_quadrupole_surface_response(modal: pd.DataFrame, traj: pd.DataFrame, out: Path) -> Asset:
    frames = sample_indices(min(len(modal), len(traj)), 82)
    hours = modal["time_s"].to_numpy() / 3600.0
    response = modal["displacement_m"].to_numpy()
    response_scale = np.max(np.abs(response))
    direction = traj[["rhat_body_x", "rhat_body_y", "rhat_body_z"]].to_numpy()
    direction = direction / np.linalg.norm(direction, axis=1)[:, None]

    theta = np.linspace(0.0, np.pi, 60)
    phi = np.linspace(0.0, 2.0 * np.pi, 90)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    sphere = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=0,
    )

    fig = plt.figure(figsize=(5.0, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-42)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.55)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    lim = 1.42
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title(r"Amplified quadrupolar surface response", pad=1)
    surface = None
    text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes)

    def update(frame_index: int) -> tuple:
        nonlocal surface
        if surface is not None:
            surface.remove()
        idx = frames[frame_index]
        d = direction[idx]
        phase = np.tensordot(d, sphere, axes=(0, 0))
        mode = 0.5 * (3.0 * phase**2 - 1.0)
        amplitude = 0.28 * response[idx] / response_scale
        radial = 1.0 + amplitude * mode
        colors = plt.cm.RdBu_r((mode * np.sign(response[idx]) + 1.0) * 0.5)
        surface = ax.plot_surface(
            radial * sphere[0],
            radial * sphere[1],
            radial * sphere[2],
            facecolors=colors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        text.set_text(rf"$t-t_p={hours[idx]:+.2f}\,\mathrm{{h}}$" "\n" r"deformation amplified")
        return surface, text

    out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out, writer=PillowWriter(fps=14))
    plt.close(fig)
    return Asset(
        "Animation 6",
        "animations/animation_06_amplified_quadrupole_surface.gif",
        "Amplified quadrupolar surface deformation driven by the benchmark tidal response.",
        "GIF",
    )


def copy_static_assets(release: Path) -> list[Asset]:
    assets: list[Asset] = []
    main_figures = [
        (
            "Figure 1a",
            ROOT / "figures/results/adim/admin_coeffs_evol_rt.png",
            release / "figures/main/figure_01a_coefficient_evolution.png",
            "Assembled nondimensional coefficient magnitudes through the signed encounter coordinate.",
            "Nondimensional hierarchy",
        ),
        (
            "Figure 1b",
            ROOT / "figures/results/adim/nondimensional_hierarchy.png",
            release / "figures/main/figure_01b_nondimensional_hierarchy.png",
            "Periapsis ordering of elastic, tidal, centrifugal, self-gravity, inertial, and Coriolis terms.",
            "Nondimensional hierarchy",
        ),
        (
            "Figure 2",
            ROOT / "figures/results/trayectoria/trajectory_periapsis_true_anomaly.png",
            release / "figures/main/figure_02_periapsis_time_anomaly.png",
            "Periapsis-centered time-anomaly check used in the tidal-tensor construction.",
            "Trajectory reconstruction",
        ),
        (
            "Figure 3a",
            ROOT / "figures/results/trayectoria/rotation_frame_components.png",
            release / "figures/main/figure_03a_rotation_frame_components.png",
            "Earth direction components in the rotating benchmark frame.",
            "Body-frame forcing",
        ),
        (
            "Figure 3b",
            ROOT / "figures/results/armonicos/tidal_tensor_body_frame.png",
            release / "figures/main/figure_03b_tidal_tensor_components.png",
            "Independent components of the trace-free body-frame tidal tensor.",
            "Body-frame forcing",
        ),
        (
            "Figure 4",
            ROOT / "figures/results/armonicos/modal_diagnostics_periapsis.png",
            release / "figures/main/figure_04_modal_diagnostics_periapsis.png",
            "Nominal quadrupolar benchmark over the periapsis window.",
            "Quadrupolar modal benchmark",
        ),
        (
            "Figure 5",
            ROOT / "figures/results/armonicos/modal_diagnostics_extended.png",
            release / "figures/main/figure_05_modal_diagnostics_extended.png",
            "Nominal quadrupolar diagnostics over the extended encounter interval.",
            "Quadrupolar modal benchmark",
        ),
        (
            "Figure 6",
            ROOT / "figures/results/armonicos/regime_maps.png",
            release / "figures/main/figure_06_regime_maps.png",
            "Encounter-speed/periapsis and stiffness/periapsis regime maps.",
            "Apophis benchmark cases",
        ),
    ]

    supplementary = [
        (
            "Supplementary Figure S1",
            ROOT / "benchmark_outputs/figures/trajectory_frames.png",
            release / "figures/supplementary/figure_S1_trajectory_frames.png",
            "Hyperbolic trajectory in inertial and asteroid-centered frames.",
            "Trajectory reconstruction",
        ),
        (
            "Supplementary Figure S2",
            ROOT / "benchmark_outputs/figures/body_frame_forcing.png",
            release / "figures/supplementary/figure_S2_body_frame_forcing_compact.png",
            "Compact body-frame forcing diagnostic.",
            "Body-frame forcing",
        ),
        (
            "Supplementary Figure S3",
            ROOT / "benchmark_outputs/figures/coefficient_bars.png",
            release / "figures/supplementary/figure_S3_coefficient_bars.png",
            "Periapsis coefficient bars used to audit the nondimensional ordering.",
            "Nondimensional hierarchy",
        ),
        (
            "Supplementary Figure S4",
            ROOT / "benchmark_outputs/figures/encounter_regime_map.png",
            release / "figures/supplementary/figure_S4_encounter_regime_map.png",
            "Encounter-speed/periapsis regime map.",
            "Benchmark extension cases",
        ),
        (
            "Supplementary Figure S5",
            ROOT / "benchmark_outputs/figures/stiffness_periapsis_regime_map.png",
            release / "figures/supplementary/figure_S5_stiffness_periapsis_regime_map.png",
            "Stiffness/periapsis regime map.",
            "Benchmark extension cases",
        ),
        (
            "Supplementary Figure S6",
            ROOT / "benchmark_outputs/figures/modal_diagnostics.png",
            release / "figures/supplementary/figure_S6_modal_diagnostics_compact.png",
            "Compact nominal modal diagnostic with forcing, displacement, amplitude, and energy.",
            "Quadrupolar modal benchmark",
        ),
    ]

    extended_optional = [
        (
            "Extended Figure E1",
            ROOT / "figures/results/armonicos/tidal_quadrupole_field_2d.png",
            release / "figures/extended/figure_E1_tidal_quadrupole_field_2d.png",
            "Two-dimensional visualization of the quadrupolar tidal field.",
            "Quadrupolar tide",
        ),
        (
            "Extended Figure E2",
            ROOT / "figures/results/armonicos/zeta_regime_map.png",
            release / "figures/extended/figure_E2_zeta_regime_map.png",
            "Standalone zeta regime map retained from the extended figure set.",
            "Benchmark extension cases",
        ),
        (
            "Extended Figure E3",
            ROOT / "figures/results/armonicos/max_h2_amplitude_map.png",
            release / "figures/extended/figure_E3_max_quadrupole_amplitude_map.png",
            "Map of maximum quadrupolar amplitude over the extension parameter space.",
            "Benchmark extension cases",
        ),
        (
            "Extended Figure E4",
            ROOT / "figures/results/adim/coefficients_trajectory_extended.png",
            release / "figures/extended/figure_E4_coefficients_trajectory_extended.png",
            "Extended coefficient history along the trajectory.",
            "Nondimensional hierarchy",
        ),
    ]

    for label, source, destination, description, section in main_figures:
        copy_file(source, destination)
        assets.append(
            Asset(label, str(destination.relative_to(release)).replace("\\", "/"), description, section, str(source.relative_to(ROOT)).replace("\\", "/"))
        )

    for label, source, destination, description, section in supplementary:
        if copy_optional(source, destination):
            assets.append(
                Asset(label, str(destination.relative_to(release)).replace("\\", "/"), description, section, str(source.relative_to(ROOT)).replace("\\", "/"))
            )

    for label, source, destination, description, section in extended_optional:
        if copy_optional(source, destination):
            assets.append(
                Asset(label, str(destination.relative_to(release)).replace("\\", "/"), description, section, str(source.relative_to(ROOT)).replace("\\", "/"))
            )

    return assets


def copy_data_and_scripts(release: Path) -> None:
    for source in (ROOT / "benchmark_outputs/data").glob("*.csv"):
        copy_file(source, release / "data" / source.name)
    for source in (ROOT / "benchmark_outputs/tables").glob("*"):
        if source.is_file():
            copy_file(source, release / "results/tables" / source.name)
    for name in ["summary.json", "inputs_snapshot.json", "manifest.json"]:
        copy_file(ROOT / "benchmark_outputs" / name, release / "results" / name)
    for name in ["run_benchmark.py", "inputs.py", "prepare_github_release.py", "reproduce_all.py"]:
        copy_file(ROOT / name, release / "scripts" / name)
    (release / "notebooks").mkdir(parents=True, exist_ok=True)
    (release / "notebooks/README.md").write_text(
        "# Notebooks\n\nNo notebooks are included in this release snapshot. "
        "The benchmark is currently generated from the Python scripts in `scripts/`.\n",
        encoding="utf-8",
    )


def markdown_table(assets: Iterable[Asset], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for asset in assets:
        values = []
        for column in columns:
            if column == "Figure":
                values.append(asset.label)
            elif column == "Animation":
                values.append(asset.label)
            elif column == "File":
                values.append(f"`{asset.path}`")
            elif column == "Description":
                values.append(asset.description)
            elif column == "Related section":
                values.append(asset.section)
            elif column == "Format":
                values.append(asset.section)
            elif column == "Source":
                values.append(f"`{asset.source}`")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def write_readme(release: Path, figures: list[Asset], animations: list[Asset]) -> None:
    main_figures = [asset for asset in figures if asset.path.startswith("figures/main/")]
    supplementary = [
        asset for asset in figures if asset.path.startswith("figures/supplementary/")
    ]
    extended = [asset for asset in figures if asset.path.startswith("figures/extended/")]
    readme = f"""# {TITLE}

Additional figures, animations, data, and code associated with:

**{TITLE}**

Authors: **{", ".join(AUTHORS)}**

Affiliations: **{"; ".join(AFFILIATIONS)}**

Corresponding author: **Francisco Cruz Perez** (`TBD`)

---

## Paper Information

- **Title:** *{TITLE}*
- **Authors:** {", ".join(AUTHORS)}
- **Status:** Manuscript version, June 2026
- **DOI:** `TBD`
- **Preprint:** `TBD`
- **Publication date:** `TBD`
- **Repository URL:** [{REPOSITORY_URL}]({REPOSITORY_URL})

Suggested citation:

```bibtex
@article{{CruzPerez2026TidalBenchmark,
  title  = {{{TITLE}}},
  author = {{Cruz Perez, Francisco and Herreros Cid, Isabel and Veiga Lopez, Fernando}},
  year   = {{2026}},
  note   = {{Manuscript in preparation}},
  doi    = {{TBD}}
}}
```

---

## Overview

This repository contains supplementary material for a continuum benchmark of
tidal deformation during the 2029 Earth encounter of (99942) Apophis. It
collects the final manuscript figures, additional explanatory figures,
animations, numerical outputs, and scripts used to generate the benchmark
material.

The purpose of this archive is to provide:

- A clean figure set matching the manuscript.
- Extended diagnostics for the quadrupolar mode, tidal tensor, and regime maps.
- GIF animations for the body-frame forcing and modal response.
- Numerical CSV outputs and tables for independent verification.
- Current scripts, with a later consolidation pass planned for a single final workflow.

---

## Abstract

```text
{ABSTRACT}
```

---

## Repository Structure

```text
.
|-- README.md
|-- figures/
|   |-- main/
|   |-- supplementary/
|   `-- extended/
|-- animations/
|-- scripts/
|-- notebooks/
|-- data/
|-- results/
|   `-- tables/
|-- docs/
|-- environment.yml
|-- requirements.txt
|-- LICENSE
|-- CITATION.cff
`-- MANIFEST.json
```

### Folder Description

| Folder | Description |
| --- | --- |
| `figures/main/` | Final figure files used in the manuscript. |
| `figures/supplementary/` | Auxiliary diagnostics generated by the benchmark pipeline. |
| `figures/extended/` | Additional high-level explanatory figures not included in the manuscript. |
| `animations/` | GIF animations showing body-frame forcing, tensor orientation, and modal response. |
| `scripts/` | Current Python scripts and input definitions. |
| `notebooks/` | Placeholder for future exploratory notebooks. |
| `data/` | CSV outputs from the benchmark calculations. |
| `results/` | Summary JSON, provenance files, and table outputs. |
| `docs/` | Figure index, data inventory, and release notes. |

---

## Figures In The Paper

{markdown_table(main_figures, ["Figure", "File", "Description", "Related section"])}

## Supplementary Figures

{markdown_table(supplementary, ["Figure", "File", "Description", "Related section"])}

## Extended Figures

{markdown_table(extended, ["Figure", "File", "Description", "Related section"])}

---

## Animations

{markdown_table(animations, ["Animation", "File", "Description", "Format"])}

Each animation is generated from the benchmark CSV outputs and is intended for
interpretation and presentation. The numerical CSV files remain the reference
material for quantitative comparison.

---

## Code

The recommended one-command workflow is:

```bash
python scripts/reproduce_all.py
```

This regenerates the benchmark calculations, data files, paper figures,
extended figures, animations, documentation indexes, image checks, and
manifests from the repository-local source files.

The included scripts are:

- `scripts/inputs.py`: canonical physical, encounter, numerical, and sweep inputs.
- `scripts/run_benchmark.py`: current benchmark generation workflow.
- `scripts/prepare_github_release.py`: provenance copy of the packaging utility used to build this folder.
- `scripts/reproduce_all.py`: unified reproducibility entry point for this GitHub repository.

The code was developed with Python 3.12 and uses NumPy, Pandas, Matplotlib, and
Pillow. Install dependencies with:

```bash
pip install -r requirements.txt
```

Run the complete repository reproduction from the repository root with:

```bash
python scripts/reproduce_all.py
```

---

## Data

| Dataset | Location | Description | Source |
| --- | --- | --- | --- |
| Trajectory histories | `data/trajectory_*.csv` | Hyperbolic trajectory and body-frame direction/tensor quantities. | Generated by `run_benchmark.py` |
| Modal histories | `data/modal_response_*.csv` | Dimensional quadrupolar response, quasi-static output, amplitudes, and energy. | Generated by `run_benchmark.py` |
| Nondimensional coefficients | `data/nondimensional_coefficients_*.csv` | Coefficient hierarchy along the full and periapsis windows. | Generated by `run_benchmark.py` |
| Regime maps | `data/*regime_map.csv` | Encounter and stiffness grids for zeta-classification plots. | Generated by `run_benchmark.py` |
| Tables | `results/tables/` | Input, derived, validation, stiffness, and parametric tables. | Generated by `run_benchmark.py` |
| Summary and provenance | `results/*.json`, `MANIFEST.json` | Benchmark summary and SHA-256 file inventory. | Generated locally |

Some CSV files are large because they preserve dense histories for verification.
For a journal archive, they can also be mirrored to Zenodo or Git LFS.

---

## Reproducibility

To reproduce the current benchmark outputs:

1. Install the dependencies.
2. Run `python scripts/reproduce_all.py`.
3. Compare generated files against `MANIFEST.json` or `results/manifest.json`.

The benchmark summary reports a peak dynamic quadrupolar displacement of
`8.176171003423148e-07 m` for the nominal Apophis reference case, with all
recorded validation checks passing in the packaged run.

---

## Versioning

| Repository version | Paper status | Description |
| --- | --- | --- |
| `v0.1` | Draft | Initial GitHub-ready supplementary structure. |
| `v1.0` | Submitted | Version associated with manuscript submission. |
| `v1.1` | Revised | Updated version after peer review. |
| `v2.0` | Published | Final version associated with the published paper. |

The published repository version should be archived with Zenodo or an equivalent
service to obtain a permanent DOI.

---

## License

Code in this repository is released under the MIT License. Figures,
documentation, animations, and supplementary material are released under the
Creative Commons Attribution 4.0 International License unless otherwise stated.
Data tables are released under CC BY 4.0 unless a future publication record
specifies a different data license.

See `LICENSE` for details.
"""
    (release / "README.md").write_text(readme, encoding="utf-8")


def write_support_files(release: Path, figures: list[Asset], animations: list[Asset]) -> None:
    (release / "requirements.txt").write_text(
        "numpy>=1.26\npandas>=2.2\nmatplotlib>=3.8\npillow>=10\n",
        encoding="utf-8",
    )
    (release / "environment.yml").write_text(
        """name: apophis-tidal-benchmark
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy
  - pandas
  - matplotlib
  - pillow
""",
        encoding="utf-8",
    )
    (release / "LICENSE").write_text(
        """MIT License for code

Copyright (c) 2026 Francisco Cruz Perez, Isabel Herreros Cid, and Fernando
Veiga Lopez.

Permission is hereby granted, free of charge, to any person obtaining a copy of
the code in this repository to deal in the code without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the code, subject to inclusion of this notice.

Figures, documentation, animations, and supplementary material are distributed
under the Creative Commons Attribution 4.0 International License (CC BY 4.0)
unless otherwise stated.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
""",
        encoding="utf-8",
    )
    (release / "CITATION.cff").write_text(
        f"""cff-version: 1.2.0
message: "If you use this repository, please cite the associated paper."
title: "{TITLE}"
authors:
  - family-names: "Cruz Perez"
    given-names: "Francisco"
  - family-names: "Herreros Cid"
    given-names: "Isabel"
  - family-names: "Veiga Lopez"
    given-names: "Fernando"
date-released: "2026-06-19"
version: "0.1.0"
doi: "TBD"
url: "{REPOSITORY_URL}"
""",
        encoding="utf-8",
    )
    docs = release / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    figure_index = (
        "# Figure Index\n\n"
        + markdown_table(figures, ["Figure", "File", "Description", "Related section", "Source"])
        + "\n\n# Animation Index\n\n"
        + markdown_table(animations, ["Animation", "File", "Description", "Format"])
        + "\n"
    )
    (docs / "figure_index.md").write_text(figure_index, encoding="utf-8")

    data_rows = []
    for path in sorted((release / "data").glob("*.csv")):
        data_rows.append(
            {
                "file": str(path.relative_to(release)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (docs / "data_inventory.md").write_text(
        "# Data Inventory\n\n"
        "| File | Bytes | SHA-256 |\n"
        "| --- | ---: | --- |\n"
        + "\n".join(
            f"| `{row['file']}` | {row['bytes']} | `{row['sha256']}` |"
            for row in data_rows
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "release_notes.md").write_text(
        """# Release Notes

## v0.1.0

- Organized manuscript figures into `figures/main/`.
- Added supplementary and extended diagnostic figures.
- Generated GIF animations from benchmark CSV outputs.
- Included current scripts, numerical CSV files, tables, and provenance manifests.
- Added README, requirements, environment, license, citation, and inventory files.
""",
        encoding="utf-8",
    )


def write_manifest(release: Path) -> None:
    files = []
    for path in sorted(release.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": str(path.relative_to(release)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "title": TITLE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    (release / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def verify_images(release: Path) -> dict[str, int]:
    from PIL import Image

    counts: dict[str, int] = {}
    for path in release.rglob("*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif"}:
            continue
        with Image.open(path) as img:
            counts[str(path.relative_to(release)).replace("\\", "/")] = getattr(
                img, "n_frames", 1
            )
            img.verify()
    return counts


def build_release(release: Path, overwrite: bool) -> Path:
    ensure_clean_release(release, overwrite)
    style()
    data = load_data()
    figure_assets = copy_static_assets(release)

    generated_dir = release / "figures/extended"
    figure_assets.append(
        generate_quadrupole_gallery(
            generated_dir / "figure_E5_quadrupole_mode_gallery.png"
        )
    )
    figure_assets.append(
        generate_tensor_eigenstructure(
            data["trajectory"],
            generated_dir / "figure_E6_tidal_tensor_eigenstructure.png",
        )
    )
    figure_assets.append(
        generate_response_scaling(
            data["stiffness"],
            data["parametric"],
            generated_dir / "figure_E7_response_scaling.png",
        )
    )
    figure_assets.append(
        generate_body_frame_story(
            data["trajectory"],
            generated_dir / "figure_E8_body_frame_tidal_geometry.png",
        )
    )

    animation_assets = [
        animate_body_frame_direction(
            data["trajectory"],
            release / "animations/animation_01_body_frame_direction.gif",
        ),
        animate_modal_response(
            data["modal"],
            release / "animations/animation_02_quadrupole_forcing_response.gif",
        ),
        animate_tensor_principal_axes(
            data["trajectory"],
            release / "animations/animation_03_tidal_tensor_principal_axes.gif",
        ),
        animate_apophis_flyby_trajectory(
            data["trajectory_full"],
            release / "animations/animation_04_apophis_flyby_trajectory.gif",
        ),
        animate_body_frame_tidal_field(
            data["trajectory"],
            release / "animations/animation_05_body_frame_tidal_field.gif",
        ),
        animate_quadrupole_surface_response(
            data["modal"],
            data["trajectory"],
            release / "animations/animation_06_amplified_quadrupole_surface.gif",
        ),
    ]

    copy_data_and_scripts(release)
    write_readme(release, figure_assets, animation_assets)
    write_support_files(release, figure_assets, animation_assets)
    write_manifest(release)
    image_frames = verify_images(release)
    (release / "results/image_verification.json").write_text(
        json.dumps(image_frames, indent=2), encoding="utf-8"
    )
    write_manifest(release)
    return release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE,
        help="Destination directory for the GitHub-ready snapshot.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing release directory inside the workspace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = build_release(args.release_dir, args.overwrite)
    manifest = json.loads((release / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"Release directory: {release}")
    print(f"Packaged files: {manifest['file_count']}")
    print("Image verification: results/image_verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
