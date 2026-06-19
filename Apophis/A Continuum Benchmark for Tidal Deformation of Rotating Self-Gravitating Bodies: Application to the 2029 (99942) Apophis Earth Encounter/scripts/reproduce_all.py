"""One-command reproduction workflow for the Apophis tidal benchmark repository.

Run from the GitHub repository root:

    python scripts/reproduce_all.py

The script regenerates numerical benchmark outputs, manuscript figures,
extended figures, GIF animations, documentation indexes, image verification,
and SHA-256 manifests using the repository-local Python files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "scripts":
    REPO_ROOT = SCRIPT_PATH.parent.parent
    SCRIPT_DIR = SCRIPT_PATH.parent
else:
    REPO_ROOT = SCRIPT_PATH.parent
    SCRIPT_DIR = REPO_ROOT

sys.path.insert(0, str(SCRIPT_DIR))

import prepare_github_release as package  # noqa: E402
import run_benchmark  # noqa: E402


def safe_copy(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    package.copy_file(source, destination)


def refresh_data_and_results(repo: Path) -> None:
    benchmark = repo / "benchmark_outputs"
    for source in (benchmark / "data").glob("*.csv"):
        package.copy_file(source, repo / "data" / source.name)
    for source in (benchmark / "tables").glob("*"):
        if source.is_file():
            package.copy_file(source, repo / "results/tables" / source.name)
    for name in ["summary.json", "inputs_snapshot.json", "manifest.json"]:
        package.copy_file(benchmark / name, repo / "results" / name)


def refresh_scripts(repo: Path, script_dir: Path) -> None:
    for name in [
        "inputs.py",
        "run_benchmark.py",
        "prepare_github_release.py",
        "reproduce_all.py",
    ]:
        source = script_dir / name
        if source.is_file():
            safe_copy(source, repo / "scripts" / name)


def regenerate_repository(quick: bool = False, skip_animations: bool = False) -> Path:
    run_benchmark.ROOT = REPO_ROOT
    package.ROOT = REPO_ROOT
    package.DEFAULT_RELEASE = REPO_ROOT

    def write_manifest_repo(output_dir: Path, generated_files: list[Path]) -> None:
        source_files = [SCRIPT_DIR / "inputs.py", SCRIPT_DIR / "run_benchmark.py"]
        records = []
        for path in sorted(source_files + generated_files, key=lambda item: str(item)):
            try:
                relative = path.relative_to(REPO_ROOT)
            except ValueError:
                relative = path
            records.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": run_benchmark.sha256(path),
                }
            )
        manifest = {
            "benchmark_id": run_benchmark.CONFIG.benchmark_id,
            "python": run_benchmark.platform.python_version(),
            "platform": run_benchmark.platform.platform(),
            "numpy": run_benchmark.np.__version__,
            "pandas": run_benchmark.pd.__version__,
            "matplotlib": run_benchmark.matplotlib.__version__,
            "files": records,
        }
        run_benchmark.write_json(output_dir / "manifest.json", manifest)

    run_benchmark.write_manifest = write_manifest_repo

    args = SimpleNamespace(
        output_dir=REPO_ROOT / "benchmark_outputs",
        quick=quick,
        no_plots=False,
    )
    run_benchmark.run(run_benchmark.CONFIG, args)

    package.style()
    data = package.load_data()
    figure_assets = package.copy_static_assets(REPO_ROOT)
    extended_dir = REPO_ROOT / "figures/extended"
    figure_assets.extend(
        [
            package.generate_quadrupole_gallery(
                extended_dir / "figure_E5_quadrupole_mode_gallery.png"
            ),
            package.generate_tensor_eigenstructure(
                data["trajectory"],
                extended_dir / "figure_E6_tidal_tensor_eigenstructure.png",
            ),
            package.generate_response_scaling(
                data["stiffness"],
                data["parametric"],
                extended_dir / "figure_E7_response_scaling.png",
            ),
            package.generate_body_frame_story(
                data["trajectory"],
                extended_dir / "figure_E8_body_frame_tidal_geometry.png",
            ),
        ]
    )

    animation_assets = []
    if not skip_animations:
        animation_assets = [
            package.animate_body_frame_direction(
                data["trajectory"],
                REPO_ROOT / "animations/animation_01_body_frame_direction.gif",
            ),
            package.animate_modal_response(
                data["modal"],
                REPO_ROOT / "animations/animation_02_quadrupole_forcing_response.gif",
            ),
            package.animate_tensor_principal_axes(
                data["trajectory"],
                REPO_ROOT / "animations/animation_03_tidal_tensor_principal_axes.gif",
            ),
            package.animate_apophis_flyby_trajectory(
                data["trajectory_full"],
                REPO_ROOT / "animations/animation_04_apophis_flyby_trajectory.gif",
            ),
            package.animate_body_frame_tidal_field(
                data["trajectory"],
                REPO_ROOT / "animations/animation_05_body_frame_tidal_field.gif",
            ),
            package.animate_quadrupole_surface_response(
                data["modal"],
                data["trajectory"],
                REPO_ROOT / "animations/animation_06_amplified_quadrupole_surface.gif",
            ),
        ]

    refresh_data_and_results(REPO_ROOT)
    refresh_scripts(REPO_ROOT, SCRIPT_DIR)
    package.write_readme(REPO_ROOT, figure_assets, animation_assets)
    package.write_support_files(REPO_ROOT, figure_assets, animation_assets)
    image_frames = package.verify_images(REPO_ROOT)
    (REPO_ROOT / "results/image_verification.json").write_text(
        json.dumps(image_frames, indent=2), encoding="utf-8"
    )
    package.write_manifest(REPO_ROOT)
    return REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller numerical grids for a fast smoke-test reproduction.",
    )
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Regenerate data and figures but skip GIF creation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = regenerate_repository(quick=args.quick, skip_animations=args.skip_animations)
    manifest = json.loads((repo / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"Repository root: {repo}")
    print(f"Packaged files: {manifest['file_count']}")
    print("Full reproduction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
