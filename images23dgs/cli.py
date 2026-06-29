from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .colmap import collect_images
from .pipeline import DEFAULT_DISCOVERSE_ROOT, DEFAULT_REAL2SIM_ROOT, PipelineConfig, run_pipeline
from .ply import read_ply_header
from .true_geometry import generate_depth_point_cloud
from .product.config import load_config, write_default_config
from .product.doctor import print_doctor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="images23dgs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run image-to-3DGS generation with optional ArtiFixer repair")
    run_parser.add_argument("--images", type=Path, required=True, help="Input image file or directory")
    run_parser.add_argument("--output", type=Path, required=True, help="Output run directory")
    run_parser.add_argument("--prompt", default="static scene with task-relevant geometry", help="Scene prompt/context")
    run_parser.add_argument("--scene-name", default="scene")
    run_parser.add_argument("--discoverse-root", type=Path, default=DEFAULT_DISCOVERSE_ROOT)
    run_parser.add_argument("--real2sim-root", type=Path, default=DEFAULT_REAL2SIM_ROOT)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--backend", default="auto", choices=["auto", "builtin", "external", "hybrid"])
    run_parser.add_argument("--config", type=Path, default=None, help="Real2Sim external-stage config JSON")
    run_parser.add_argument("--enable-prune", action="store_true", help="Enable Real2Sim 3DGS pruning; off by default to avoid optional gaussian_renderer dependency")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--symlink-images", action="store_true", help="Symlink source images instead of copying")
    run_parser.add_argument("--no-colmap", action="store_true")
    run_parser.add_argument("--colmap-binary", default="colmap")
    run_parser.add_argument("--colmap-matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    run_parser.add_argument("--colmap-cpu", action="store_true")
    run_parser.add_argument("--colmap-num-threads", type=int, default=None)
    run_parser.add_argument("--colmap-max-image-size", type=int, default=None)
    run_parser.add_argument("--colmap-separate-cameras", action="store_true", help="Do not force COLMAP to share one camera calibration across all images")
    run_parser.add_argument("--min-images-for-direct", type=int, default=24)
    run_parser.add_argument("--min-registered-ratio", type=float, default=0.65)
    run_parser.add_argument("--force-artifixer", action="store_true")
    run_parser.add_argument("--skip-artifixer", action="store_true")
    run_parser.add_argument("--artifixer-root", type=Path, default=None)
    run_parser.add_argument("--artifixer-python", default="python")
    run_parser.add_argument("--artifixer-checkpoint", type=Path, default=None)
    run_parser.add_argument("--artifixer-render-trajectory", default="all_frames")
    run_parser.add_argument("--artifixer-prepare-phases", default=None, help="Override ArtiFixer prepare phases, e.g. prepare,reconstruct,render,scale")
    run_parser.add_argument("--artifixer-skip-caption", action="store_true", help="Skip ArtiFixer caption phase and use the model's empty-prompt fallback")
    run_parser.add_argument("--artifixer-anchor-count", type=int, default=None, help="Number of registered COLMAP images to keep as real ArtiFixer anchors; defaults to about half")
    run_parser.add_argument("--artifixer-reconstruction-steps", type=int, default=10000)
    run_parser.add_argument("--fetch-artifixer", action="store_true")
    run_parser.set_defaults(func=_run)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect input images or a PLY file")
    inspect_parser.add_argument("--images", type=Path, default=None)
    inspect_parser.add_argument("--ply", type=Path, default=None)
    inspect_parser.set_defaults(func=_inspect)

    true_geometry_parser = subparsers.add_parser(
        "true-geometry",
        help="Back-project ArtiFixer depth maps and camera poses into an RGB point cloud",
    )
    true_geometry_parser.add_argument("--prep-root", type=Path, required=True, help="ArtiFixer prepare scene root")
    true_geometry_parser.add_argument("--output-ply", type=Path, required=True)
    true_geometry_parser.add_argument("--predicted-frames", type=Path, default=None, help="Optional ArtiFixer repaired frame directory")
    true_geometry_parser.add_argument("--max-points", type=int, default=350_000)
    true_geometry_parser.add_argument("--opacity-threshold", type=float, default=0.05)
    true_geometry_parser.add_argument("--z-sign", type=float, default=-1.0, choices=[-1.0, 1.0])
    true_geometry_parser.set_defaults(func=_true_geometry)

    product_parser = subparsers.add_parser("product", help="Run and inspect the Wuying single-machine web product")
    product_subparsers = product_parser.add_subparsers(dest="product_command", required=True)
    doctor_parser = product_subparsers.add_parser("doctor", help="Check product runtime dependencies")
    doctor_parser.add_argument("--config", type=Path, default=None)
    doctor_parser.set_defaults(func=_product_doctor)
    serve_parser = product_subparsers.add_parser("serve", help="Start the product web server")
    serve_parser.add_argument("--config", type=Path, default=None)
    serve_parser.set_defaults(func=_product_serve)
    write_config_parser = product_subparsers.add_parser("write-config", help="Write the default product config")
    write_config_parser.add_argument("--path", type=Path, required=True)
    write_config_parser.add_argument("--overwrite", action="store_true")
    write_config_parser.set_defaults(func=_product_write_config)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _run(args: argparse.Namespace) -> int:
    manifest = run_pipeline(
        PipelineConfig(
            images=args.images,
            output=args.output,
            prompt=args.prompt,
            scene_name=args.scene_name,
            discoverse_root=args.discoverse_root,
            real2sim_root=args.real2sim_root,
            python=args.python,
            backend=args.backend,
            config_file=args.config,
            enable_prune=args.enable_prune,
            dry_run=args.dry_run,
            resume=args.resume,
            copy_images=not args.symlink_images,
            run_colmap_stage=not args.no_colmap,
            colmap_binary=args.colmap_binary,
            colmap_matcher=args.colmap_matcher,
            colmap_use_gpu=not args.colmap_cpu,
            colmap_num_threads=args.colmap_num_threads,
            colmap_max_image_size=args.colmap_max_image_size,
            colmap_single_camera=not args.colmap_separate_cameras,
            min_images_for_direct=args.min_images_for_direct,
            min_registered_ratio=args.min_registered_ratio,
            force_artifixer=args.force_artifixer,
            skip_artifixer=args.skip_artifixer,
            artifixer_root=args.artifixer_root,
            artifixer_python=args.artifixer_python,
            artifixer_checkpoint=args.artifixer_checkpoint,
            artifixer_render_trajectory=args.artifixer_render_trajectory,
            artifixer_prepare_phases=_artifixer_prepare_phases(args),
            artifixer_anchor_count=args.artifixer_anchor_count,
            artifixer_reconstruction_steps=args.artifixer_reconstruction_steps,
            fetch_artifixer=args.fetch_artifixer,
        )
    )
    print(json.dumps(_summary(manifest), indent=2, ensure_ascii=False))
    return 0


def _artifixer_prepare_phases(args: argparse.Namespace) -> str | None:
    if args.artifixer_prepare_phases:
        return args.artifixer_prepare_phases
    if args.artifixer_skip_caption:
        return "prepare,reconstruct,render,scale"
    return None


def _inspect(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {}
    if args.images:
        images = collect_images(args.images)
        payload["images"] = {
            "source": str(args.images),
            "count": len(images),
            "first": str(images[0]) if images else None,
            "last": str(images[-1]) if images else None,
        }
    if args.ply:
        payload["ply"] = read_ply_header(args.ply).to_jsonable()
    if not payload:
        raise SystemExit("pass --images or --ply")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _true_geometry(args: argparse.Namespace) -> int:
    result = generate_depth_point_cloud(
        prep_root=args.prep_root,
        output_ply=args.output_ply,
        predicted_frames_dir=args.predicted_frames,
        max_points=args.max_points,
        opacity_threshold=args.opacity_threshold,
        z_sign=args.z_sign,
    )
    print(json.dumps(result.to_jsonable(), indent=2, ensure_ascii=False))
    return 0


def _product_doctor(args: argparse.Namespace) -> int:
    print_doctor(load_config(args.config))
    return 0


def _product_serve(args: argparse.Namespace) -> int:
    from .product.server import create_app
    import uvicorn

    config = load_config(args.config)
    uvicorn.run(create_app(config), host=config.host, port=config.port)
    return 0


def _product_write_config(args: argparse.Namespace) -> int:
    write_default_config(args.path, overwrite=args.overwrite)
    print(str(args.path))
    return 0


def _summary(manifest: dict[str, object]) -> dict[str, object]:
    colmap = manifest.get("colmap", {})
    artifixer = manifest.get("artifixer", {})
    discoverse = manifest.get("discoverse", {})
    return {
        "manifest": str(Path(str(manifest["output"])) / "reports" / "run_manifest.json"),
        "run_commands": str(Path(str(manifest["output"])) / "run_commands.sh"),
        "source_images": colmap.get("source_images") if isinstance(colmap, dict) else None,
        "colmap_status": colmap.get("status") if isinstance(colmap, dict) else None,
        "artifixer_selected": artifixer.get("selected") if isinstance(artifixer, dict) else None,
        "artifixer_status": artifixer.get("status") if isinstance(artifixer, dict) else None,
        "discoverse_status": discoverse.get("status") if isinstance(discoverse, dict) else None,
        "final_scene": manifest.get("final_scene"),
        "viewer": manifest.get("viewer", {}).get("index_html") if isinstance(manifest.get("viewer"), dict) else None,
        "dry_run": manifest.get("dry_run"),
    }
