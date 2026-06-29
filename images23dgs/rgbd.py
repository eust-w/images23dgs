from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ply import read_ply_header
from .viewer import VIEWER_HTML, _prepare_spark_ply


LogFn = Callable[[str], None]


AHOLO_VIEWER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aholo 3DGS 预览</title>
  <style>
    html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#070a0f;color:#eaf0f7;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    #viewer{position:fixed;inset:0}
    .hud{position:fixed;left:14px;top:14px;z-index:10;max-width:min(420px,calc(100vw - 28px));background:rgba(9,13,18,.82);border:1px solid rgba(137,160,190,.28);border-radius:8px;padding:12px 14px;backdrop-filter:blur(12px)}
    h1{font-size:15px;margin:0 0 8px}.muted{color:#a9b7c8;font-size:12px;line-height:1.55;margin:6px 0}.status{font-size:13px;margin:0}.error{color:#ff9a9a;white-space:pre-wrap}
    .actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.actions a,.actions button{background:#162234;color:#edf5ff;border:1px solid #334964;border-radius:6px;padding:7px 9px;text-decoration:none;font-size:12px;cursor:pointer}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div class="hud">
    <h1>Aholo 3DGS 高性能预览</h1>
    <p id="status" class="status">正在读取 viewer_manifest.json...</p>
    <p class="muted">适合大规模 3D Gaussian Splatting 预览；如果浏览器禁用了外部 ESM CDN，请使用原始分层 viewer。</p>
    <div class="actions">
      <a href="../index.html">分层 viewer</a>
      <a id="download" href="#">下载 PLY</a>
      <button id="reset">重置相机</button>
    </div>
  </div>
  <script type="module">
    const statusEl = document.getElementById("status");
    const container = document.getElementById("viewer");
    const download = document.getElementById("download");
    let viewer = null;
    let camera = null;
    let cameraTarget = [0, 0, 0];
    let cameraRadius = 1.25;

    function setStatus(text, cls = "") {
      statusEl.className = cls ? `status ${cls}` : "status";
      statusEl.textContent = text;
    }

    async function loadManifest() {
      const manifest = await fetch("../viewer_manifest.json", { cache: "no-store" }).then(r => {
        if (!r.ok) throw new Error(`manifest HTTP ${r.status}`);
        return r.json();
      });
      const layer = manifest.layers?.aholo_3dgs || manifest.layers?.spark_3dgs;
      if (!layer?.available || (!layer.file && !layer.lod_meta)) throw new Error("当前任务没有可用 3DGS。");
      const url = layer.file ? new URL("../" + layer.file, location.href).href : null;
      const lodUrl = layer.lod_meta ? new URL("../" + layer.lod_meta, location.href).href : null;
      return { manifest, layer, url, lodUrl };
    }

    function resetCamera(Vector3) {
      if (!camera || !viewer) return;
      camera.up.set(0, -1, 0);
      const [x, y, z] = cameraTarget;
      const d = Math.max(1, cameraRadius);
      camera.position.set(x, y - d * 0.35, z + d * 1.8);
      camera.lookAt(new Vector3(x, y, z));
      viewer.setCamera(camera);
      viewer.render();
    }

    function fitCameraToBox(box) {
      if (!box?.min || !box?.max) return;
      cameraTarget = [
        (box.min[0] + box.max[0]) / 2,
        (box.min[1] + box.max[1]) / 2,
        (box.min[2] + box.max[2]) / 2,
      ];
      const dx = box.max[0] - box.min[0];
      const dy = box.max[1] - box.min[1];
      const dz = box.max[2] - box.min[2];
      cameraRadius = Math.max(dx, dy, dz, 1);
    }

    async function fetchBytes(url, label) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`3DGS HTTP ${response.status}`);
      const total = Number(response.headers.get("content-length") || 0);
      if (!response.body?.getReader) {
        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
      }
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        const mb = (received / 1048576).toFixed(1);
        const totalText = total ? ` / ${(total / 1048576).toFixed(1)} MB` : " MB";
        const pct = total ? ` (${Math.round(received * 100 / total)}%)` : "";
        setStatus(`正在下载 ${label}: ${mb}${totalText}${pct}`);
      }
      const bytes = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return bytes;
    }

    async function main() {
      const { manifest, layer, url, lodUrl } = await loadManifest();
      download.href = url || lodUrl || "#";
      setStatus(`正在加载 Aholo 包和 ${Number(layer.gaussians || 0).toLocaleString()} 个高斯...`);
      const {
        BackgroundMode,
        Color,
        PerspectiveCamera,
        SplatLoader,
        SplatUtils,
        Vector3,
        createViewerContext,
        createViewer,
        setViewerConfig,
      } = await import("https://esm.sh/@manycore/aholo-viewer@1.5.1");

      viewer = createViewer("images23dgs-aholo-viewer", container, {});
      camera = new PerspectiveCamera(60, Math.max(0.1, container.clientWidth / Math.max(1, container.clientHeight)), 0.01, 2000);

      const tickers = [];
      if (lodUrl) {
        setStatus(`正在加载 Aholo LOD: ${layer.lod_meta}`);
        const meta = await fetch(lodUrl, { cache: "no-store" }).then(r => {
          if (!r.ok) throw new Error(`LOD HTTP ${r.status}`);
          return r.json();
        });
        if (!(meta.magicCode === 2500660 && meta.type === "lod-splat")) {
          throw new Error("LOD metadata is not a supported lod-splat manifest.");
        }
        fitCameraToBox(meta.forwardBox);
        const baseUrl = new URL("./", lodUrl).href;
        const loadResource = async relativeUrl => {
          const resourceUrl = new URL(relativeUrl, baseUrl).href;
          const fileType = SplatLoader.detectSplatFileType(resourceUrl, new Uint8Array());
          if (fileType === undefined) throw new Error(`Unsupported LOD resource: ${relativeUrl}`);
          return SplatLoader.parseSplatData(fileType, resourceUrl, SplatLoader.SplatPackType.Compressed, {
            maxShDegree: 0,
            maxTextureSize: 8192,
          });
        };
        const lod = new SplatUtils.LodSplat(
          meta,
          {
            minLevel: Math.max(0, meta.levels - 1),
            maxBudget: 3000000,
            backgroundPenalty: 0.5,
            outsidePenalty: 0.4,
            behindPenalty: 0.1,
            behindTolerance: -0.2,
            behindDistanceTolerance: 2,
            hysteresisTicks: 4,
            schedulerParallelCounts: 8,
            schedulerExistingTaskLimit: 64,
            schedulerMinDuration: 120,
          },
          createViewerContext(viewer),
          loadResource,
        );
        viewer.getScene().add(lod.container);
        lod.tick(camera);
        lod.start();
        tickers.push(() => lod.tick(camera));
        const warmupUntil = performance.now() + 12000;
        const warmup = () => {
          lod.tick(camera);
          viewer.render();
          if (performance.now() < warmupUntil) requestAnimationFrame(warmup);
        };
        requestAnimationFrame(warmup);
        setStatus(`Aholo LOD 正在流式加载：${Number(layer.gaussians || meta.counts || 0).toLocaleString()} 个高斯。`);
        await lod.onFinishSchedule();
        lod.tick(camera);
        viewer.render();
        setStatus(`Aholo LOD 已加载：${Number(layer.gaussians || meta.counts || 0).toLocaleString()} 个高斯。`);
      } else {
        const bytes = await fetchBytes(url, layer.file);
        setStatus(`正在解析 ${Math.round(bytes.byteLength / 1048576)} MB 3DGS...`);
        const fileType = SplatLoader.detectSplatFileType?.(url, bytes) ?? SplatLoader.SplatFileType.PLY;
        const ext = new URL(url).pathname.split(".").pop().toLowerCase();
        const packType = ext === "sog" ? SplatLoader.SplatPackType.Sog : (ext === "ply" ? SplatLoader.SplatPackType.SuperCompressed : SplatLoader.SplatPackType.Compressed);
        const input = ext === "ply" ? bytes : url;
        const data = await SplatLoader.parseSplatData(fileType, input, packType, {
          maxShDegree: ext === "sog" ? 0 : 3,
          maxTextureSize: 8192,
        });
        const splat = await SplatUtils.createSplat(data);
        splat.autoFreeResourceOnGpuPacked = true;

        viewer.getScene().add(splat);
        setStatus(`Aholo 已加载：${Number(layer.gaussians || 0).toLocaleString()} 个高斯。`);
      }
      setViewerConfig(viewer, {
        pixelRatio: Math.min(1, 1 / Math.max(1, window.devicePixelRatio || 1)),
        pipeline: {
          Background: {
            background: { active: BackgroundMode.BasicBackground, basic: { color: new Color(0.02, 0.025, 0.03) } },
            ground: { enabled: false },
          },
          Splatting: {
            enabled: true,
            sort: { frustumCullingEnabled: true },
          },
          TAA: { enabled: false },
        },
      });
      resetCamera(Vector3);
      const render = () => {
        for (const tick of tickers) tick();
        viewer.render();
      };
      viewer.requestRenderHandler = () => requestAnimationFrame(render);
      window.addEventListener("resize", () => resetCamera(Vector3));
      document.getElementById("reset").onclick = () => resetCamera(Vector3);
      requestAnimationFrame(render);
    }

    main().catch(error => {
      console.error(error);
      setStatus(`Aholo 加载失败：${error?.message || error}`, "error");
    });
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class RGBDOptimizeConfig:
    source: Path
    output: Path
    scene_name: str = "RGBD优化"
    max_point_count: int = 850_000
    point_stride: int = 4
    point_keep_every: int = 2
    train_frames_hint: int = 220
    trained_ply: Path | None = None
    training_metrics: Path | None = None
    training_preview: Path | None = None
    training_contact_sheet: Path | None = None
    train_gsplat: bool = False
    gsplat_python: Path | None = None
    gsplat_train_script: Path | None = None
    gsplat_max_steps: int = 200
    gsplat_max_frames: int = 16
    gsplat_image_max_size: int = 448
    gsplat_max_points: int = 80_000
    gsplat_target_gaussians: int = 50_000
    gsplat_dense_image_points_per_frame: int = 0
    gsplat_initial_scale: float = 0.0
    gsplat_device: str = "cuda"
    aholo_splat_transform_binary: Path | None = None
    aholo_convert_format: str = "spz"
    metadata_pose_convention: str = "auto"
    dry_run: bool = False


def run_rgbd_optimized(config: RGBDOptimizeConfig, log: LogFn | None = None) -> dict[str, Any]:
    logger = log or (lambda _message: None)
    source = config.source.resolve()
    output = config.output.resolve()
    reports = output / "reports"
    viewer = output / "viewer"
    assets = viewer / "assets"
    colmap_dir = assets / "colmap_rgbd"
    thumbs = colmap_dir / "thumbs"
    pose_dir = output / "rgbd_pose_init"
    for path in [reports, viewer, assets, colmap_dir, thumbs, pose_dir / "images"]:
        path.mkdir(parents=True, exist_ok=True)

    rgb_files, depth_files = _find_rgbd_pairs(source)
    if not rgb_files or not depth_files:
        raise RuntimeError(f"RGBD 数据不完整，需要 images/rgb 与 depth/depth_selected/frames2: {source}")
    if len(rgb_files) != len(depth_files):
        raise RuntimeError(f"RGB/Depth 数量不一致: rgb={len(rgb_files)}, depth={len(depth_files)}")
    logger(f"RGBD 输入: RGB {len(rgb_files)} 张, Depth {len(depth_files)} 张")

    if config.dry_run:
        return _write_dry_run(config, rgb_files, depth_files)

    cv2, np, Image = _load_rgbd_dependencies()
    metadata = _load_capture_metadata(source)
    first = Image.open(rgb_files[0]).convert("RGB")
    width, height = first.size
    intrinsics = _metadata_intrinsics(metadata, width, height) or _estimate_intrinsics(width, height)
    depth_width, depth_height = _depth_size(depth_files[0], cv2, Image)
    depth_intrinsics = _scale_intrinsics(intrinsics, depth_width / width, depth_height / height)
    logger(f"内参: fx={intrinsics['fx']:.3f}, fy={intrinsics['fy']:.3f}, cx={intrinsics['cx']:.3f}, cy={intrinsics['cy']:.3f}")
    metadata_poses = _metadata_poses(metadata, len(rgb_files), np)
    pose_convention = "rgbd_pnp"
    pose_transform_note = None
    if metadata_poses:
        conversion, pose_convention, pose_transform_note = _metadata_pose_coordinate_conversion(
            metadata,
            config.metadata_pose_convention,
            np,
        )
        poses = [pose @ conversion for pose in metadata_poses] if conversion is not None else metadata_poses
        stats = [
            {"i": index, "name": path.name, "ok": True, "matches": 0, "obj": 0, "inliers": 0, "step": None, "pos": pose[:3, 3].round(5).tolist(), "source": pose_convention}
            for index, (path, pose) in enumerate(zip(rgb_files, poses))
        ]
        logger(f"使用 metadata 真实/外部 pose: {len(poses)} 帧, 坐标约定={pose_convention}")
        if pose_transform_note:
            logger(f"pose 坐标转换: {pose_transform_note}")
    else:
        poses, stats = _estimate_rgbd_pnp(
            rgb_files,
            depth_files,
            intrinsics=intrinsics,
            depth_intrinsics=depth_intrinsics,
            target_size=(width, height),
            log=logger,
        )
    ok_steps = sum(1 for item in stats if item["ok"])
    pose_source = "metadata" if metadata_poses else "RGBD-PnP估计"
    logger(f"{pose_source} 完成: ok_steps={ok_steps}, fail_steps={len(stats) - ok_steps}")

    _copy_images(rgb_files, pose_dir / "images")
    _write_colmap_text(colmap_dir, rgb_files, poses, intrinsics, width, height)
    points, colors = _fuse_rgbd_points(
        rgb_files,
        depth_files,
        poses,
        depth_intrinsics=depth_intrinsics,
        target_size=(width, height),
        max_point_count=config.max_point_count,
        stride=config.point_stride,
        keep_every=config.point_keep_every,
        log=logger,
    )
    point_cloud = assets / "rgbd_fused_points.ply"
    pose_point_cloud = pose_dir / "colmap_sparse_points.ply"
    _write_binary_rgb_ply(point_cloud, points, colors)
    _write_ascii_rgb_ply(pose_point_cloud, points, colors)
    _write_transforms(pose_dir, rgb_files, poses, intrinsics, width, height, stats)
    _write_thumbnails(rgb_files, thumbs, Image)

    trained = _run_or_resolve_training(config, pose_dir, source, logger)
    spark_asset = None
    aholo_asset = None
    aholo_transform: dict[str, Any] = {"attempted": False, "ok": False}
    ply_info: dict[str, Any] = {}
    if trained.get("ply"):
        source_ply = Path(trained["ply"])
        copied = assets / source_ply.name
        if source_ply.resolve() != copied.resolve():
            shutil.copy2(source_ply, copied)
        spark_asset = _prepare_spark_ply(copied, viewer)
        ply_info = read_ply_header(spark_asset or copied).to_jsonable()
        logger(f"Spark 3DGS 已接入: {spark_asset or copied}")
        aholo_asset, aholo_transform = _prepare_aholo_asset(config, spark_asset or copied, assets, logger)
    copied_training = _copy_training_previews(trained, assets)

    trajectory = _trajectory_bounds(poses)
    point_bounds = _point_bounds(points)
    training_metrics = _read_json(Path(trained["metrics"])) if trained.get("metrics") else {}
    viewer_manifest = _build_viewer_manifest(
        config=config,
        viewer=viewer,
        point_cloud=point_cloud,
        colmap_dir=colmap_dir,
        spark_asset=spark_asset,
        aholo_asset=aholo_asset,
        aholo_transform=aholo_transform,
        spark_info=ply_info,
        metrics={
            "rgb_frames": len(rgb_files),
            "depth_frames": len(depth_files),
            "point_count": int(len(points)),
            "rgbd_pnp_ok_steps": ok_steps,
            "rgbd_pnp_fail_steps": len(stats) - ok_steps,
            "median_inliers_ok": _median([s["inliers"] for s in stats if s["ok"]]),
            "trajectory_bbox_min": trajectory[0],
            "trajectory_bbox_max": trajectory[1],
            "point_bbox_min": point_bounds[0],
            "point_bbox_max": point_bounds[1],
            "gsplat_metrics": str(trained["metrics"]) if trained.get("metrics") else None,
            "aholo_transform": aholo_transform,
            **_extract_training_metrics(training_metrics),
        },
    )
    (viewer / "viewer_manifest.json").write_text(json.dumps(viewer_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (viewer / "index.html").write_text(VIEWER_HTML, encoding="utf-8")
    aholo_viewer = _write_aholo_viewer(viewer, aholo_asset or spark_asset, ply_info)

    run_manifest = {
        "schema": "images23dgs.rgbd_optimized.v1",
        "source": str(source),
        "output": str(output),
        "pose_source": pose_source,
        "real_pose": bool(metadata_poses),
        "photo_risk": "低" if metadata_poses else "中",
        "rgbd": {
            "rgb_frames": len(rgb_files),
            "depth_frames": len(depth_files),
            "ok_steps": ok_steps,
            "fail_steps": len(stats) - ok_steps,
            "pose_dir": str(pose_dir),
            "point_cloud": str(point_cloud),
            "pose_coordinate_convention": pose_convention,
            "pose_coordinate_transform": pose_transform_note,
        },
        "training": trained,
        "copied_training_assets": copied_training,
        "viewer": {"index_html": str(viewer / "index.html"), "manifest_json": str(viewer / "viewer_manifest.json"), "aholo_index_html": str(aholo_viewer) if aholo_viewer else None},
        "source_view_qa": str(output / "source_view_qa.html"),
        "dry_run": False,
    }
    (reports / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_source_view_qa(output / "source_view_qa.html", run_manifest, viewer_manifest, copied_training)
    return run_manifest


def _find_rgbd_pairs(source: Path) -> tuple[list[Path], list[Path]]:
    image_dir = None
    for name in ["images", "rgb", "color", "colors"]:
        candidate = source / name
        if candidate.is_dir():
            image_dir = candidate
            break
    if image_dir is None:
        image_dir = source
    depth_dir = None
    for name in ["depth_selected", "depth", "depths", "frames2"]:
        candidate = source / name
        if candidate.is_dir():
            depth_dir = candidate
            break
    rgb_files = _rgb_files(image_dir)
    depth_files = _depth_files(depth_dir) if depth_dir else []
    common = {p.stem for p in rgb_files} & {p.stem for p in depth_files}
    if common:
        rgb_files = [p for p in rgb_files if p.stem in common]
        depth_files = [p for p in depth_files if p.stem in common]
    return rgb_files, depth_files


def _rgb_files(path: Path | None) -> list[Path]:
    if path is None:
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=_frame_sort_key)


def _depth_files(path: Path | None) -> list[Path]:
    if path is None:
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".exr"}], key=_frame_sort_key)


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def _load_rgbd_dependencies():
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("RGBD 优化需要安装 opencv-python、numpy、Pillow") from exc
    return cv2, np, Image


def _estimate_intrinsics(width: int, height: int) -> dict[str, float]:
    base_width, base_height = 1920.0, 1440.0
    return {
        "fx": 1335.671142578125 * (width / base_width),
        "fy": 1335.671142578125 * (height / base_height),
        "cx": 963.5147705078125 * (width / base_width),
        "cy": 723.2529296875 * (height / base_height),
    }


def _scale_intrinsics(intrinsics: dict[str, float], sx: float, sy: float) -> dict[str, float]:
    return {"fx": intrinsics["fx"] * sx, "fy": intrinsics["fy"] * sy, "cx": intrinsics["cx"] * sx, "cy": intrinsics["cy"] * sy}


def _load_capture_metadata(source: Path) -> dict[str, Any]:
    path = source / "metadata.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _metadata_intrinsics(metadata: dict[str, Any], width: int, height: int) -> dict[str, float] | None:
    coeffs = metadata.get("perFrameIntrinsicCoeffs")
    if isinstance(coeffs, list) and coeffs and isinstance(coeffs[0], list) and len(coeffs[0]) >= 4:
        fx, fy, cx, cy = coeffs[0][:4]
        return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
    matrix = metadata.get("K")
    if isinstance(matrix, list) and len(matrix) >= 9:
        return {"fx": float(matrix[0]), "fy": float(matrix[4]), "cx": float(matrix[6]), "cy": float(matrix[7])}
    for key in ["intrinsics", "camera_intrinsics"]:
        value = metadata.get(key)
        if isinstance(value, dict) and all(name in value for name in ["fx", "fy", "cx", "cy"]):
            return {"fx": float(value["fx"]), "fy": float(value["fy"]), "cx": float(value["cx"]), "cy": float(value["cy"])}
    return None


def _metadata_poses(metadata: dict[str, Any], frame_count: int, np) -> list[Any]:
    poses = metadata.get("poses")
    if not isinstance(poses, list) or len(poses) < frame_count:
        return []
    matrices = []
    for item in poses[:frame_count]:
        if not isinstance(item, list):
            return []
        if len(item) == 16:
            matrices.append(np.asarray(item, dtype=np.float64).reshape(4, 4))
            continue
        if len(item) >= 7:
            qx, qy, qz, qw, tx, ty, tz = [float(x) for x in item[:7]]
            matrices.append(_quat_xyzw_to_matrix(qx, qy, qz, qw, tx, ty, tz, np))
            continue
        return []
    return matrices


def _metadata_pose_coordinate_conversion(metadata: dict[str, Any], convention: str, np) -> tuple[Any | None, str, str | None]:
    normalized = (convention or "auto").strip().lower().replace("-", "_")
    if normalized in {"none", "opencv", "cv", "gsplat", "open_cv"}:
        return None, "metadata_opencv", None
    if normalized in {"arkit", "arkit_to_cv", "arkit_to_opencv", "ios", "iphone"}:
        return _arkit_to_opencv_camera_basis(np), "metadata_arkit_to_cv", "camera basis diag(1,-1,-1): ARKit -Z/Y-up to OpenCV/gsplat +Z/Y-down"
    if normalized == "auto" and _looks_like_arkit_rgbd_metadata(metadata):
        return _arkit_to_opencv_camera_basis(np), "metadata_arkit_to_cv_auto", "auto-detected EXR_RGBD/ARKit metadata; camera basis diag(1,-1,-1)"
    return None, "metadata_auto_no_conversion", None


def _looks_like_arkit_rgbd_metadata(metadata: dict[str, Any]) -> bool:
    poses = metadata.get("poses")
    has_pose_quat = isinstance(poses, list) and bool(poses) and isinstance(poses[0], list) and len(poses[0]) >= 7
    return bool(
        has_pose_quat
        and isinstance(metadata.get("perFrameIntrinsicCoeffs"), list)
        and "dw" in metadata
        and "dh" in metadata
        and ("initPose" in metadata or "frameTimestamps" in metadata)
    )


def _arkit_to_opencv_camera_basis(np):
    return np.diag([1.0, -1.0, -1.0, 1.0])


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float, tx: float, ty: float, tz: float, np):
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        qx, qy, qz, qw = (quat / norm).tolist()
        rotation = np.asarray(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]
    return transform


def _depth_size(path: Path, cv2, Image) -> tuple[int, int]:
    if path.suffix.lower() == ".exr":
        depth = _read_depth_meters(path, cv2, None)
        return int(depth.shape[1]), int(depth.shape[0])
    if path.suffix.lower() == ".npy":
        depth = _read_depth_meters(path, cv2, None)
        return int(depth.shape[1]), int(depth.shape[0])
    with Image.open(path) as depth_image:
        return depth_image.size


def _read_depth_meters(path: Path, cv2, np):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        import numpy as _np

        depth = _np.asarray(_np.load(path), dtype=_np.float32)
    elif suffix == ".exr":
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"无法读取 EXR depth: {path}")
        if depth.ndim == 3:
            counts = [(depth[:, :, channel] > 0).sum() for channel in range(depth.shape[2])]
            depth = depth[:, :, int(max(range(depth.shape[2]), key=lambda channel: counts[channel]))]
        import numpy as _np

        depth = _np.asarray(depth, dtype=_np.float32)
    else:
        from PIL import Image as _Image
        import numpy as _np

        depth = _np.asarray(_Image.open(path), dtype=_np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    finite = depth[depth > 0]
    if finite.size and float(finite.max()) > 20.0:
        depth = depth * 0.001
    return depth


def _estimate_rgbd_pnp(
    rgb_files: list[Path],
    depth_files: list[Path],
    *,
    intrinsics: dict[str, float],
    depth_intrinsics: dict[str, float],
    target_size: tuple[int, int],
    log: LogFn,
):
    cv2, np, Image = _load_rgbd_dependencies()
    width, height = target_size
    camera_matrix = np.array(
        [[intrinsics["fx"], 0, intrinsics["cx"]], [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]],
        np.float64,
    )
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    poses = [np.eye(4, dtype=np.float64)]
    stats: list[dict[str, Any]] = []
    last_kp = last_des = last_depth = None
    for index, (rgb, depth) in enumerate(zip(rgb_files, depth_files)):
        gray = cv2.imread(str(rgb), cv2.IMREAD_GRAYSCALE)
        dep = _read_depth_meters(depth, cv2, np)
        kp, des = orb.detectAndCompute(gray, None)
        if index == 0:
            last_kp, last_des, last_depth = kp, des, dep
            continue
        ok = False
        inliers = matches_n = objn = 0
        step = None
        if last_des is not None and des is not None and len(last_des) > 20 and len(des) > 20:
            good = []
            for pair in matcher.knnMatch(last_des, des, k=2):
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            matches_n = len(good)
            objects = []
            img2 = []
            depth_height, depth_width = last_depth.shape[:2]
            for match in good:
                u, v = last_kp[match.queryIdx].pt
                du = int(round(u * depth_width / width))
                dv = int(round(v * depth_height / height))
                if not (0 <= du < depth_width and 0 <= dv < depth_height):
                    continue
                z = float(last_depth[dv, du])
                if not (0.35 < z < 5.5):
                    continue
                objects.append([(du - depth_intrinsics["cx"]) / depth_intrinsics["fx"] * z, (dv - depth_intrinsics["cy"]) / depth_intrinsics["fy"] * z, z])
                img2.append(kp[match.trainIdx].pt)
            objn = len(objects)
            if objn >= 12:
                objects_np = np.asarray(objects, np.float32)
                img2_np = np.asarray(img2, np.float32)
                ret, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
                    objects_np,
                    img2_np,
                    camera_matrix,
                    None,
                    iterationsCount=300,
                    reprojectionError=5.0,
                    confidence=0.995,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                inliers = 0 if inlier_idx is None else len(inlier_idx)
                if ret and inliers >= 20:
                    rotation, _ = cv2.Rodrigues(rvec)
                    transform = np.eye(4)
                    transform[:3, :3] = rotation
                    transform[:3, 3] = tvec[:, 0]
                    step = float(np.linalg.norm(transform[:3, 3]))
                    if step < 1.25:
                        poses.append(poses[-1] @ np.linalg.inv(transform))
                        ok = True
        if not ok:
            poses.append(poses[-1].copy())
        stats.append(
            {
                "i": index,
                "name": rgb.name,
                "ok": ok,
                "matches": matches_n,
                "obj": objn,
                "inliers": inliers,
                "step": step,
                "pos": poses[-1][:3, 3].round(5).tolist(),
            }
        )
        if index % 50 == 0 or index == len(rgb_files) - 1:
            log(f"RGBD-PnP: {index + 1}/{len(rgb_files)}")
        last_kp, last_des, last_depth = kp, des, dep
    return poses, stats


def _fuse_rgbd_points(
    rgb_files: list[Path],
    depth_files: list[Path],
    poses: list[Any],
    *,
    depth_intrinsics: dict[str, float],
    target_size: tuple[int, int],
    max_point_count: int,
    stride: int,
    keep_every: int,
    log: LogFn,
):
    cv2, np, Image = _load_rgbd_dependencies()
    width, height = target_size
    depth_width, depth_height = _depth_size(depth_files[0], cv2, Image)
    ys = np.arange(0, depth_height, stride, dtype=np.float64)
    xs = np.arange(0, depth_width, stride, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    points = []
    colors = []
    for index, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        dep = _read_depth_meters(depth_path, cv2, np).astype(np.float64)
        z = dep[flat_y.astype(np.int32), flat_x.astype(np.int32)]
        mask = (z > 0.35) & (z < 5.5)
        if not np.any(mask):
            continue
        cam = np.stack(
            [
                (flat_x[mask] - depth_intrinsics["cx"]) / depth_intrinsics["fx"] * z[mask],
                (flat_y[mask] - depth_intrinsics["cy"]) / depth_intrinsics["fy"] * z[mask],
                z[mask],
            ],
            axis=1,
        )
        c2w = poses[index]
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        keep = np.arange(world.shape[0]) % keep_every == 0
        world = world[keep]
        sx = np.clip((flat_x[mask][keep] * (width / depth_width)).astype(np.int32), 0, width - 1)
        sy = np.clip((flat_y[mask][keep] * (height / depth_height)).astype(np.int32), 0, height - 1)
        points.append(world.astype(np.float32))
        colors.append(rgb[sy, sx, :].astype(np.uint8))
        if index % 100 == 0:
            log(f"融合点云: {index + 1}/{len(rgb_files)}")
    pts = np.concatenate(points, axis=0) if points else np.zeros((0, 3), np.float32)
    cols = np.concatenate(colors, axis=0) if colors else np.zeros((0, 3), np.uint8)
    if len(pts) > max_point_count:
        selected = np.linspace(0, len(pts) - 1, max_point_count, dtype=np.int64)
        pts = pts[selected]
        cols = cols[selected]
    log(f"融合点云完成: {len(pts)} points")
    return pts, cols


def _copy_images(files: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = output / path.name
        if not target.is_file():
            shutil.copy2(path, target)


def _write_thumbnails(files: list[Path], output: Path, Image) -> None:
    output.mkdir(parents=True, exist_ok=True)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    for path in files:
        target = output / f"{path.stem}.jpg"
        if target.is_file():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 270), resample)
        image.save(target, quality=86)


def _write_colmap_text(colmap_dir: Path, rgb_files: list[Path], poses: list[Any], intrinsics: dict[str, float], width: int, height: int) -> None:
    colmap_dir.mkdir(parents=True, exist_ok=True)
    (colmap_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {intrinsics['fx']:.9f} {intrinsics['fy']:.9f} {intrinsics['cx']:.9f} {intrinsics['cy']:.9f}\n",
        encoding="utf-8",
    )
    with (colmap_dir / "images.txt").open("w", encoding="utf-8") as stream:
        stream.write("# Image list with two lines of data per image:\n")
        stream.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        stream.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        stream.write(f"# Number of images: {len(rgb_files)}, mean observations per image: 1\n")
        for index, path in enumerate(rgb_files, start=1):
            c2w = poses[index - 1]
            rotation = c2w[:3, :3].T
            translation = -rotation @ c2w[:3, 3]
            quat = _mat_to_qwxyz(rotation)
            stream.write(
                f"{index} {quat[0]:.12f} {quat[1]:.12f} {quat[2]:.12f} {quat[3]:.12f} "
                f"{translation[0]:.12f} {translation[1]:.12f} {translation[2]:.12f} 1 {path.name}\n0 0 -1\n"
            )
    (colmap_dir / "points3D.txt").write_text("# Empty: RGBD fused point cloud is shown in point layer.\n", encoding="utf-8")


def _mat_to_qwxyz(rotation) -> list[float]:
    import numpy as np

    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quat = [(0.25 * scale), (rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
    else:
        axis = int(np.argmax([rotation[0, 0], rotation[1, 1], rotation[2, 2]]))
        if axis == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quat = [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale]
        elif axis == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quat = [(rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale]
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quat = [(rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale]
    q = np.array(quat, dtype=np.float64)
    q /= np.linalg.norm(q)
    return [float(x) for x in q]


def _write_transforms(pose_dir: Path, rgb_files: list[Path], poses: list[Any], intrinsics: dict[str, float], width: int, height: int, stats: list[dict[str, Any]]) -> None:
    import numpy as np

    frames = []
    for path, pose in zip(rgb_files, poses):
        frames.append(
            {
                "file_path": f"images/{path.name}",
                "colmap_transform_matrix": pose.astype(float).tolist(),
                "fl_x": intrinsics["fx"],
                "fl_y": intrinsics["fy"],
                "cx": intrinsics["cx"],
                "cy": intrinsics["cy"],
                "w": width,
                "h": height,
            }
        )
    trajectory = np.array([pose[:3, 3] for pose in poses])
    metadata = {
        "source": "RGBD odometry via ORB depth PnP RANSAC",
        "ok_steps": sum(item["ok"] for item in stats),
        "fail_steps": sum(not item["ok"] for item in stats),
        "median_inliers_ok": _median([item["inliers"] for item in stats if item["ok"]]),
        "trajectory_bbox_min": trajectory.min(0).tolist(),
        "trajectory_bbox_max": trajectory.max(0).tolist(),
    }
    transforms = {"camera_model": "PINHOLE", "fl_x": intrinsics["fx"], "fl_y": intrinsics["fy"], "cx": intrinsics["cx"], "cy": intrinsics["cy"], "w": width, "h": height, "frames": frames, "metadata": metadata}
    (pose_dir / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    (pose_dir / "rgbd_odometry_stats.json").write_text(json.dumps({"frames": len(rgb_files), "stats": stats, "summary": metadata}, indent=2), encoding="utf-8")


def _write_binary_rgb_ply(path: Path, points, colors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(
            (
                f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode("ascii")
        )
        for point, color in zip(points, colors):
            stream.write(struct.pack("<fffBBB", float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2])))


def _write_ascii_rgb_ply(path: Path, points, colors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, color in zip(points, colors):
            stream.write(f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def _resolve_training_artifacts(config: RGBDOptimizeConfig, source: Path) -> dict[str, str | None]:
    explicit = {
        "ply": str(config.trained_ply) if config.trained_ply else None,
        "metrics": str(config.training_metrics) if config.training_metrics else None,
        "preview": str(config.training_preview) if config.training_preview else None,
        "contact_sheet": str(config.training_contact_sheet) if config.training_contact_sheet else None,
    }
    if explicit["ply"]:
        return explicit
    candidates = []
    if source.name.endswith("_input"):
        candidates.append(source.with_name(source.name[: -len("_input")]))
    candidates.append(source.parent / source.name.replace("_input", ""))
    for root in candidates:
        if not root.is_dir():
            continue
        for ply in sorted(root.glob("gsplat*/pose_init_trained_scene.ply")):
            return {
                "ply": str(ply),
                "metrics": str(ply.with_name("pose_init_training_metrics.json")) if ply.with_name("pose_init_training_metrics.json").is_file() else None,
                "preview": str(ply.with_name("pose_init_training_preview.png")) if ply.with_name("pose_init_training_preview.png").is_file() else None,
                "contact_sheet": str(ply.with_name("pose_init_training_contact_sheet.png")) if ply.with_name("pose_init_training_contact_sheet.png").is_file() else None,
            }
    return explicit


def _run_or_resolve_training(config: RGBDOptimizeConfig, pose_dir: Path, source: Path, log: LogFn) -> dict[str, str | None]:
    if config.train_gsplat:
        if config.gsplat_python is None or config.gsplat_train_script is None:
            raise RuntimeError("训练 3DGS 需要 gsplat_python 和 gsplat_train_script")
        output_dir = config.output / "gsplat_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ply = output_dir / "pose_init_trained_scene.ply"
        preview = output_dir / "pose_init_training_preview.png"
        contact = output_dir / "pose_init_training_contact_sheet.png"
        metrics = output_dir / "pose_init_training_metrics.json"
        command = [
            str(config.gsplat_python),
            str(config.gsplat_train_script),
            "--pose-init-dir",
            str(pose_dir),
            "--output-dir",
            str(output_dir),
            "--output-ply",
            str(output_ply),
            "--preview-output",
            str(preview),
            "--multi-preview-output",
            str(contact),
            "--metrics-output",
            str(metrics),
            "--max-steps",
            str(config.gsplat_max_steps),
            "--max-frames",
            str(config.gsplat_max_frames),
            "--image-max-size",
            str(config.gsplat_image_max_size),
            "--max-points",
            str(config.gsplat_max_points),
            "--target-gaussians",
            str(config.gsplat_target_gaussians),
            "--dense-image-points-per-frame",
            str(config.gsplat_dense_image_points_per_frame),
            "--device",
            config.gsplat_device,
        ]
        if config.gsplat_initial_scale > 0:
            command.extend(["--initial-scale", str(config.gsplat_initial_scale)])
        log("开始从零训练 3DGS: " + " ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log("[gsplat] " + line.rstrip())
        process.stdout.close()
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"3DGS 训练失败，退出码 {code}")
        if not output_ply.is_file() or not metrics.is_file():
            raise RuntimeError("3DGS 训练结束但缺少输出 PLY 或 metrics")
        log(f"3DGS 训练完成: {output_ply}")
        return {"ply": str(output_ply), "metrics": str(metrics), "preview": str(preview) if preview.is_file() else None, "contact_sheet": str(contact) if contact.is_file() else None}
    return _resolve_training_artifacts(config, source)


def _copy_training_previews(training: dict[str, str | None], assets: Path) -> dict[str, str]:
    copied = {}
    for key in ["metrics", "preview", "contact_sheet"]:
        value = training.get(key)
        if not value:
            continue
        source = Path(value)
        if not source.is_file():
            continue
        target = assets / source.name
        shutil.copy2(source, target)
        copied[key] = target.name
    return copied


def _build_viewer_manifest(
    *,
    config: RGBDOptimizeConfig,
    viewer: Path,
    point_cloud: Path,
    colmap_dir: Path,
    spark_asset: Path | None,
    aholo_asset: Path | None,
    aholo_transform: dict[str, Any],
    spark_info: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    aholo_source = aholo_asset or spark_asset
    return {
        "schema": "images23dgs.layered_viewer.v1",
        "title": config.scene_name,
        "package_dir": None,
        "layers": {
            "spark_3dgs": {
                "label": "Spark 3DGS visual from RGBD optimization",
                "available": bool(spark_asset and spark_asset.is_file() and spark_info.get("has_3dgs_fields")),
                "file": _relative(viewer, spark_asset),
                "gaussians": spark_info.get("vertex_count"),
                "has_3dgs_fields": spark_info.get("has_3dgs_fields", False),
                "source": "trained gsplat PLY attached to RGBD-PnP trajectory" if spark_asset else "No trained gsplat artifact supplied or discovered.",
            },
            "aholo_3dgs": {
                "label": "Aholo 3DGS high-performance preview",
                "available": bool(aholo_source and aholo_source.is_file() and spark_info.get("has_3dgs_fields")),
                "viewer": "aholo/index.html" if aholo_source and aholo_source.is_file() and spark_info.get("has_3dgs_fields") else None,
                "file": _relative(viewer, aholo_source),
                "fallback_file": _relative(viewer, spark_asset),
                "gaussians": spark_info.get("vertex_count"),
                "format": (aholo_source.suffix.lower().lstrip(".") if aholo_source else None),
                "transform": aholo_transform,
                "source": "Aholo viewer uses converted SPZ/SOG/LOD when available, otherwise falls back to the trained gsplat PLY.",
            },
            "point_cloud": {
                "label": "RGBD fused point cloud using PnP odometry",
                "available": True,
                "file": _relative(viewer, point_cloud),
                "points": metrics["point_count"],
                "source": "Depth images back-projected with estimated RGBD-PnP camera poses.",
            },
            "collision_mesh": {"label": "collision mesh unavailable", "available": False, "obj": None, "source": "Not generated yet."},
            "mjcf_geoms": {"label": "MJCF unavailable", "available": False, "xml": None},
            "colmap": {
                "label": "RGBD-PnP camera trajectory + image planes",
                "available": True,
                "images_txt": _relative(viewer, colmap_dir / "images.txt"),
                "points3D_txt": _relative(viewer, colmap_dir / "points3D.txt"),
                "cameras_txt": _relative(viewer, colmap_dir / "cameras.txt"),
                "transform": {"scale_xyz": [1, 1, 1], "translate_xyz": [0, 0, 0], "note": "Camera poses are RGBD-PnP estimates, not COLMAP SfM."},
                "image_thumbnail_base": _relative(viewer, colmap_dir / "thumbs"),
                "image_thumbnail_ext": ".jpg",
                "image_thumbnail_max": min(220, int(metrics["rgb_frames"])),
                "camera_frustum_depth": 0.55,
                "source": "RGBD odometry from input RGB/depth frames.",
            },
        },
        "metrics": metrics,
        "notes": [
            "This job really ran RGBD ORB + depth PnP RANSAC and fused the depth point cloud.",
            "COLMAP layer is a text-model compatibility layer backed by RGBD-PnP poses.",
        ],
}


def _prepare_aholo_asset(config: RGBDOptimizeConfig, source_ply: Path, assets: Path, log: LogFn) -> tuple[Path | None, dict[str, Any]]:
    fmt = config.aholo_convert_format.lower().lstrip(".")
    result: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "format": fmt,
        "source": str(source_ply),
        "output": None,
        "command": None,
        "summary": None,
    }
    if fmt not in {"spz", "sog", "splat", "ply"}:
        result["summary"] = f"unsupported format:{fmt}"
        log(f"Aholo 转换跳过: {result['summary']}")
        return None, result
    if fmt == "ply":
        result.update({"ok": True, "output": str(source_ply), "summary": "using ply directly"})
        return source_ply, result
    binary = config.aholo_splat_transform_binary
    if binary is None:
        result["summary"] = "missing:splat-transform"
        log("Aholo 转换跳过: 未配置 splat-transform")
        return None, result
    if not (binary.exists() or shutil.which(str(binary))):
        result["summary"] = f"missing:{binary}"
        log(f"Aholo 转换跳过: {result['summary']}")
        return None, result
    output = assets / f"{source_ply.stem}.{fmt}"
    command = [str(binary), "create", str(source_ply), str(output)]
    result.update({"attempted": True, "output": str(output), "command": command})
    log(f"Aholo 转换开始: {' '.join(command)}")
    env = os.environ.copy()
    env["PATH"] = f"{binary.parent}:{env.get('PATH', '')}"
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800, env=env)
    except Exception as exc:
        result["summary"] = str(exc)
        log(f"Aholo 转换失败: {exc}")
        return None, result
    output_text = (completed.stdout or "").strip()
    result["summary"] = output_text.splitlines()[-1] if output_text else f"exit={completed.returncode}"
    if completed.returncode != 0 or not output.is_file():
        log(f"Aholo 转换失败: {result['summary']}")
        return None, result
    result["ok"] = True
    result["size"] = output.stat().st_size
    log(f"Aholo 转换完成: {output} ({output.stat().st_size / 1024**2:.1f} MB)")
    return output, result


def _write_aholo_viewer(viewer: Path, spark_asset: Path | None, spark_info: dict[str, Any]) -> Path | None:
    if not (spark_asset and spark_asset.is_file() and spark_info.get("has_3dgs_fields")):
        return None
    aholo_dir = viewer / "aholo"
    aholo_dir.mkdir(parents=True, exist_ok=True)
    index = aholo_dir / "index.html"
    index.write_text(AHOLO_VIEWER_HTML, encoding="utf-8")
    return index


def _write_source_view_qa(path: Path, run_manifest: dict[str, Any], viewer_manifest: dict[str, Any], copied_training: dict[str, str]) -> None:
    metrics = viewer_manifest.get("metrics", {})
    preview = copied_training.get("preview")
    contact = copied_training.get("contact_sheet")
    preview_html = f"<img src='viewer/assets/{preview}' alt='training preview'>" if preview else "<p>未发现训练 preview。</p>"
    contact_html = f"<img src='viewer/assets/{contact}' alt='training contact sheet'>" if contact else ""
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>源视角质检</title>"
        "<style>body{font-family:system-ui;background:#101417;color:#eef;padding:28px}img{max-width:100%;border:1px solid #34404a;border-radius:6px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{background:#161d24;border:1px solid #2c3844;border-radius:8px;padding:14px}</style>"
        "<h1>源视角质检</h1>"
        "<div class='grid'>"
        f"<div class='card'><b>RGB帧</b><p>{metrics.get('rgb_frames')}</p></div>"
        f"<div class='card'><b>Depth帧</b><p>{metrics.get('depth_frames')}</p></div>"
        f"<div class='card'><b>RGBD-PnP成功步</b><p>{metrics.get('rgbd_pnp_ok_steps')}</p></div>"
        f"<div class='card'><b>融合点</b><p>{metrics.get('point_count')}</p></div>"
        f"<div class='card'><b>Spark 3DGS</b><p>{metrics.get('trained_gaussian_count') or '未接入训练结果'}</p></div>"
        f"<div class='card'><b>PSNR</b><p>{metrics.get('gsplat_multi_preview_mean_psnr') or metrics.get('gsplat_final_preview_psnr') or '未训练'}</p></div>"
        "</div>"
        "<h2>训练预览</h2>"
        f"{preview_html}{contact_html}"
        f"<h2>Manifest</h2><pre>{json.dumps(run_manifest, indent=2, ensure_ascii=False)}</pre>",
        encoding="utf-8",
    )


def _write_dry_run(config: RGBDOptimizeConfig, rgb_files: list[Path], depth_files: list[Path]) -> dict[str, Any]:
    reports = config.output / "reports"
    viewer = config.output / "viewer"
    reports.mkdir(parents=True, exist_ok=True)
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "index.html").write_text(VIEWER_HTML, encoding="utf-8")
    (viewer / "viewer_manifest.json").write_text(
        json.dumps(
            {
                "schema": "images23dgs.layered_viewer.v1",
                "title": config.scene_name,
                "layers": {
                    "spark_3dgs": {"available": False},
                    "aholo_3dgs": {"available": False},
                    "point_cloud": {"available": False},
                    "collision_mesh": {"available": False},
                    "mjcf_geoms": {"available": False},
                    "colmap": {"available": False},
                },
                "metrics": {"rgb_frames": len(rgb_files), "depth_frames": len(depth_files)},
                "notes": ["RGBD dry-run did not execute PnP or point fusion."],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {"schema": "images23dgs.rgbd_optimized.v1", "source": str(config.source), "output": str(config.output), "rgbd": {"rgb_frames": len(rgb_files), "depth_frames": len(depth_files)}, "dry_run": True}
    (reports / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (config.output / "source_view_qa.html").write_text("<!doctype html><meta charset='utf-8'><h1>RGBD dry-run</h1>", encoding="utf-8")
    return manifest


def _relative(base: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        import os

        return os.path.relpath(path.resolve(), base.resolve())


def _trajectory_bounds(poses: list[Any]) -> tuple[list[float], list[float]]:
    import numpy as np

    trajectory = np.array([pose[:3, 3] for pose in poses])
    return trajectory.min(0).tolist(), trajectory.max(0).tolist()


def _point_bounds(points) -> tuple[list[float], list[float]]:
    if len(points) == 0:
        return [0, 0, 0], [0, 0, 0]
    return points.min(0).tolist(), points.max(0).tolist()


def _median(values: list[float | int]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_training_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "trained_gaussian_count",
        "max_steps",
        "frame_count",
        "source_frame_size",
        "final_loss",
        "final_preview_psnr",
        "multi_preview_mean_psnr",
        "mean_step_psnr_last_10",
    ]
    return {f"gsplat_{key}": metrics[key] for key in keys if key in metrics}
