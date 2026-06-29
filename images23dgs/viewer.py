from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .discoverse import DiscoverseResult
from .ply import read_ply_header


@dataclass(frozen=True)
class ViewerResult:
    viewer_dir: Path
    index_html: Path
    manifest_json: Path
    url_hint: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "viewer_dir": str(self.viewer_dir),
            "index_html": str(self.index_html),
            "manifest_json": str(self.manifest_json),
            "url_hint": self.url_hint,
        }


def write_layered_viewer(
    *,
    output_root: Path,
    discoverse_result: DiscoverseResult,
    final_scene: Path | None,
    scene_name: str,
) -> ViewerResult:
    viewer_dir = output_root / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    index_html = viewer_dir / "index.html"
    manifest_json = viewer_dir / "viewer_manifest.json"

    package_dir = discoverse_result.package_dir
    package_manifest = _read_json(package_dir / "package_manifest.json")
    scene_manifest = package_manifest.get("scene", {}) if isinstance(package_manifest, dict) else {}

    package_scene_ply = package_dir / "3dgs" / "scene.ply"
    scene_ply = package_scene_ply if package_scene_ply.is_file() else final_scene
    spark_scene_ply = _prepare_spark_ply(scene_ply, viewer_dir)
    scene_mesh = _resolve_package_path(package_dir, scene_manifest.get("mesh_obj")) or package_dir / "meshes" / "scene.obj"
    highres_mesh = package_dir / "meshes" / "scene_highres.obj"
    mjcf_xml = package_dir / "mjcf" / "scene.xml"

    colmap_sparse_text = output_root / "colmap" / "sparse_text"
    points3d = colmap_sparse_text / "points3D.txt"
    images_txt = colmap_sparse_text / "images.txt"
    cameras_txt = colmap_sparse_text / "cameras.txt"

    ply_info = _ply_info(scene_ply)
    manifest = {
        "schema": "images23dgs.layered_viewer.v1",
        "title": scene_name,
        "package_dir": _relative_or_none(viewer_dir, package_dir),
        "layers": {
            "spark_3dgs": {
                "label": "Spark 3DGS visual",
                "available": bool(spark_scene_ply and spark_scene_ply.is_file() and ply_info.get("has_3dgs_fields")),
                "file": _relative_or_none(viewer_dir, spark_scene_ply),
                "gaussians": ply_info.get("vertex_count"),
                "has_3dgs_fields": ply_info.get("has_3dgs_fields", False),
                "source_file": _relative_or_none(viewer_dir, scene_ply),
            },
            "point_cloud": {
                "label": "3DGS point preview",
                "available": bool(scene_ply and scene_ply.is_file()),
                "file": _relative_or_none(viewer_dir, scene_ply),
                "points": ply_info.get("vertex_count"),
                "source": "same 3DGS centers, colored from RGB or f_dc_* fields",
            },
            "collision_mesh": {
                "label": "collision/scene mesh",
                "available": scene_mesh.is_file() or highres_mesh.is_file(),
                "obj": _relative_or_none(viewer_dir, scene_mesh if scene_mesh.is_file() else highres_mesh),
                "highres_obj": _relative_or_none(viewer_dir, highres_mesh if highres_mesh.is_file() else None),
            },
            "mjcf_geoms": {
                "label": "MJCF geom overlay",
                "available": mjcf_xml.is_file(),
                "xml": _relative_or_none(viewer_dir, mjcf_xml),
            },
            "colmap": {
                "label": "COLMAP sparse layers",
                "available": images_txt.is_file() or points3d.is_file(),
                "images_txt": _relative_or_none(viewer_dir, images_txt if images_txt.is_file() else None),
                "points3D_txt": _relative_or_none(viewer_dir, points3d if points3d.is_file() else None),
                "cameras_txt": _relative_or_none(viewer_dir, cameras_txt if cameras_txt.is_file() else None),
                "transform": {
                    "scale_xyz": [1.0, 1.0, 1.0],
                    "translate_xyz": [0.0, 0.0, 0.0],
                    "note": "identity unless upstream Real2Sim writes explicit COLMAP-to-visual transform metadata",
                },
            },
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index_html.write_text(VIEWER_HTML, encoding="utf-8")
    _write_server(viewer_dir / "serve.py")
    return ViewerResult(
        viewer_dir=viewer_dir,
        index_html=index_html,
        manifest_json=manifest_json,
        url_hint="python3 viewer/serve.py --port 18123",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_package_path(package_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else package_dir / path


def _relative_or_none(base: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _ply_info(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        return read_ply_header(path).to_jsonable()
    except Exception as exc:
        return {"error": str(exc)}


def _prepare_spark_ply(path: Path | None, viewer_dir: Path) -> Path | None:
    if path is None or not path.is_file():
        return None
    header = read_ply_header(path)
    if header.format == "binary_little_endian":
        return path
    if header.format != "ascii":
        return path
    output = viewer_dir / "assets" / f"{path.stem}_spark_binary.ply"
    if output.is_file() and output.stat().st_mtime >= path.stat().st_mtime:
        return output
    _convert_ascii_ply_to_binary(path, output)
    return output


def _convert_ascii_ply_to_binary(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header_lines: list[str] = []
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            header_lines.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
        properties: list[tuple[str, str]] = []
        vertex_count = 0
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            if in_vertex and parts[0] == "property" and len(parts) >= 3:
                properties.append((parts[1], parts[2]))

        packers = [_ply_struct_format(kind) for kind, _name in properties]
        binary_header = ["ply", "format binary_little_endian 1.0"]
        for line in header_lines[2:]:
            if line.startswith("format "):
                continue
            if line == "end_header":
                binary_header.append("comment converted_from_ascii_for_sparkjs")
                binary_header.append("end_header")
            else:
                binary_header.append(line)

        with output.open("wb") as out:
            out.write(("\n".join(binary_header) + "\n").encode("ascii", "replace"))
            written = 0
            for line in stream:
                if not line.strip():
                    continue
                values = line.split()
                if len(values) < len(properties):
                    continue
                for index, fmt in enumerate(packers):
                    out.write(struct.pack(fmt, _coerce_ply_value(values[index], fmt)))
                written += 1
                if written >= vertex_count:
                    break
    if written != vertex_count:
        raise RuntimeError(f"converted {written} vertices, expected {vertex_count}: {source}")


def _ply_struct_format(kind: str) -> str:
    mapping = {
        "float": "<f",
        "float32": "<f",
        "double": "<d",
        "float64": "<d",
        "uchar": "<B",
        "uint8": "<B",
        "char": "<b",
        "int8": "<b",
        "ushort": "<H",
        "uint16": "<H",
        "short": "<h",
        "int16": "<h",
        "uint": "<I",
        "uint32": "<I",
        "int": "<i",
        "int32": "<i",
    }
    if kind not in mapping:
        raise ValueError(f"unsupported PLY type for Spark conversion: {kind}")
    return mapping[kind]


def _coerce_ply_value(value: str, fmt: str) -> int | float:
    if fmt.endswith(("f", "d")):
        return float(value)
    return int(float(value))


def _write_server(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ply": "application/octet-stream",
        ".obj": "text/plain",
        ".mtl": "text/plain",
        ".xml": "application/xml",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18123)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/viewer/index.html")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


VIEWER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>images23dgs layered viewer</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #101214;
      color: #eef2f4;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #viewport {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      display: block;
    }
    .panel {
      position: fixed;
      left: 14px;
      top: 14px;
      width: min(360px, calc(100vw - 28px));
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px solid rgba(238, 242, 244, 0.16);
      border-radius: 8px;
      background: rgba(16, 18, 20, 0.88);
      box-shadow: 0 16px 44px rgba(0, 0, 0, 0.32);
      backdrop-filter: blur(14px);
      box-sizing: border-box;
    }
    .title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 14px;
      font-weight: 700;
    }
    .status {
      min-height: 38px;
      color: #b9c4ca;
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
      white-space: pre-line;
    }
    .layers {
      display: grid;
      gap: 6px;
    }
    .layer {
      display: grid;
      grid-template-columns: 20px 1fr auto;
      align-items: center;
      gap: 8px;
      min-height: 30px;
      font-size: 13px;
    }
    .layer input {
      width: 16px;
      height: 16px;
      margin: 0;
    }
    .state {
      color: #8fa0a9;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .controls {
      display: grid;
      grid-template-columns: 80px 1fr 48px;
      gap: 8px;
      align-items: center;
      color: #cad3d8;
      font-size: 12px;
    }
    input[type="range"] {
      width: 100%;
    }
    .buttons {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    button {
      height: 32px;
      border: 1px solid rgba(238, 242, 244, 0.18);
      border-radius: 6px;
      color: #eef2f4;
      background: #283039;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    button:hover {
      background: #35404a;
    }
    button:disabled {
      color: #738089;
      cursor: not-allowed;
      background: #1f2429;
    }
    .bar {
      height: 3px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(238, 242, 244, 0.12);
    }
    .bar > div {
      width: 0%;
      height: 100%;
      background: #66c6a8;
      transition: width 120ms ease-out;
    }
    @media (max-width: 640px) {
      .panel {
        left: 8px;
        top: 8px;
        padding: 10px;
      }
      .buttons {
        grid-template-columns: 1fr;
      }
    }
  </style>
  <script type="importmap">
    {
      "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.179.1/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/",
        "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/2.1.0/spark.module.js"
      }
    }
  </script>
</head>
<body>
  <canvas id="viewport"></canvas>
  <section class="panel">
    <div class="title">
      <span id="title">images23dgs</span>
      <span id="summary" class="state">loading</span>
    </div>
    <div class="layers">
      <label class="layer"><input id="layerSpark" type="checkbox" /><span>Spark 3DGS</span><span id="stateSpark" class="state">off</span></label>
      <label class="layer"><input id="layerPointCloud" type="checkbox" /><span>3DGS point cloud</span><span id="statePointCloud" class="state">off</span></label>
      <label class="layer"><input id="layerMesh" type="checkbox" /><span>Collision mesh</span><span id="stateMesh" class="state">off</span></label>
      <label class="layer"><input id="layerMjcf" type="checkbox" /><span>MJCF geoms</span><span id="stateMjcf" class="state">off</span></label>
      <label class="layer"><input id="layerColmap" type="checkbox" /><span>COLMAP</span><span id="stateColmap" class="state">off</span></label>
    </div>
    <div class="controls">
      <label for="pointSize">Point</label>
      <input id="pointSize" type="range" min="0.002" max="0.05" step="0.001" value="0.012" />
      <span id="pointSizeText">0.012</span>
    </div>
    <div class="controls">
      <label for="meshAlpha">Mesh</label>
      <input id="meshAlpha" type="range" min="0.05" max="1" step="0.05" value="0.35" />
      <span id="meshAlphaText">0.35</span>
    </div>
    <div class="buttons">
      <button id="loadVisual">Visual</button>
      <button id="loadDiagnostic">Diagnostic</button>
      <button id="resetView">Reset</button>
    </div>
    <div class="bar"><div id="progress"></div></div>
    <div id="status" class="status">Loading manifest...</div>
  </section>

  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

    const MANIFEST_URL = "viewer_manifest.json";
    const APP_VERSION = "images23dgs-layered-viewer-v1";
    const SH_C0 = 0.28209479177387814;

    const canvas = document.getElementById("viewport");
    const statusEl = document.getElementById("status");
    const progressEl = document.getElementById("progress");
    const titleEl = document.getElementById("title");
    const summaryEl = document.getElementById("summary");
    const pointSizeEl = document.getElementById("pointSize");
    const pointSizeTextEl = document.getElementById("pointSizeText");
    const meshAlphaEl = document.getElementById("meshAlpha");
    const meshAlphaTextEl = document.getElementById("meshAlphaText");
    const controlsByLayer = {
      spark: document.getElementById("layerSpark"),
      pointCloud: document.getElementById("layerPointCloud"),
      mesh: document.getElementById("layerMesh"),
      mjcf: document.getElementById("layerMjcf"),
      colmap: document.getElementById("layerColmap")
    };
    const statesByLayer = {
      spark: document.getElementById("stateSpark"),
      pointCloud: document.getElementById("statePointCloud"),
      mesh: document.getElementById("stateMesh"),
      mjcf: document.getElementById("stateMjcf"),
      colmap: document.getElementById("stateColmap")
    };

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setClearColor(0x101214, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    const grid = new THREE.GridHelper(8, 16, 0x46515b, 0x252b31);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AmbientLight(0xffffff, 0.58));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
    keyLight.position.set(-4, -5, 8);
    scene.add(keyLight);

    const state = {
      manifest: null,
      sparkModulePromise: null,
      sparkRenderer: null,
      sparkMesh: null,
      pointCloud: null,
      mesh: null,
      mjcf: null,
      colmap: null,
      lastBox: new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)),
      counts: { gaussians: 0, points: 0, mesh: 0, mjcf: 0, cameras: 0, sparse: 0 }
    };

    function cacheBust(url) {
      const glue = url.includes("?") ? "&" : "?";
      return `${url}${glue}v=${encodeURIComponent(APP_VERSION)}`;
    }

    function assetUrl(relativePath) {
      return new URL(relativePath, new URL(".", new URL(MANIFEST_URL, window.location.href))).href;
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function setProgress(value) {
      progressEl.style.width = `${Math.max(0, Math.min(100, value))}%`;
    }

    function setLayerState(layer, value) {
      statesByLayer[layer].textContent = value;
    }

    function setLayerChecked(layer, checked) {
      controlsByLayer[layer].checked = checked;
    }

    function updateSummary() {
      const active = [];
      if (state.sparkMesh) active.push("3DGS");
      if (state.pointCloud) active.push("points");
      if (state.mesh) active.push("mesh");
      if (state.mjcf) active.push("mjcf");
      if (state.colmap) active.push("colmap");
      summaryEl.textContent = active.length ? active.join(" + ") : "none";
    }

    function removeObject(object) {
      if (!object) return;
      object.traverse?.((child) => {
        child.geometry?.dispose?.();
        if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose?.());
        else child.material?.dispose?.();
      });
      object.parent?.remove(object);
    }

    function boxFromObject(object) {
      const box = new THREE.Box3().setFromObject(object);
      if (!Number.isFinite(box.min.x) || !Number.isFinite(box.max.x)) return null;
      return box;
    }

    function fitBox(box, remember = true) {
      if (!box) box = state.lastBox;
      if (remember) state.lastBox = box.clone();
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);
      const radius = Math.max(size.length() * 0.65, 0.8);
      controls.target.copy(center);
      camera.near = Math.max(radius / 1600, 0.005);
      camera.far = Math.max(radius * 14, 100);
      camera.position.set(center.x - radius * 0.6, center.y - radius * 1.25, center.z + radius * 0.62);
      camera.updateProjectionMatrix();
      controls.update();
      grid.position.set(center.x, center.y, Math.min(box.min.z, 0));
      grid.scale.setScalar(Math.max(radius / 4, 1));
    }

    function resize() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    }

    function animate() {
      resize();
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    async function loadManifest() {
      if (state.manifest) return state.manifest;
      const response = await fetch(cacheBust(MANIFEST_URL), { cache: "no-store" });
      if (!response.ok) throw new Error(`manifest ${response.status}`);
      state.manifest = await response.json();
      titleEl.textContent = state.manifest.title || "images23dgs";
      const layers = state.manifest.layers;
      controlsByLayer.spark.disabled = !layers.spark_3dgs.available;
      controlsByLayer.pointCloud.disabled = !layers.point_cloud.available;
      controlsByLayer.mesh.disabled = !layers.collision_mesh.available;
      controlsByLayer.mjcf.disabled = !layers.mjcf_geoms.available;
      controlsByLayer.colmap.disabled = !layers.colmap.available;
      state.counts.gaussians = layers.spark_3dgs.gaussians || 0;
      state.counts.points = layers.point_cloud.points || 0;
      setStatus(`Manifest loaded.\n3DGS ${state.counts.gaussians.toLocaleString()} | point preview ${state.counts.points.toLocaleString()}`);
      return state.manifest;
    }

    async function loadSparkModule() {
      if (!state.sparkModulePromise) state.sparkModulePromise = import("@sparkjsdev/spark");
      return state.sparkModulePromise;
    }

    async function ensureSparkRenderer() {
      if (state.sparkRenderer) return state.sparkRenderer;
      const { SparkRenderer } = await loadSparkModule();
      state.sparkRenderer = new SparkRenderer({ renderer, focalAdjustment: 1.0 });
      scene.add(state.sparkRenderer);
      return state.sparkRenderer;
    }

    async function loadSpark() {
      await loadManifest();
      if (state.sparkMesh) return;
      const layer = state.manifest.layers.spark_3dgs;
      if (!layer.available) throw new Error("Spark layer is unavailable");
      setLayerState("spark", "loading");
      setStatus("Loading Spark 3DGS...");
      setProgress(8);
      const { SplatMesh } = await loadSparkModule();
      await ensureSparkRenderer();
      const mesh = await new Promise((resolve, reject) => {
        const splat = new SplatMesh({
          url: cacheBust(assetUrl(layer.file)),
          onProgress: (event) => {
            if (event?.lengthComputable && event.total) setProgress(8 + (event.loaded / event.total) * 82);
          },
          onLoad: () => resolve(splat)
        });
        setTimeout(() => reject(new Error("Spark 3DGS load timed out")), 180000);
      });
      state.sparkMesh = mesh;
      scene.add(mesh);
      setLayerChecked("spark", true);
      setLayerState("spark", "on");
      setProgress(100);
      setStatus(`Spark 3DGS loaded: ${(layer.gaussians || 0).toLocaleString()} splats.`);
      updateSummary();
      try {
        const box = mesh.getBoundingBox?.(true);
        if (box) fitBox(box);
      } catch {
        fitBox(state.lastBox);
      }
    }

    function unloadSpark() {
      removeObject(state.sparkMesh);
      state.sparkMesh = null;
      setLayerChecked("spark", false);
      setLayerState("spark", "off");
      updateSummary();
    }

    function findHeaderEnd(bytes) {
      const needle = new TextEncoder().encode("end_header");
      outer: for (let i = 0; i <= bytes.length - needle.length; i++) {
        for (let j = 0; j < needle.length; j++) {
          if (bytes[i + j] !== needle[j]) continue outer;
        }
        let end = i + needle.length;
        while (end < bytes.length && (bytes[end] === 10 || bytes[end] === 13)) end++;
        return end;
      }
      return -1;
    }

    function parsePlyHeader(buffer) {
      const bytes = new Uint8Array(buffer);
      const headerEnd = findHeaderEnd(bytes.subarray(0, Math.min(bytes.length, 1024 * 1024)));
      if (headerEnd < 0) throw new Error("PLY header not found");
      const header = new TextDecoder("ascii").decode(bytes.subarray(0, headerEnd));
      const format = /format\s+(\S+)/.exec(header)?.[1];
      const vertexCount = Number(/element vertex\s+(\d+)/.exec(header)?.[1] || 0);
      const properties = [];
      let inVertex = false;
      for (const line of header.split(/\r?\n/)) {
        const parts = line.trim().split(/\s+/);
        if (parts[0] === "element" && parts[1] === "vertex") {
          inVertex = true;
          continue;
        }
        if (parts[0] === "element" && parts[1] !== "vertex") inVertex = false;
        if (inVertex && parts[0] === "property" && parts.length >= 3) {
          properties.push({ type: parts[1], name: parts[2] });
        }
      }
      const typeSize = { float: 4, float32: 4, double: 8, uchar: 1, uint8: 1, char: 1, int8: 1, ushort: 2, uint16: 2, short: 2, int16: 2, uint: 4, uint32: 4, int: 4, int32: 4 };
      let stride = 0;
      for (const property of properties) {
        property.offset = stride;
        property.size = typeSize[property.type] || 0;
        stride += property.size;
      }
      return { header, headerEnd, format, vertexCount, properties, stride };
    }

    function readProperty(view, base, property) {
      const offset = base + property.offset;
      switch (property.type) {
        case "float":
        case "float32": return view.getFloat32(offset, true);
        case "double": return view.getFloat64(offset, true);
        case "uchar":
        case "uint8": return view.getUint8(offset);
        case "char":
        case "int8": return view.getInt8(offset);
        case "ushort":
        case "uint16": return view.getUint16(offset, true);
        case "short":
        case "int16": return view.getInt16(offset, true);
        case "uint":
        case "uint32": return view.getUint32(offset, true);
        case "int":
        case "int32": return view.getInt32(offset, true);
        default: return 0;
      }
    }

    function shToLinear(value) {
      return Math.min(1, Math.max(0, value * SH_C0 + 0.5));
    }

    function pointColor(propByName, getter, fallback = [0.86, 0.9, 0.92]) {
      if (propByName.has("red") && propByName.has("green") && propByName.has("blue")) {
        return [getter("red") / 255, getter("green") / 255, getter("blue") / 255];
      }
      if (propByName.has("f_dc_0") && propByName.has("f_dc_1") && propByName.has("f_dc_2")) {
        return [shToLinear(getter("f_dc_0")), shToLinear(getter("f_dc_1")), shToLinear(getter("f_dc_2"))];
      }
      return fallback;
    }

    function parsePointCloud(buffer, maxPoints = 260000) {
      const parsed = parsePlyHeader(buffer);
      const propByName = new Map(parsed.properties.map((property) => [property.name, property]));
      for (const name of ["x", "y", "z"]) {
        if (!propByName.has(name)) throw new Error(`Missing PLY property: ${name}`);
      }
      const count = Math.min(parsed.vertexCount, maxPoints);
      const strideSelect = Math.max(1, Math.floor(parsed.vertexCount / Math.max(1, count)));
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      const bbox = new THREE.Box3();
      let written = 0;

      if (parsed.format === "binary_little_endian") {
        const view = new DataView(buffer, parsed.headerEnd);
        for (let source = 0; source < parsed.vertexCount && written < count; source += strideSelect) {
          const base = source * parsed.stride;
          const getter = (name) => readProperty(view, base, propByName.get(name));
          written = writePoint(written, getter("x"), getter("y"), getter("z"), pointColor(propByName, getter), positions, colors, bbox);
        }
      } else if (parsed.format === "ascii") {
        const body = new TextDecoder("utf-8").decode(new Uint8Array(buffer, parsed.headerEnd));
        const names = parsed.properties.map((property) => property.name);
        const xIdx = names.indexOf("x");
        const yIdx = names.indexOf("y");
        const zIdx = names.indexOf("z");
        const lines = body.split(/\r?\n/);
        for (let source = 0; source < lines.length && written < count; source += strideSelect) {
          const parts = lines[source].trim().split(/\s+/);
          if (parts.length < names.length) continue;
          const getter = (name) => Number(parts[names.indexOf(name)]);
          written = writePoint(written, Number(parts[xIdx]), Number(parts[yIdx]), Number(parts[zIdx]), pointColor(propByName, getter), positions, colors, bbox);
        }
      } else {
        throw new Error(`Unsupported PLY format: ${parsed.format}`);
      }
      return {
        vertexCount: written,
        positions: positions.slice(0, written * 3),
        colors: colors.slice(0, written * 3),
        bbox
      };
    }

    function writePoint(index, x, y, z, color, positions, colors, bbox) {
      if (![x, y, z].every(Number.isFinite)) return index;
      positions[index * 3] = x;
      positions[index * 3 + 1] = y;
      positions[index * 3 + 2] = z;
      colors[index * 3] = color[0];
      colors[index * 3 + 1] = color[1];
      colors[index * 3 + 2] = color[2];
      bbox.expandByPoint(new THREE.Vector3(x, y, z));
      return index + 1;
    }

    async function fetchArrayBuffer(url) {
      const response = await fetch(cacheBust(url), { cache: "no-store" });
      if (!response.ok) throw new Error(`${url} ${response.status}`);
      const total = Number(response.headers.get("content-length") || 0);
      if (!response.body || !total) return response.arrayBuffer();
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        setProgress((loaded / total) * 78);
      }
      const out = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return out.buffer;
    }

    async function loadPointCloud() {
      await loadManifest();
      if (state.pointCloud) return;
      const layer = state.manifest.layers.point_cloud;
      if (!layer.available) throw new Error("point cloud layer is unavailable");
      setLayerState("pointCloud", "loading");
      setStatus("Loading point cloud preview...");
      setProgress(0);
      const parsed = parsePointCloud(await fetchArrayBuffer(assetUrl(layer.file)));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(parsed.colors, 3));
      geometry.computeBoundingSphere();
      const material = new THREE.PointsMaterial({
        size: Number(pointSizeEl.value),
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.98
      });
      state.pointCloud = new THREE.Points(geometry, material);
      state.pointCloud.frustumCulled = false;
      scene.add(state.pointCloud);
      setLayerChecked("pointCloud", true);
      setLayerState("pointCloud", "on");
      setProgress(100);
      setStatus(`Point cloud loaded: ${parsed.vertexCount.toLocaleString()} sampled points.`);
      updateSummary();
      fitBox(parsed.bbox);
    }

    function unloadPointCloud() {
      removeObject(state.pointCloud);
      state.pointCloud = null;
      setLayerChecked("pointCloud", false);
      setLayerState("pointCloud", "off");
      updateSummary();
    }

    async function loadMesh() {
      await loadManifest();
      if (state.mesh) return;
      const layer = state.manifest.layers.collision_mesh;
      if (!layer.available || !layer.obj) throw new Error("mesh layer is unavailable");
      setLayerState("mesh", "loading");
      setStatus("Loading collision/scene mesh...");
      const loader = new OBJLoader();
      const object = await loader.loadAsync(cacheBust(assetUrl(layer.obj)));
      object.traverse((child) => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: 0x6aa6d8,
          roughness: 0.85,
          metalness: 0.0,
          transparent: true,
          opacity: Number(meshAlphaEl.value),
          side: THREE.DoubleSide
        });
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(child.geometry),
          new THREE.LineBasicMaterial({ color: 0xd7e7f3, transparent: true, opacity: 0.42 })
        );
        child.add(edges);
      });
      state.mesh = object;
      scene.add(object);
      setLayerChecked("mesh", true);
      setLayerState("mesh", "on");
      setStatus("Collision/scene mesh loaded.");
      updateSummary();
      fitBox(boxFromObject(object));
    }

    function unloadMesh() {
      removeObject(state.mesh);
      state.mesh = null;
      setLayerChecked("mesh", false);
      setLayerState("mesh", "off");
      updateSummary();
    }

    function parseNumbers(value) {
      return (value || "").trim().split(/\s+/).filter(Boolean).map(Number);
    }

    async function loadMjcf() {
      await loadManifest();
      if (state.mjcf) return;
      const layer = state.manifest.layers.mjcf_geoms;
      if (!layer.available || !layer.xml) throw new Error("MJCF layer is unavailable");
      setLayerState("mjcf", "loading");
      setStatus("Loading MJCF geoms...");
      const response = await fetch(cacheBust(assetUrl(layer.xml)), { cache: "no-store" });
      if (!response.ok) throw new Error(`MJCF ${response.status}`);
      const doc = new DOMParser().parseFromString(await response.text(), "application/xml");
      const geoms = [...doc.querySelectorAll("geom")].filter((geom) => (geom.getAttribute("type") || "sphere") === "box");
      const limit = Math.min(geoms.length, 20000);
      const group = new THREE.Group();
      group.name = "mjcf_geoms";
      const geometry = new THREE.BoxGeometry(1, 1, 1);
      const material = new THREE.MeshBasicMaterial({ color: 0xf0b85c, transparent: true, opacity: Math.min(0.45, Number(meshAlphaEl.value)), wireframe: false, depthWrite: false });
      const instanced = new THREE.InstancedMesh(geometry, material, limit);
      const matrix = new THREE.Matrix4();
      const color = new THREE.Color();
      const box = new THREE.Box3();
      for (let i = 0; i < limit; i++) {
        const geom = geoms[i];
        const pos = parseNumbers(geom.getAttribute("pos"));
        const size = parseNumbers(geom.getAttribute("size"));
        const rgba = parseNumbers(geom.getAttribute("rgba"));
        const scale = new THREE.Vector3((size[0] || 0.01) * 2, (size[1] || 0.01) * 2, (size[2] || 0.01) * 2);
        const center = new THREE.Vector3(pos[0] || 0, pos[1] || 0, pos[2] || 0);
        matrix.compose(center, new THREE.Quaternion(), scale);
        instanced.setMatrixAt(i, matrix);
        color.setRGB(rgba[0] ?? 0.95, rgba[1] ?? 0.72, rgba[2] ?? 0.36);
        instanced.setColorAt(i, color);
        box.expandByPoint(center);
      }
      instanced.instanceMatrix.needsUpdate = true;
      if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
      group.add(instanced);
      state.mjcf = group;
      scene.add(group);
      setLayerChecked("mjcf", true);
      setLayerState("mjcf", "on");
      setStatus(`MJCF geoms loaded: ${limit.toLocaleString()} boxes${geoms.length > limit ? " sampled" : ""}.`);
      updateSummary();
      if (!box.isEmpty()) fitBox(box);
    }

    function unloadMjcf() {
      removeObject(state.mjcf);
      state.mjcf = null;
      setLayerChecked("mjcf", false);
      setLayerState("mjcf", "off");
      updateSummary();
    }

    function quatToMatrix(qw, qx, qy, qz) {
      return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
      ];
    }

    function cameraCenterFromColmap(qw, qx, qy, qz, tx, ty, tz) {
      const r = quatToMatrix(qw, qx, qy, qz);
      return new THREE.Vector3(
        -(r[0][0] * tx + r[1][0] * ty + r[2][0] * tz),
        -(r[0][1] * tx + r[1][1] * ty + r[2][1] * tz),
        -(r[0][2] * tx + r[1][2] * ty + r[2][2] * tz)
      );
    }

    function transformColmapPoint(point) {
      const transform = state.manifest.layers.colmap.transform || {};
      const scale = transform.scale_xyz || [1, 1, 1];
      const translate = transform.translate_xyz || [0, 0, 0];
      return new THREE.Vector3(point.x * scale[0] + translate[0], point.y * scale[1] + translate[1], point.z * scale[2] + translate[2]);
    }

    function parseColmapImageCenters(text) {
      const lines = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => line && !line.startsWith("#"));
      const centers = [];
      for (let i = 0; i < lines.length; i += 2) {
        const parts = lines[i].split(/\s+/);
        if (parts.length < 10) continue;
        const nums = parts.slice(1, 8).map(Number);
        if (nums.some((value) => !Number.isFinite(value))) continue;
        centers.push(transformColmapPoint(cameraCenterFromColmap(...nums)));
      }
      return centers;
    }

    function parseColmapSparsePoints(text, maxPoints = 200000) {
      const raw = text.split(/\r?\n/).filter((line) => line.trim() && !line.startsWith("#"));
      const stride = Math.max(1, Math.floor(raw.length / maxPoints));
      const positions = [];
      const colors = [];
      const box = new THREE.Box3();
      for (let i = 0; i < raw.length; i += stride) {
        const parts = raw[i].trim().split(/\s+/);
        if (parts.length < 7) continue;
        const point = transformColmapPoint(new THREE.Vector3(Number(parts[1]), Number(parts[2]), Number(parts[3])));
        if (![point.x, point.y, point.z].every(Number.isFinite)) continue;
        positions.push(point.x, point.y, point.z);
        colors.push(Number(parts[4]) / 255, Number(parts[5]) / 255, Number(parts[6]) / 255);
        box.expandByPoint(point);
      }
      return { positions: new Float32Array(positions), colors: new Float32Array(colors), count: positions.length / 3, box };
    }

    async function loadColmap() {
      await loadManifest();
      if (state.colmap) return;
      const layer = state.manifest.layers.colmap;
      if (!layer.available) throw new Error("COLMAP layer is unavailable");
      setLayerState("colmap", "loading");
      setStatus("Loading COLMAP layers...");
      const group = new THREE.Group();
      group.name = "colmap_layers";
      const box = new THREE.Box3();

      if (layer.points3D_txt) {
        const response = await fetch(cacheBust(assetUrl(layer.points3D_txt)), { cache: "no-store" });
        if (response.ok) {
          const sparse = parseColmapSparsePoints(await response.text());
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute("position", new THREE.BufferAttribute(sparse.positions, 3));
          geometry.setAttribute("color", new THREE.BufferAttribute(sparse.colors, 3));
          group.add(new THREE.Points(geometry, new THREE.PointsMaterial({ size: Number(pointSizeEl.value) * 1.3, vertexColors: true, transparent: true, opacity: 0.86 })));
          box.union(sparse.box);
          state.counts.sparse = sparse.count;
        }
      }

      if (layer.images_txt) {
        const response = await fetch(cacheBust(assetUrl(layer.images_txt)), { cache: "no-store" });
        if (response.ok) {
          const centers = parseColmapImageCenters(await response.text());
          const lineGeometry = new THREE.BufferGeometry();
          const positions = new Float32Array(centers.length * 3);
          centers.forEach((center, i) => {
            positions[i * 3] = center.x;
            positions[i * 3 + 1] = center.y;
            positions[i * 3 + 2] = center.z;
            box.expandByPoint(center);
          });
          lineGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          group.add(new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: 0x66c6a8, transparent: true, opacity: 0.86 })));
          const markerGeometry = new THREE.SphereGeometry(0.045, 8, 6);
          const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x66c6a8, transparent: true, opacity: 0.92 });
          const step = Math.max(1, Math.floor(centers.length / 100));
          for (let i = 0; i < centers.length; i += step) {
            const marker = new THREE.Mesh(markerGeometry, markerMaterial);
            marker.position.copy(centers[i]);
            group.add(marker);
          }
          state.counts.cameras = centers.length;
        }
      }

      state.colmap = group;
      scene.add(group);
      setLayerChecked("colmap", true);
      setLayerState("colmap", "on");
      setStatus(`COLMAP loaded: ${state.counts.cameras.toLocaleString()} cameras, ${state.counts.sparse.toLocaleString()} sparse points.`);
      updateSummary();
      if (!box.isEmpty()) fitBox(box);
    }

    function unloadColmap() {
      removeObject(state.colmap);
      state.colmap = null;
      setLayerChecked("colmap", false);
      setLayerState("colmap", "off");
      updateSummary();
    }

    function updatePointSize() {
      pointSizeTextEl.textContent = Number(pointSizeEl.value).toFixed(3);
      if (state.pointCloud?.material) state.pointCloud.material.size = Number(pointSizeEl.value);
      state.colmap?.traverse((child) => {
        if (child.isPoints) child.material.size = Number(pointSizeEl.value) * 1.3;
      });
    }

    function updateMeshAlpha() {
      meshAlphaTextEl.textContent = Number(meshAlphaEl.value).toFixed(2);
      const alpha = Number(meshAlphaEl.value);
      for (const object of [state.mesh, state.mjcf]) {
        object?.traverse((child) => {
          if (child.material) {
            child.material.opacity = child.isLineSegments ? Math.min(0.9, alpha + 0.2) : alpha;
            child.material.needsUpdate = true;
          }
        });
      }
    }

    async function toggleLayer(layer, loader, unloader) {
      try {
        if (controlsByLayer[layer].checked) await loader();
        else unloader();
      } catch (error) {
        console.error(error);
        setLayerChecked(layer, false);
        setLayerState(layer, "error");
        setStatus(error instanceof Error ? error.message : String(error));
      }
    }

    controlsByLayer.spark.addEventListener("change", () => toggleLayer("spark", loadSpark, unloadSpark));
    controlsByLayer.pointCloud.addEventListener("change", () => toggleLayer("pointCloud", loadPointCloud, unloadPointCloud));
    controlsByLayer.mesh.addEventListener("change", () => toggleLayer("mesh", loadMesh, unloadMesh));
    controlsByLayer.mjcf.addEventListener("change", () => toggleLayer("mjcf", loadMjcf, unloadMjcf));
    controlsByLayer.colmap.addEventListener("change", () => toggleLayer("colmap", loadColmap, unloadColmap));
    pointSizeEl.addEventListener("input", updatePointSize);
    meshAlphaEl.addEventListener("input", updateMeshAlpha);
    document.getElementById("resetView").addEventListener("click", () => fitBox(state.lastBox, false));
    document.getElementById("loadVisual").addEventListener("click", async () => {
      try {
        if (state.manifest?.layers.spark_3dgs.available) await loadSpark();
        else await loadPointCloud();
      } catch (error) {
        setStatus(error instanceof Error ? error.message : String(error));
      }
    });
    document.getElementById("loadDiagnostic").addEventListener("click", async () => {
      for (const [layer, loader] of [["pointCloud", loadPointCloud], ["mesh", loadMesh], ["mjcf", loadMjcf], ["colmap", loadColmap]]) {
        try {
          await loader();
        } catch (error) {
          console.warn(layer, error);
        }
      }
    });
    window.addEventListener("resize", resize);

    camera.position.set(-1.8, -3.2, 1.8);
    controls.target.set(0, 0, 0);
    controls.update();
    updatePointSize();
    updateMeshAlpha();
    animate();
    loadManifest().catch((error) => setStatus(error instanceof Error ? error.message : String(error)));
  </script>
</body>
</html>
"""
