# images23dgs

Language: [中文](README.md) | English

`images23dgs` turns image/RGBD captures into a productized 3D Gaussian workflow:
dataset import, COLMAP/RGBD pose handling, sparse-view repair hooks, 3DGS
training/preview, source-view QA, and a Chinese web console for Wuying-style
single-machine deployment.

## Quick Start

On a clean Linux/Wuying machine, install and start the web product with one
command:

```bash
curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | sudo bash -s -- --start
```

If the machine has `wget` but no `curl`:

```bash
wget -qO- https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | sudo bash -s -- --start
```

If the machine has neither downloader, install only the downloader first:

```bash
sudo apt-get update && sudo apt-get install -y curl ca-certificates && \
  curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | sudo bash -s -- --start
```

After startup, open:

```text
http://SERVER_IP:18123/
```

For SSH tunneling from your laptop:

```bash
ssh -L 18124:127.0.0.1:18123 user@server
```

Then open:

```text
http://127.0.0.1:18124/
```

## What The Installer Does

The single-file installer downloads this public GitHub repository and installs
the product under `/opt/images23dgs_app`.

It performs:

1. Installs minimal base tools when possible: `bash`, `python3`, `curl`, `tar`,
   `ca-certificates`.
2. Downloads the repository archive from GitHub.
3. Creates `/opt/images23dgs_app/src`, `/opt/images23dgs_app/workspace`, and a
   Python virtual environment.
4. Installs `images23dgs[web]` and `uv`.
5. Installs bundled Node.js 22 and `@manycore/aholo-splat-transform@1.5.1` for
   converting large 3DGS PLY files to Aholo-friendly streaming `chunk-lod`.
6. Writes `/opt/images23dgs_app/config.toml`.
7. Runs `images23dgs product doctor`.
8. With `--start`, starts the FastAPI backend, static Chinese frontend, and
   serial task worker on `0.0.0.0:18123`.

The installer supports these package managers for minimal base tools:
`apt-get`, `dnf`, `yum`, `microdnf`, `zypper`, and `apk`.

## System Requirements

Minimum for installing the web app:

- Linux shell with `sudo` or root access.
- Network access to GitHub and Python package indexes.
- `curl` or `wget` to fetch the installer.

Recommended for real reconstruction/training:

- NVIDIA GPU and working CUDA driver.
- COLMAP, default path `/usr/local/bin/colmap`.
- Real2Sim checkout, default path `/opt/gs_playground_real2sim_48q`.
- gsplat training Python, default path
  `/opt/real2sim_paper_runtime/envs/anysplat/bin/python`.
- Aholo transform, installed by default at
  `/opt/images23dgs_app/node/bin/splat-transform`.
- Optional ArtiFixer checkout/checkpoint for sparse-view repair.

Check the environment at any time:

```bash
/opt/images23dgs_app/venv/bin/images23dgs product doctor \
  --config /opt/images23dgs_app/config.toml
```

## Web Usage

The web UI is Chinese by default and contains these tabs:

- `数据集`: upload a zip/video/RGBD directory, import a server-local path, and export `EXR_RGBD.zip` packages with `EXR_RGBD/rgb/*.jpg`, `EXR_RGBD/depth/*.exr`, and `EXR_RGBD/metadata.json`.
- `任务`: create reconstruction jobs and choose templates.
- `预览`: open Aholo high-performance 3DGS, Spark 3DGS, point cloud, COLMAP trajectory, and mesh layers.
- `质检`: inspect source-view QA, metrics, COLMAP registration, and pose source.
- `设置`: view paths, port, workspace, and doctor results.
- `采集指南`: iPhone/ARKit capture format guidance.

Typical flow:

1. Open `http://SERVER_IP:18123/`.
2. In `数据集`, upload data or import a local directory.
3. In `任务`, choose a dataset and one template:
   - `快速预览`: smoke test, low cost.
   - `标准重建`: standard images23dgs pipeline.
   - `RGBD优化`: for iPhone/ARKit RGBD data with depth/pose support.
   - `高质量训练`: more frames/steps, intended for better captures.
4. Watch live logs in `任务`.
5. Open `预览` for the layered viewer.
6. Open `质检` for source-view QA and artifact links.

Job artifacts are stored under:

```text
/opt/images23dgs_app/workspace/runs/JOB_ID/
```

Important files:

- `logs/job.log`
- `reports/run_manifest.json`
- `viewer/index.html`
- `source_view_qa.html`
- `artifacts/`

## Service Commands

Install from an existing source checkout:

```bash
bash scripts/install_wuying.sh
```

Start the frontend, backend, and worker:

```bash
bash scripts/start_wuying.sh
```

Check only:

```bash
bash scripts/start_wuying.sh --check-only
```

Skip bundled Node/Aholo transform installation:

```bash
IMAGES23DGS_SKIP_AHOLO_NODE=1 bash scripts/install_wuying.sh
```

Use a custom app root:

```bash
IMAGES23DGS_APP_ROOT=/data/images23dgs_app bash scripts/install_wuying.sh
IMAGES23DGS_APP_ROOT=/data/images23dgs_app bash scripts/start_wuying.sh
```

Use the single-file installer with a custom app root:

```bash
curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | \
  sudo bash -s -- --app-root /data/images23dgs_app --start
```

## CLI Usage

Run the classic image-to-3DGS pipeline:

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --prompt "static indoor room with task-relevant furniture and floor"
```

Dry-run without GPU work:

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --dry-run
```

Force ArtiFixer sparse-view repair:

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --artifixer-root /path/to/ArtiFixer \
  --artifixer-checkpoint /data/artifixer-checkpoints/artifixer-14b.pt \
  --force-artifixer
```

Inspect image count or a PLY:

```bash
python -m images23dgs inspect --images /path/to/photos
python -m images23dgs inspect --ply /path/to/scene.ply
```

## Pipeline Behavior

For image-only input, the classic pipeline:

1. Copies source images into the output package.
2. Runs COLMAP when available and records registration quality.
3. Chooses the ArtiFixer branch when image count or COLMAP registration is too
   low.
4. Calls the local DISCOVERSE/Real2Sim-compatible generation flow.
5. Validates the final Gaussian PLY and writes a run manifest.

For RGBD/iPhone-style input, the web product can use RGB/depth data and
estimated or provided pose sources. The UI explicitly reports:

- real pose availability
- pose source, such as `RGBD-PnP估计`, `COLMAP`, or `ARKit`
- photo-level quality risk

## Layered Viewer

Each run writes `viewer/index.html` and `viewer/viewer_manifest.json`.

The viewer exposes independent layers:

- Spark 3DGS visual render from a full 3DGS PLY.
- Aholo high-performance 3DGS preview, using converted `spz` when available.
- 3DGS point-cloud preview from Gaussian centers.
- Collision/scene mesh preview when available.
- COLMAP sparse points and camera trajectory.
- Source-view QA links and artifact downloads.

To serve a standalone run directory:

```bash
cd /tmp/images23dgs_scene
python3 viewer/serve.py --port 18123
```

Then open:

```text
http://127.0.0.1:18123/viewer/index.html
```

## Configuration

Default config path:

```text
/opt/images23dgs_app/config.toml
```

Default values:

```toml
app_root = "/opt/images23dgs_app"
workspace_dir = "/opt/images23dgs_app/workspace"
real2sim_root = "/opt/gs_playground_real2sim_48q"
gsplat_python = "/opt/real2sim_paper_runtime/envs/anysplat/bin/python"
gsplat_train_script = "/opt/gs_playground_real2sim_48q/scripts/real2sim_pose_init_gsplat_train.py"
aholo_splat_transform_binary = "/opt/images23dgs_app/node/bin/splat-transform"
aholo_convert_format = "chunk-lod"
colmap_binary = "/usr/local/bin/colmap"
host = "0.0.0.0"
port = 18123
```

Edit this file when your COLMAP, Real2Sim, gsplat, ArtiFixer, or workspace paths
differ from the defaults.

## Troubleshooting

If port `18123` is already in use, either stop the old service or change
`port` in `/opt/images23dgs_app/config.toml`.

If installation succeeds but reconstruction quality is poor, check:

- source image count and overlap
- whether real pose/depth exists
- COLMAP registered image count
- source-view QA
- `reports/run_manifest.json`
- `logs/job.log`

If `doctor` reports missing optional heavy dependencies, the web app can still
start, but real reconstruction/training may fail until those dependencies are
installed.
