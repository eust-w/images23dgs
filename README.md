# images23dgs

`images23dgs` wraps the local DISCOVERSE/Real2Sim scene generation flow and
adds an ArtiFixer branch for sparse-view captures.

The default command is:

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --prompt "static indoor room with task-relevant furniture and floor"
```

What it does:

1. Copies the source images into the output package.
2. Runs COLMAP when available and records registration quality.
3. Chooses the ArtiFixer branch when the source image count or COLMAP
   registered ratio is too low.
4. Runs `python -m real2sim generate-scene` from the local GS-Playground /
   DISCOVERSE-compatible Real2Sim checkout.
5. Validates the final `3dgs/scene.ply` header and writes a run manifest.

The wrapper leaves Real2Sim pruning disabled by default because pruning requires
the optional `gaussian_renderer` package. Use `--enable-prune` only in an
environment where that optional dependency is installed.

The current local defaults are:

- DISCOVERSE root: `/Users/d-robotics/workSpace/DISCOVERSE`
- Real2Sim root: `/Users/d-robotics/workSpace/gs_playground`
- Direct path threshold: at least `24` images and `0.65` COLMAP registered ratio

## Artifixer

ArtiFixer is not a light image inpaint command. Its official workflow expects a
COLMAP scene, a CUDA environment with the ArtiFixer-compatible 3DGRUT submodule,
and the `nvidia/ArtiFixer` checkpoint. This wrapper wires the official stages:

```bash
python -m data_processing.prepare_colmap_artifixer_inputs
python -m model_eval.run_inference
python -m data_processing.run_artifixer3d
python -m model_eval.run_inference
```

Run with an existing ArtiFixer checkout and checkpoint:

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --artifixer-root /path/to/ArtiFixer \
  --artifixer-checkpoint /data/artifixer-checkpoints/artifixer-14b.pt \
  --force-artifixer
```

To just inspect the decisions and generated commands without running GPU work:

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --dry-run
```

Artifacts:

- `reports/run_manifest.json`: stage decisions, paths, PLY validation
- `run_commands.sh`: reproducible shell command log
- `colmap/`: Artifixer-compatible COLMAP scene root, including `images/` and `sparse/0/`
- `artifixer/`: ArtiFixer prepared scene and corrected frames when used
- `discoverse_package/3dgs/scene.ply`: generated scene Gaussian asset
- `viewer/`: Spark.js / Three.js layered preview

## Layered Viewer

Each run writes `viewer/index.html` and `viewer/viewer_manifest.json`.

Serve it from the run output root:

```bash
cd /tmp/images23dgs_scene
python3 viewer/serve.py --port 18123
```

Open:

```text
http://127.0.0.1:18123/viewer/index.html
```

The viewer exposes independent layers:

- Spark 3DGS visual render from a full 3DGS PLY. ASCII PLY files are converted
  to `viewer/assets/*_spark_binary.ply` for Spark.js compatibility.
- 3DGS point-cloud preview from Gaussian centers and RGB or `f_dc_*` colors.
- Collision/scene mesh preview from the packaged OBJ.
- MJCF geom overlay from `mjcf/scene.xml`.
- COLMAP sparse points and camera trajectory when `sparse_text/*.txt` exists.

## Wuying Web Product

For a completely clean Wuying machine, use the single-file installer. It
downloads this GitHub repository, creates `/opt/images23dgs_app`, installs the
web product, and runs the product doctor check.

```bash
curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer -o images23dgs-installer
chmod +x images23dgs-installer
sudo ./images23dgs-installer --start
```

For the private GitHub repository, pass a token that can read the repository:

```bash
export GITHUB_TOKEN=ghp_xxx
curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer \
  -o images23dgs-installer
chmod +x images23dgs-installer
sudo -E ./images23dgs-installer --start
```

If the source is already present on the machine:

```bash
bash scripts/install_wuying.sh
```

Start the frontend, backend, and worker:

```bash
bash scripts/start_wuying.sh
```

The default service listens on `0.0.0.0:18123`. For local browser access through
SSH tunneling, map the remote port to a local port such as `18124`.
