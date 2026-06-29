# images23dgs

语言：中文 | [English](README_EN.md)

`images23dgs` 是一个把图片/RGBD 采集数据转成 3D Gaussian 工作流的产品化工具，覆盖数据集导入、COLMAP/RGBD pose 处理、少图场景修补接口、3DGS 训练与预览、源视角质检，以及适合无影单机部署的中文 Web 控制台。

## 快速开始

在一台干净的 Linux/无影机器上，用一条命令安装并启动 Web 产品：

```bash
curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | sudo bash -s -- --start
```

如果机器有 `wget` 但没有 `curl`：

```bash
wget -qO- https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | sudo bash -s -- --start
```

如果机器连下载器都没有，先只安装下载器：

```bash
sudo apt-get update && sudo apt-get install -y curl ca-certificates && \
  curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | sudo bash -s -- --start
```

启动后打开：

```text
http://SERVER_IP:18123/
```

如果从本机通过 SSH 隧道访问远端机器：

```bash
ssh -L 18124:127.0.0.1:18123 user@server
```

然后在本机浏览器打开：

```text
http://127.0.0.1:18124/
```

## 安装器做了什么

单文件安装器会从这个公开 GitHub 仓库下载源码，并把产品安装到 `/opt/images23dgs_app`。

安装过程包括：

1. 尽可能自动补齐基础工具：`bash`、`python3`、`curl`、`tar`、`ca-certificates`。
2. 从 GitHub 下载仓库源码包。
3. 创建 `/opt/images23dgs_app/src`、`/opt/images23dgs_app/workspace` 和 Python 虚拟环境。
4. 安装 `images23dgs[web]` 和 `uv`。
5. 安装内置 Node.js 22 和 `@manycore/aholo-splat-transform`，用于把大 3DGS PLY 转成 Aholo 更适合加载的 `spz`。
6. 写入 `/opt/images23dgs_app/config.toml`。
7. 执行 `images23dgs product doctor` 做环境检查。
8. 带 `--start` 时，启动 FastAPI 后端、中文静态前端和串行任务 worker，默认监听 `0.0.0.0:18123`。

安装器支持通过这些包管理器补齐基础工具：`apt-get`、`dnf`、`yum`、`microdnf`、`zypper`、`apk`。

## 系统要求

安装 Web 产品的最低要求：

- Linux shell，具备 `sudo` 或 root 权限。
- 能访问 GitHub 和 Python 包索引。
- 有 `curl` 或 `wget` 用于下载安装器。

真实重建/训练的推荐环境：

- NVIDIA GPU 和可用 CUDA 驱动。
- COLMAP，默认路径 `/usr/local/bin/colmap`。
- Real2Sim checkout，默认路径 `/opt/gs_playground_real2sim_48q`。
- gsplat 训练 Python，默认路径 `/opt/real2sim_paper_runtime/envs/anysplat/bin/python`。
- Aholo transform，默认由安装器安装到 `/opt/images23dgs_app/node/bin/splat-transform`。
- 可选 ArtiFixer checkout/checkpoint，用于少图场景修补。

随时检查环境：

```bash
/opt/images23dgs_app/venv/bin/images23dgs product doctor \
  --config /opt/images23dgs_app/config.toml
```

## Web 使用流程

Web UI 默认中文，包含这些 tab：

- `数据集`：上传 zip/video/RGBD 目录、导入服务器本地路径，并支持导出 `EXR_RGBD.zip`，结构为 `EXR_RGBD/rgb/*.jpg`、`EXR_RGBD/depth/*.exr`、`EXR_RGBD/metadata.json`。
- `任务`：创建重建任务并选择参数模板。
- `预览`：打开 Aholo 高性能 3DGS、Spark 3DGS、点云、COLMAP 轨迹和 mesh 图层。
- `质检`：查看源视角 QA、指标、COLMAP 注册率和 pose 来源。
- `设置`：查看路径、端口、workspace 和 doctor 检查结果。
- `采集指南`：iPhone/ARKit 采集格式说明。

典型流程：

1. 打开 `http://SERVER_IP:18123/`。
2. 在 `数据集` 中上传数据，或导入服务器本地目录。
3. 在 `任务` 中选择数据集和模板：
   - `快速预览`：低成本 smoke test。
   - `标准重建`：标准 images23dgs pipeline。
   - `RGBD优化`：适合 iPhone/ARKit RGBD 数据，支持 depth/pose。
   - `高质量训练`：更多帧和训练步数，适合更好的采集数据。
4. 在 `任务` 中查看实时日志。
5. 在 `预览` 中打开分层 viewer。
6. 在 `质检` 中查看 source-view QA 和 artifact 链接。

任务产物存储在：

```text
/opt/images23dgs_app/workspace/runs/JOB_ID/
```

重要文件：

- `logs/job.log`
- `reports/run_manifest.json`
- `viewer/index.html`
- `source_view_qa.html`
- `artifacts/`

## 服务命令

已有源码 checkout 时安装：

```bash
bash scripts/install_wuying.sh
```

启动前端、后端和 worker：

```bash
bash scripts/start_wuying.sh
```

只检查环境：

```bash
bash scripts/start_wuying.sh --check-only
```

使用自定义安装目录：

```bash
IMAGES23DGS_APP_ROOT=/data/images23dgs_app bash scripts/install_wuying.sh
IMAGES23DGS_APP_ROOT=/data/images23dgs_app bash scripts/start_wuying.sh
```

跳过内置 Node/Aholo transform 安装：

```bash
IMAGES23DGS_SKIP_AHOLO_NODE=1 bash scripts/install_wuying.sh
```

通过单文件安装器指定自定义目录：

```bash
curl -fsSL https://raw.githubusercontent.com/eust-w/images23dgs/main/scripts/images23dgs-installer | \
  sudo bash -s -- --app-root /data/images23dgs_app --start
```

## CLI 使用

运行经典 image-to-3DGS pipeline：

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --prompt "static indoor room with task-relevant furniture and floor"
```

只做 dry-run，不执行 GPU 工作：

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --dry-run
```

强制使用 ArtiFixer 少图修补：

```bash
python -m images23dgs run \
  --images /path/to/photos \
  --output /tmp/images23dgs_scene \
  --artifixer-root /path/to/ArtiFixer \
  --artifixer-checkpoint /data/artifixer-checkpoints/artifixer-14b.pt \
  --force-artifixer
```

检查图片数量或 PLY：

```bash
python -m images23dgs inspect --images /path/to/photos
python -m images23dgs inspect --ply /path/to/scene.ply
```

## Pipeline 行为

对于纯图片输入，经典 pipeline 会：

1. 复制源图片到输出包。
2. 在可用时运行 COLMAP，并记录注册质量。
3. 当图片数量或 COLMAP 注册率过低时选择 ArtiFixer 分支。
4. 调用本地 DISCOVERSE/Real2Sim 兼容生成流程。
5. 校验最终 Gaussian PLY，并写入 run manifest。

对于 RGBD/iPhone 风格输入，Web 产品可以使用 RGB/depth 数据以及估计或提供的 pose。UI 会明确展示：

- 是否存在真实 pose
- pose 来源，例如 `RGBD-PnP估计`、`COLMAP` 或 `ARKit`
- 照片级质量风险

## 分层 Viewer

每次运行都会写入 `viewer/index.html` 和 `viewer/viewer_manifest.json`。

Viewer 支持独立开关这些图层：

- Spark 3DGS 视觉渲染。
- Gaussian 中心点云预览。
- 可用时显示 collision/scene mesh。
- COLMAP 稀疏点和相机轨迹。
- source-view QA 链接和 artifact 下载。

单独服务一个 run 目录：

```bash
cd /tmp/images23dgs_scene
python3 viewer/serve.py --port 18123
```

然后打开：

```text
http://127.0.0.1:18123/viewer/index.html
```

## 配置

默认配置路径：

```text
/opt/images23dgs_app/config.toml
```

默认值：

```toml
app_root = "/opt/images23dgs_app"
workspace_dir = "/opt/images23dgs_app/workspace"
real2sim_root = "/opt/gs_playground_real2sim_48q"
gsplat_python = "/opt/real2sim_paper_runtime/envs/anysplat/bin/python"
gsplat_train_script = "/opt/gs_playground_real2sim_48q/scripts/real2sim_pose_init_gsplat_train.py"
aholo_splat_transform_binary = "/opt/images23dgs_app/node/bin/splat-transform"
aholo_convert_format = "spz"
colmap_binary = "/usr/local/bin/colmap"
host = "0.0.0.0"
port = 18123
```

如果你的 COLMAP、Real2Sim、gsplat、ArtiFixer 或 workspace 路径不同，修改这个配置文件即可。

## 排障

如果 `18123` 端口已被占用，停止旧服务，或修改 `/opt/images23dgs_app/config.toml` 中的 `port`。

如果安装成功但重建质量不好，重点检查：

- 源图片数量和视角重叠
- 是否存在真实 pose/depth
- COLMAP 注册图片数量
- source-view QA
- `reports/run_manifest.json`
- `logs/job.log`

如果 `doctor` 报告缺少可选重依赖，Web app 仍可以启动，但真实重建/训练可能会失败，需要先补齐对应依赖。
