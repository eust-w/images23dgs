from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from images23dgs.product.config import ProductConfig, write_default_config, load_config
from images23dgs.product.datasets import ingest_upload, import_path_dataset, scan_dataset
from images23dgs.product.doctor import run_doctor
from images23dgs.product.store import ProductStore
from images23dgs.product.worker import JobWorker
from images23dgs.product.worker import TEMPLATES
from images23dgs.rgbd import RGBDOptimizeConfig, run_rgbd_optimized


class ProductTests(unittest.TestCase):
    def test_scan_dataset_detects_missing_pose_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images" / "frame_0001.png").write_bytes(b"png")
            (root / "frames2").mkdir()
            (root / "frames2" / "00000001.png").write_bytes(b"depth")
            (root / "data.jsonl").write_text('{"sensor":"accelerometer"}\n', encoding="utf-8")

            scan = scan_dataset(root)

            self.assertEqual(scan.image_count, 1)
            self.assertEqual(scan.depth_count, 1)
            self.assertFalse(scan.has_pose)
            self.assertEqual(scan.pose_source, "RGBD-PnP估计")
            self.assertEqual(scan.photo_risk, "中")

    def test_scan_dataset_detects_arkit_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rgb.png").write_bytes(b"png")
            (root / "transforms.json").write_text(
                json.dumps({"frames": [{"camera_to_world": [[1, 0, 0, 0]]}], "intrinsics": [[1, 0, 0]]}),
                encoding="utf-8",
            )

            scan = scan_dataset(root)

            self.assertTrue(scan.has_pose)
            self.assertTrue(scan.has_intrinsics)
            self.assertEqual(scan.photo_risk, "低")

    def test_ingest_upload_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("images/a.jpg", b"x")
            dataset_dir, scan = ingest_upload(archive, root / "datasets")
            self.assertTrue(dataset_dir.is_dir())
            self.assertEqual(scan.image_count, 1)

    def test_store_and_worker_quick_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"x")
            dataset_dir, scan = import_path_dataset(images, root / "datasets")
            config = ProductConfig(
                app_root=root,
                workspace_dir=root / "workspace",
                discoverse_root=root / "discoverse",
                real2sim_root=root / "real2sim",
                colmap_binary=root / "colmap",
            )
            (config.discoverse_root / "discoverse").mkdir(parents=True)
            (config.real2sim_root / "real2sim").mkdir(parents=True)
            (config.real2sim_root / "real2sim" / "cli.py").write_text("", encoding="utf-8")
            config.colmap_binary.write_text("#!/usr/bin/env bash\necho colmap\n", encoding="ascii")
            config.colmap_binary.chmod(0o755)
            store = ProductStore(config.db_path)
            dataset = store.create_dataset(name="images", path=dataset_dir, scan=scan.to_jsonable())
            job = store.create_job(
                dataset_id=dataset["id"],
                template="quick_preview",
                parameters={},
                run_dir=config.runs_dir / dataset["id"] / "quick_preview",
            )

            worker = JobWorker(store, config)
            worker._run_job(job)
            completed = store.get_job(job["id"])

            self.assertEqual(completed["status"], "succeeded")
            self.assertTrue((Path(completed["run_dir"]) / "reports" / "run_manifest.json").is_file())
            self.assertTrue((Path(completed["run_dir"]) / "viewer" / "index.html").is_file())

    def test_config_roundtrip_and_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            write_default_config(path)
            config = load_config(path)
            report = run_doctor(config)
            self.assertIn("checks", report)
            self.assertEqual(config.workspace_dir, Path(tmp) / "workspace")

    def test_fastapi_health_when_dependency_available(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from images23dgs.product.server import create_app
        except ModuleNotFoundError:
            self.skipTest("FastAPI is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            config = ProductConfig(app_root=Path(tmp), workspace_dir=Path(tmp) / "workspace")
            client = TestClient(create_app(config))
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])

    def test_fastapi_job_artifact_routes(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from images23dgs.product.server import create_app
        except ModuleNotFoundError:
            self.skipTest("FastAPI is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_source = root / "source"
            dataset_source.mkdir()
            (dataset_source / "rgb.png").write_bytes(b"png")
            (dataset_source / "frames2").mkdir()
            (dataset_source / "frames2" / "0001.png").write_bytes(b"depth")
            config = ProductConfig(app_root=root, workspace_dir=root / "workspace")
            with TestClient(create_app(config)) as client:
                dataset = client.post("/api/datasets/import-path", json={"path": str(dataset_source)}).json()
                job = client.post(
                    "/api/jobs",
                    json={"dataset_id": dataset["id"], "template": "rgbd_optimized", "parameters": {"dry_run": True}},
                ).json()
                for _ in range(20):
                    loaded = client.get(f"/api/jobs/{job['id']}").json()
                    if loaded["status"] == "succeeded":
                        break
                    time.sleep(0.1)

                self.assertEqual(client.get(f"/api/jobs/{job['id']}").json()["status"], "succeeded")
                artifacts = client.get(f"/api/jobs/{job['id']}/artifacts").json()
                urls = {artifact["url"] for artifact in artifacts}
                self.assertIn(f"/runs/{job['id']}/viewer/index.html", urls)
                self.assertEqual(client.get(f"/runs/{job['id']}/viewer/index.html").status_code, 200)
                self.assertEqual(client.get(f"/runs/{job['id']}/viewer/viewer_manifest.json").status_code, 200)
                self.assertEqual(client.get(f"/runs/{job['id']}/source_view_qa.html").status_code, 200)

    def test_rgbd_template_runs_for_real_by_default(self) -> None:
        self.assertFalse(TEMPLATES["rgbd_optimized"]["dry_run"])

    def test_rgbd_runner_can_train_gsplat_with_configured_script(self) -> None:
        try:
            from PIL import Image
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("Pillow/numpy is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            (source / "depth_selected").mkdir()
            for index in range(2):
                image = Image.new("RGB", (64, 48), (120 + index, 40, 80))
                image.save(source / "images" / f"frame_{index:05d}.png")
                depth = np.full((12, 16), 1000, dtype=np.uint16)
                Image.fromarray(depth).save(source / "depth_selected" / f"frame_{index:05d}.png")
            fake_train = root / "fake_train.py"
            fake_train.write_text(
                "import argparse,json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--output-ply'); p.add_argument('--metrics-output'); p.add_argument('--preview-output'); p.add_argument('--multi-preview-output'); p.add_argument('--pose-init-dir'); p.add_argument('--output-dir')\n"
                "args,_=p.parse_known_args()\n"
                "open(args.output_ply,'w').write('ply\\nformat ascii 1.0\\nelement vertex 1\\nproperty float x\\nproperty float y\\nproperty float z\\nproperty float f_dc_0\\nproperty float f_dc_1\\nproperty float f_dc_2\\nproperty float opacity\\nproperty float scale_0\\nproperty float scale_1\\nproperty float scale_2\\nproperty float rot_0\\nproperty float rot_1\\nproperty float rot_2\\nproperty float rot_3\\nend_header\\n0 0 0 0 0 0 1 0 0 0 1 0 0 0\\n')\n"
                "open(args.metrics_output,'w').write(json.dumps({'trained_gaussian_count':1,'max_steps':3,'final_preview_psnr':12.3}))\n",
                encoding="utf-8",
            )
            manifest = run_rgbd_optimized(
                RGBDOptimizeConfig(
                    source=source,
                    output=root / "out",
                    max_point_count=200,
                    train_gsplat=True,
                    gsplat_python=Path(sys.executable),
                    gsplat_train_script=fake_train,
                    gsplat_max_steps=3,
                    gsplat_max_frames=1,
                    gsplat_max_points=100,
                    gsplat_target_gaussians=10,
                )
            )
            self.assertFalse(manifest["dry_run"])
            viewer_manifest = json.loads((root / "out" / "viewer" / "viewer_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(viewer_manifest["layers"]["spark_3dgs"]["available"])
            self.assertEqual(viewer_manifest["metrics"]["gsplat_trained_gaussian_count"], 1)

    def test_fastapi_job_control_and_download(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from images23dgs.product.server import create_app
        except ModuleNotFoundError:
            self.skipTest("FastAPI is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "images"
            source.mkdir()
            (source / "a.jpg").write_bytes(b"x")
            config = ProductConfig(
                app_root=root,
                workspace_dir=root / "workspace",
                discoverse_root=root / "discoverse",
                real2sim_root=root / "real2sim",
                colmap_binary=root / "colmap",
            )
            (config.discoverse_root / "discoverse").mkdir(parents=True)
            (config.real2sim_root / "real2sim").mkdir(parents=True)
            (config.real2sim_root / "real2sim" / "cli.py").write_text("", encoding="utf-8")
            config.colmap_binary.write_text("#!/usr/bin/env bash\necho colmap\n", encoding="ascii")
            config.colmap_binary.chmod(0o755)
            with TestClient(create_app(config)) as client:
                dataset = client.post("/api/datasets/import-path", json={"path": str(source)}).json()
                queued = client.post("/api/jobs", json={"dataset_id": dataset["id"], "template": "quick_preview"}).json()
                canceled = client.post(f"/api/jobs/{queued['id']}/cancel").json()
                self.assertEqual(canceled["status"], "canceled")

                retry = client.post(f"/api/jobs/{queued['id']}/retry").json()
                for _ in range(20):
                    loaded = client.get(f"/api/jobs/{retry['id']}").json()
                    if loaded["status"] == "succeeded":
                        break
                    time.sleep(0.1)
                self.assertEqual(client.get(f"/api/jobs/{retry['id']}").json()["status"], "succeeded")
                artifacts = client.get(f"/api/jobs/{retry['id']}/artifacts").json()
                self.assertTrue(any(item["url"].endswith("/download") for item in artifacts))
                download = client.get(f"/api/jobs/{retry['id']}/download")
                self.assertEqual(download.status_code, 200)
                self.assertGreater(len(download.content), 100)


if __name__ == "__main__":
    unittest.main()
