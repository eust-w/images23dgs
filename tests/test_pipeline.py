from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from images23dgs.artifixer import should_use_artifixer, write_empty_artifixer_caption_files
from images23dgs.colmap import collect_images, detect_colmap_option_names, parse_colmap_registered_image_names
from images23dgs.pipeline import PipelineConfig, choose_artifixer_anchor_names, run_pipeline
from images23dgs.ply import read_ply_header
from images23dgs.discoverse import DiscoverseResult
from images23dgs.true_geometry import generate_depth_point_cloud
from images23dgs.viewer import write_layered_viewer


class PipelineTests(unittest.TestCase):
    def test_artifixer_decision_for_sparse_images(self) -> None:
        selected, reasons = should_use_artifixer(
            source_images=8,
            registered_ratio=0.9,
            min_images_for_direct=24,
            min_registered_ratio=0.65,
        )
        self.assertTrue(selected)
        self.assertIn("source_images<24", reasons)

    def test_artifixer_decision_for_bad_registration(self) -> None:
        selected, reasons = should_use_artifixer(
            source_images=40,
            registered_ratio=0.25,
            min_images_for_direct=24,
            min_registered_ratio=0.65,
        )
        self.assertTrue(selected)
        self.assertIn("registered_ratio<0.65", reasons)

    def test_artifixer_anchor_selection_leaves_targets(self) -> None:
        names = [f"frame_{index:03d}.jpg" for index in range(16)]
        anchors = choose_artifixer_anchor_names(names, anchor_count=None)
        self.assertIsNotNone(anchors)
        self.assertEqual(len(anchors or []), 8)
        self.assertLess(len(anchors or []), len(names))
        self.assertEqual(anchors[0], "frame_000.jpg")
        self.assertEqual(anchors[-1], "frame_015.jpg")

    def test_colmap_images_txt_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.txt"
            path.write_text(
                "# header\n"
                "1 1 0 0 0 0 0 0 1 frame 001.jpg\n"
                "10.0 12.0 1\n"
                "2 1 0 0 0 0 0 0 1 frame_002.jpg\n"
                "\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_colmap_registered_image_names(path), ["frame 001.jpg", "frame_002.jpg"])

    def test_ply_header_detects_3dgs_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.ply"
            fields = [
                "x",
                "y",
                "z",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
            path.write_text(
                "ply\nformat ascii 1.0\n"
                "element vertex 1\n"
                + "".join(f"property float {field}\n" for field in fields)
                + "end_header\n"
                + " ".join(["0"] * len(fields))
                + "\n",
                encoding="ascii",
            )
            header = read_ply_header(path)
            self.assertEqual(header.vertex_count, 1)
            self.assertTrue(header.has_3dgs_fields)

    def test_dry_run_writes_manifest_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            discoverse = root / "DISCOVERSE"
            real2sim = root / "gs_playground"
            out = root / "out"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"not-a-real-jpeg")
            (images / "b.jpg").write_bytes(b"not-a-real-jpeg")
            (discoverse / "discoverse").mkdir(parents=True)
            (real2sim / "real2sim").mkdir(parents=True)
            (real2sim / "real2sim" / "cli.py").write_text("", encoding="utf-8")

            manifest = run_pipeline(
                PipelineConfig(
                    images=images,
                    output=out,
                    prompt="test scene",
                    discoverse_root=discoverse,
                    real2sim_root=real2sim,
                    dry_run=True,
                    skip_artifixer=True,
                )
            )

            self.assertTrue((out / "reports" / "run_manifest.json").is_file())
            self.assertTrue((out / "run_commands.sh").is_file())
            self.assertTrue((out / "viewer" / "index.html").is_file())
            self.assertTrue((out / "viewer" / "viewer_manifest.json").is_file())
            self.assertEqual(manifest["input"]["materialized_images"], 2)
            self.assertFalse(manifest["artifixer"]["selected"])
            commands = json.loads((out / "reports" / "run_manifest.json").read_text())["commands"]
            self.assertTrue(any(record["stage"] == "discoverse.real2sim.generate_scene" for record in commands))
            feature_command = next(record["command"] for record in commands if record["stage"] == "colmap.feature_extractor")
            self.assertIn("--ImageReader.single_camera", feature_command)
            self.assertEqual(feature_command[feature_command.index("--ImageReader.single_camera") + 1], "1")

    def test_artifixer_prepare_phases_are_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            discoverse = root / "DISCOVERSE"
            real2sim = root / "gs_playground"
            out = root / "out"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"not-a-real-jpeg")
            (images / "b.jpg").write_bytes(b"not-a-real-jpeg")
            (discoverse / "discoverse").mkdir(parents=True)
            (real2sim / "real2sim").mkdir(parents=True)
            (real2sim / "real2sim" / "cli.py").write_text("", encoding="utf-8")

            run_pipeline(
                PipelineConfig(
                    images=images,
                    output=out,
                    prompt="test scene",
                    discoverse_root=discoverse,
                    real2sim_root=real2sim,
                    dry_run=True,
                    force_artifixer=True,
                    artifixer_checkpoint=root / "artifixer-14b.pt",
                    artifixer_prepare_phases="prepare,reconstruct,render,scale",
                )
            )

            commands = json.loads((out / "reports" / "run_manifest.json").read_text())["commands"]
            prepare_command = next(record["command"] for record in commands if record["stage"] == "artifixer.prepare_colmap")
            self.assertIn("--phases", prepare_command)
            self.assertEqual(
                prepare_command[prepare_command.index("--phases") + 1],
                "prepare,reconstruct,render,scale",
            )
            self.assertTrue(any(record["stage"] == "artifixer.write_empty_caption" for record in commands))

    def test_empty_artifixer_caption_file_matches_split(self) -> None:
        try:
            import h5py
        except ModuleNotFoundError:
            self.skipTest("h5py is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prep = root / "prep"
            prep.mkdir()
            (prep / "split.json").write_text(
                json.dumps({"test": {"scene": {"prompt_path": "captions/scene/caption.h5"}}}),
                encoding="utf-8",
            )

            written = write_empty_artifixer_caption_files(prep)

            self.assertEqual(written, [prep / "captions" / "scene" / "caption.h5"])

            with h5py.File(written[0], "r") as handle:
                dataset = handle["empty_prompt"]
                self.assertEqual(dataset.shape, (1, 4096))
                self.assertEqual(dataset.attrs["caption"], "")

    def test_collect_images_accepts_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.png"
            path.write_bytes(b"x")
            self.assertEqual(collect_images(path), [path])

    def test_collect_images_ignores_macos_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "imagefew"
            sidecars = root / "__MACOSX" / "imagefew"
            images.mkdir()
            sidecars.mkdir(parents=True)
            real = images / "DSC05572.jpg"
            real.write_bytes(b"real")
            (sidecars / "._DSC05572.jpg").write_bytes(b"sidecar")

            self.assertEqual(collect_images(root), [real])

    def test_detect_colmap_sift_option_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "colmap"
            binary.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"$1\" = feature_extractor ]; then\n"
                "  echo '  --SiftExtraction.use_gpu arg (=1)'\n"
                "else\n"
                "  echo '  --SiftMatching.use_gpu arg (=1)'\n"
                "fi\n",
                encoding="ascii",
            )
            binary.chmod(0o755)

            names = detect_colmap_option_names(str(binary), matcher="sequential")
            self.assertEqual(names.feature_prefix, "SiftExtraction")
            self.assertEqual(names.matching_prefix, "SiftMatching")

    def test_viewer_converts_ascii_3dgs_for_spark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "discoverse_package"
            ply = package / "3dgs" / "scene.ply"
            mesh = package / "meshes" / "scene.obj"
            mjcf = package / "mjcf" / "scene.xml"
            ply.parent.mkdir(parents=True)
            mesh.parent.mkdir(parents=True)
            mjcf.parent.mkdir(parents=True)
            fields = [
                "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                *[f"f_rest_{idx}" for idx in range(45)],
                "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            ]
            ply.write_text(
                "ply\nformat ascii 1.0\n"
                "element vertex 1\n"
                + "".join(f"property float {field}\n" for field in fields)
                + "end_header\n"
                + " ".join(["0"] * len(fields))
                + "\n",
                encoding="ascii",
            )
            mesh.write_text("o scene\nv 0 0 0\n", encoding="ascii")
            mjcf.write_text("<mujoco />\n", encoding="ascii")
            (package / "package_manifest.json").write_text(
                json.dumps({"scene": {"mesh_obj": "meshes/scene.obj"}}),
                encoding="utf-8",
            )

            result = write_layered_viewer(
                output_root=root,
                discoverse_result=DiscoverseResult(package, ply, read_ply_header(ply), "ok"),
                final_scene=None,
                scene_name="test",
            )
            manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
            spark_file = root / "viewer" / manifest["layers"]["spark_3dgs"]["file"]
            header = read_ply_header(spark_file)
            self.assertEqual(header.format, "binary_little_endian")
            self.assertTrue(header.has_3dgs_fields)

    def test_true_geometry_writes_rgb_depth_point_cloud(self) -> None:
        try:
            import numpy as np
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("numpy and Pillow are required for depth point-cloud generation")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "prep" / "imagefew"
            ns = scene / "3dgrut_input" / "imagefew" / "nerfstudio"
            depth = scene / "recon_results" / "imagefew" / "reconstruction" / "imagefew" / "ours_500" / "depth"
            renders = depth.parent / "renders"
            opacity = depth.parent / "opacity"
            ns.mkdir(parents=True)
            depth.mkdir(parents=True)
            renders.mkdir()
            opacity.mkdir()
            (ns / "transforms.json").write_text(
                json.dumps(
                    {
                        "w": 2,
                        "h": 2,
                        "fl_x": 2.0,
                        "fl_y": 2.0,
                        "cx": 0.5,
                        "cy": 0.5,
                        "frames": [
                            {
                                "file_path": "00000.png",
                                "transform_matrix": [
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            np.save(depth / "00000.npy", np.ones((2, 2), dtype=np.float32))
            Image.new("L", (2, 2), 255).save(opacity / "00000.png")
            Image.new("RGB", (2, 2), (10, 20, 30)).save(renders / "00000.png")

            result = generate_depth_point_cloud(
                prep_root=scene,
                output_ply=root / "points.ply",
                max_points=4,
            )

            header = read_ply_header(result.output_ply)
            self.assertEqual(result.points, 4)
            self.assertEqual(header.vertex_count, 4)
            self.assertEqual(header.format, "binary_little_endian")
            self.assertLess(result.bounds_min[2], 0)


if __name__ == "__main__":
    unittest.main()
