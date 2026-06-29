from pathlib import Path

old = r'''    function parseColmapImageCenters(text) {
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
'''

new = r'''    function parseColmapCameras(text) {
      const cameras = new Map();
      for (const raw of text.split(/\r?\n/)) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        const parts = line.split(/\s+/);
        if (parts.length < 5) continue;
        const id = Number(parts[0]);
        const model = parts[1];
        const width = Number(parts[2]);
        const height = Number(parts[3]);
        const params = parts.slice(4).map(Number);
        if (!Number.isFinite(id) || !Number.isFinite(width) || !Number.isFinite(height)) continue;
        let fx = params[0], fy = params[1], cx = params[2], cy = params[3];
        if (model === "SIMPLE_PINHOLE" || model === "SIMPLE_RADIAL" || model === "RADIAL") {
          fx = fy = params[0];
          cx = params[1];
          cy = params[2];
        }
        cameras.set(id, { id, model, width, height, fx, fy, cx, cy });
      }
      return cameras;
    }

    function cameraWorldPointFromColmap(qw, qx, qy, qz, tx, ty, tz, x, y, z) {
      const r = quatToMatrix(qw, qx, qy, qz);
      const center = cameraCenterFromColmap(qw, qx, qy, qz, tx, ty, tz);
      return new THREE.Vector3(
        center.x + r[0][0] * x + r[1][0] * y + r[2][0] * z,
        center.y + r[0][1] * x + r[1][1] * y + r[2][1] * z,
        center.z + r[0][2] * x + r[1][2] * y + r[2][2] * z
      );
    }

    function parseColmapImages(text) {
      const lines = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => line && !line.startsWith("#"));
      const images = [];
      for (let i = 0; i < lines.length; i += 2) {
        const parts = lines[i].split(/\s+/);
        if (parts.length < 10) continue;
        const imageId = Number(parts[0]);
        const nums = parts.slice(1, 8).map(Number);
        const cameraId = Number(parts[8]);
        const name = parts.slice(9).join(" ");
        if (!Number.isFinite(imageId) || nums.some((value) => !Number.isFinite(value)) || !Number.isFinite(cameraId)) continue;
        const [qw, qx, qy, qz, tx, ty, tz] = nums;
        images.push({ imageId, qw, qx, qy, qz, tx, ty, tz, cameraId, name, center: transformColmapPoint(cameraCenterFromColmap(...nums)) });
      }
      return images;
    }

    function addColmapCameraFrustums(group, images, cameras, box, layer) {
      const frustumPositions = [];
      const centerPositions = [];
      const textureLoader = new THREE.TextureLoader();
      const depth = Number(layer.camera_frustum_depth || 0.42);
      const thumbMax = Number(layer.image_thumbnail_max || 0);
      const thumbStep = thumbMax > 0 ? Math.max(1, Math.ceil(images.length / thumbMax)) : Infinity;
      const thumbBase = layer.image_thumbnail_base || null;
      const thumbExt = layer.image_thumbnail_ext || "";

      function pushSegment(a, b) {
        frustumPositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }

      images.forEach((image, index) => {
        const camera = cameras.get(image.cameraId) || cameras.values().next().value || { width: 1, height: 1, fx: 1, fy: 1, cx: 0.5, cy: 0.5 };
        const cornersCam = [
          [(0 - camera.cx) / camera.fx * depth, (0 - camera.cy) / camera.fy * depth, depth],
          [(camera.width - camera.cx) / camera.fx * depth, (0 - camera.cy) / camera.fy * depth, depth],
          [(camera.width - camera.cx) / camera.fx * depth, (camera.height - camera.cy) / camera.fy * depth, depth],
          [(0 - camera.cx) / camera.fx * depth, (camera.height - camera.cy) / camera.fy * depth, depth]
        ];
        const center = image.center;
        const corners = cornersCam.map(([x, y, z]) => transformColmapPoint(cameraWorldPointFromColmap(image.qw, image.qx, image.qy, image.qz, image.tx, image.ty, image.tz, x, y, z)));
        centerPositions.push(center.x, center.y, center.z);
        for (const corner of corners) {
          pushSegment(center, corner);
          box.expandByPoint(corner);
        }
        pushSegment(corners[0], corners[1]);
        pushSegment(corners[1], corners[2]);
        pushSegment(corners[2], corners[3]);
        pushSegment(corners[3], corners[0]);
        box.expandByPoint(center);

        if (thumbBase && index % thumbStep === 0) {
          const thumbName = image.name.replace(/\.[^.]+$/, thumbExt);
          const textureUrl = assetUrl(`${thumbBase}/${thumbName}`);
          const geometry = new THREE.BufferGeometry();
          const verts = new Float32Array([
            corners[0].x, corners[0].y, corners[0].z,
            corners[1].x, corners[1].y, corners[1].z,
            corners[2].x, corners[2].y, corners[2].z,
            corners[3].x, corners[3].y, corners[3].z
          ]);
          geometry.setAttribute("position", new THREE.BufferAttribute(verts, 3));
          geometry.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]), 2));
          geometry.setIndex([0, 1, 2, 0, 2, 3]);
          const material = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.78 });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.name = `colmap_image_${image.imageId}`;
          group.add(mesh);
          textureLoader.load(cacheBust(textureUrl), (texture) => {
            texture.colorSpace = THREE.SRGBColorSpace;
            material.map = texture;
            material.needsUpdate = true;
          }, undefined, () => {
            material.color.set(0x66c6a8);
            material.opacity = 0.22;
          });
        }
      });

      if (frustumPositions.length) {
        const frustumGeometry = new THREE.BufferGeometry();
        frustumGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(frustumPositions), 3));
        group.add(new THREE.LineSegments(frustumGeometry, new THREE.LineBasicMaterial({ color: 0x66c6a8, transparent: true, opacity: 0.88 })));
      }
      if (centerPositions.length) {
        const centerGeometry = new THREE.BufferGeometry();
        centerGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(centerPositions), 3));
        group.add(new THREE.Points(centerGeometry, new THREE.PointsMaterial({ color: 0x66c6a8, size: Number(pointSizeEl.value) * 3.0, transparent: true, opacity: 0.95 })));
      }
    }
'''

old_load = r'''      if (layer.images_txt) {
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
'''

new_load = r'''      if (layer.images_txt) {
        const response = await fetch(cacheBust(assetUrl(layer.images_txt)), { cache: "no-store" });
        if (response.ok) {
          let cameras = new Map();
          if (layer.cameras_txt) {
            const cameraResponse = await fetch(cacheBust(assetUrl(layer.cameras_txt)), { cache: "no-store" });
            if (cameraResponse.ok) cameras = parseColmapCameras(await cameraResponse.text());
          }
          const images = parseColmapImages(await response.text());
          addColmapCameraFrustums(group, images, cameras, box, layer);
          state.counts.cameras = images.length;
        }
      }
'''

paths = [
    Path("/tmp/images23dgs_wuying_20260626/images23dgs_src/images23dgs/viewer.py"),
    Path("/tmp/images23dgs_wuying_20260626/preview_filtered120_1600/viewer/index.html"),
    Path("/tmp/images23dgs_wuying_20260626/preview_traj240_anchor4_half/viewer/index.html"),
]

for path in paths:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"missing parse block in {path}")
    if old_load not in text:
        raise SystemExit(f"missing load block in {path}")
    text = text.replace(old, new).replace(old_load, new_load)
    path.write_text(text)
    print(f"patched {path}")
