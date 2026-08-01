"""
Geometry export helpers shared by the 3D reconstruction backends.

VGGT hands back a dense per-pixel point cloud; TRELLIS hands back a mesh
directly. Both funnel through here so the export formats behave identically
whichever tier produced the geometry.

Formats:
    GLB   primary deliverable — mesh with vertex colours, or a point cloud
    PLY   point cloud, for tools that want the raw solve
    OBJ   Phase 3 / Nuke-Natron path, written only when asked for
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _log(message: str, level: str = "info"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    colors = {
        'info': '\033[96m', 'success': '\033[92m',
        'warning': '\033[93m', 'error': '\033[91m', 'end': '\033[0m',
    }
    symbols = {'info': '', 'success': '✔ ', 'warning': '⚠ ', 'error': '✗ '}
    print(
        f"{colors.get(level, colors['info'])}[{timestamp}] "
        f"{symbols.get(level, '')}[THREED] {message}{colors['end']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Background rejection
# ---------------------------------------------------------------------------

def neutral_background_mask(
    colors: np.ndarray,
    luma_min: float = 0.62,
    sat_max: float = 0.10,
) -> np.ndarray:
    """Foreground mask for subjects shot against a plain studio backdrop.

    VGGT reconstructs a *scene*, not an object — it assigns depth to every
    pixel, background included. A plain white or grey backdrop is textureless,
    so those depths are unconstrained and the network invents a curved sheet
    behind the subject. Across generated views those invented sheets disagree
    with each other, and what lands in the point cloud is a shell of
    contradictory geometry wrapped around the thing you actually wanted.
    Poisson then dutifully closes a surface over the shell.

    Measured on a 64-view Dataset Builder orbit: 69% of surviving points were
    backdrop. Confidence filtering does not remove them — VGGT is *confident*
    about a smooth backdrop, which is precisely the problem.

    Bright plus near-neutral is the signature of a studio backdrop; a real
    subject that is both very bright and fully desaturated (white ceramic,
    chrome) will be partly cut too, which is why the thresholds are exposed.

    Returns a boolean array, True where the pixel is foreground.
    """
    flat = colors.reshape(-1, 3)
    luma = flat.mean(axis=1)
    saturation = flat.max(axis=1) - flat.min(axis=1)
    return ~((luma > luma_min) & (saturation < sat_max))


def alpha_masks_for(image_paths, reference_shape) -> Optional[np.ndarray]:
    """Foreground masks taken from the source images' alpha channels.

    VGGT itself throws alpha away — its loader composites RGBA onto white
    before inference — so alpha cannot help the solve. It is still the most
    reliable matte available, so it is applied afterwards instead, against the
    predicted per-pixel geometry.

    The masks are pushed through VGGT's own preprocessing rather than a
    reimplementation of it: each alpha channel is written out as an opaque
    black-and-white image and run through `load_and_preprocess_images`, so the
    resize, crop and pad geometry is identical by construction and the mask
    lines up with `world_points` exactly.

    Returns None when no input carries usable alpha.
    """
    import tempfile

    from PIL import Image
    from vggt.utils.load_fn import load_and_preprocess_images

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths, found_alpha = [], False
        for index, path in enumerate(image_paths):
            image = Image.open(path)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                alpha = image.convert("RGBA").split()[-1]
                if np.asarray(alpha).min() < 250:
                    found_alpha = True
            else:
                alpha = Image.new("L", image.size, 255)
            out = Path(tmpdir) / f"mask_{index:05d}.png"
            # Written as RGB, not LA — an alpha channel here would be
            # composited onto white by the loader and defeat the purpose.
            alpha.convert("RGB").save(out)
            tmp_paths.append(str(out))

        if not found_alpha:
            return None

        masks = load_and_preprocess_images(tmp_paths)

    if masks.shape[-2:] != tuple(reference_shape):
        _log(
            f"Alpha masks came back {tuple(masks.shape[-2:])} but geometry is "
            f"{tuple(reference_shape)} — ignoring alpha",
            "warning",
        )
        return None

    # Back to a per-pixel boolean in the same layout as world_points.
    return (masks.mean(dim=1) > 0.5).numpy()


# ---------------------------------------------------------------------------
# Point cloud conditioning
# ---------------------------------------------------------------------------

def filter_by_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    percentile: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the most confident points.

    A percentile is used rather than an absolute threshold because VGGT's
    confidence scale is not calibrated across scenes — synthetic Qwen views in
    particular come back with a narrow, low band of values, and any fixed cut
    either keeps everything or nothing.
    """
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    conf = conf.reshape(-1)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    points, colors, conf = points[finite], colors[finite], conf[finite]

    if points.size == 0:
        return points, colors

    percentile = float(np.clip(percentile, 0.0, 99.0))
    if percentile > 0:
        threshold = np.percentile(conf, percentile)
        mask = conf >= threshold
        # A degenerate confidence distribution (all values identical) makes the
        # mask drop everything — keep the cloud rather than return nothing.
        if mask.sum() >= 100:
            points, colors = points[mask], colors[mask]

    return points, colors


def trim_outliers(points: np.ndarray, colors: np.ndarray, std_ratio: float = 3.0):
    """Drop points far outside the bulk of the cloud.

    Depth predictions blow up near frame edges and on background sky, throwing
    a handful of points thousands of units away. Those alone would set the
    bounding box and make the mesh appear as a speck in the viewer.
    """
    if len(points) < 100:
        return points, colors

    centre = np.median(points, axis=0)
    distance = np.linalg.norm(points - centre, axis=1)
    cutoff = np.median(distance) + std_ratio * (np.std(distance) + 1e-8)
    mask = distance <= cutoff
    if mask.sum() < 100:
        return points, colors
    return points[mask], colors[mask]


def normalize_to_unit_cube(points: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Centre the cloud on the origin and scale its longest axis to 1.

    VGGT solves geometry up to an arbitrary scale, so the raw numbers carry no
    real-world units. Normalising gives the viewer a predictable framing and
    keeps exported assets a sane size in Blender/Nuke. The transform is
    recorded in the metadata so the original solve can be recovered.
    """
    if len(points) == 0:
        return points, {"centre": [0.0, 0.0, 0.0], "scale": 1.0}

    centre = points.mean(axis=0)
    centred = points - centre
    extent = float(np.abs(centred).max())
    scale = 1.0 / extent if extent > 1e-8 else 1.0
    return centred * scale, {"centre": centre.tolist(), "scale": scale}


def voxel_downsample(points: np.ndarray, colors: np.ndarray, target: int = 500_000):
    """Thin a dense cloud to roughly `target` points on a voxel grid.

    A 24-image VGGT solve is ~6.4M points, which no browser will orbit
    smoothly and Poisson reconstruction would grind on. Voxel downsampling
    preserves surface shape far better than random sampling.
    """
    if len(points) <= target:
        return points, colors

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))

    # Voxel size that lands near the target count, assuming points spread over
    # a roughly 2D surface inside the bounding volume.
    extent = points.max(axis=0) - points.min(axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-6)))
    voxel = max((volume / max(target, 1)) ** (1 / 3), 1e-6)

    for _ in range(8):
        reduced = cloud.voxel_down_sample(voxel)
        if len(reduced.points) <= target:
            cloud = reduced
            break
        voxel *= 1.3
    else:
        cloud = reduced

    return (
        np.asarray(cloud.points, dtype=np.float32),
        np.asarray(cloud.colors, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Surface reconstruction
# ---------------------------------------------------------------------------

def pointcloud_to_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    depth: int = 9,
    density_quantile: float = 0.08,
    progress_callback=None,
):
    """Poisson-reconstruct a surface from an oriented point cloud.

    Returns a trimesh.Trimesh with vertex colours, or None if the cloud is too
    sparse or degenerate to reconstruct.
    """
    import open3d as o3d
    import trimesh

    if len(points) < 1000:
        _log(f"Only {len(points)} points — too sparse to mesh", "warning")
        return None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))

    if progress_callback:
        progress_callback(0.75, "Estimating normals")

    # Poisson needs oriented normals. The radius is tied to the cloud's own
    # scale so this works whether the solve came back unit-sized or not.
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    radius = max(extent * 0.01, 1e-5)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    cloud.orient_normals_consistent_tangent_plane(k=15)

    if progress_callback:
        progress_callback(0.82, f"Poisson reconstruction (depth {depth})")

    _log(f"Poisson reconstruction from {len(points):,} points (depth={depth})")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth, width=0, scale=1.1, linear_fit=False
        )

    if len(mesh.vertices) == 0:
        _log("Poisson reconstruction produced no geometry", "warning")
        return None

    # Poisson closes the surface over regions with no support, producing large
    # balloon-like sheets around the subject. Cutting the lowest-density
    # vertices removes them.
    if density_quantile > 0:
        densities = np.asarray(densities)
        cutoff = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < cutoff)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        _log("Mesh empty after cleanup", "warning")
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    vertex_colors = (
        (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        if len(mesh.vertex_colors) == len(vertices)
        else None
    )

    _log(f"Mesh: {len(vertices):,} vertices, {len(faces):,} faces", "success")
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> Path:
    """Write a binary PLY point cloud."""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)

    _log(f"PLY: {path.name} ({len(points):,} points)", "success")
    return path


def write_glb(mesh, path: Path) -> Path:
    """Write a mesh (or trimesh PointCloud) as GLB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="glb")
    _log(f"GLB: {path.name}", "success")
    return path


def write_obj(mesh, path: Path) -> Path:
    """Write a mesh as OBJ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="obj")
    _log(f"OBJ: {path.name}", "success")
    return path


def pointcloud_as_glb(points: np.ndarray, colors: np.ndarray, path: Path) -> Path:
    """Write a point cloud as GLB, for when meshing fails or isn't wanted."""
    import trimesh

    rgba = np.concatenate(
        [
            (np.clip(colors, 0, 1) * 255).astype(np.uint8),
            np.full((len(colors), 1), 255, dtype=np.uint8),
        ],
        axis=1,
    )
    cloud = trimesh.PointCloud(vertices=points.astype(np.float32), colors=rgba)
    return write_glb(cloud, path)
