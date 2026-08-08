"""
TRELLIS worker — runs INSIDE the isolated TRELLIS environment.

This file is never imported by FUK. The main venv launches it as a subprocess
using the interpreter at vendor/TRELLIS_ENV/env/bin/python, because TRELLIS
needs torch 2.6/cu126 plus CUDA extensions that would conflict with the main
venv's torch 2.9/cu130 (see install_trellis_env.sh).

Protocol
    argv[1]   path to a JSON job file
    stdout    human-readable progress, streamed to FUK's console panel, plus
              one machine-readable line:  __TRELLIS_RESULT__ {json}
    exit 0    success

Job file keys: image, output_dir, weights, export_formats, seed, steps,
cfg_strength, simplify, texture_size.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Must be set before trellis imports — both attention modules read this at
# import time. TRELLIS's sparse attention accepts only 'xformers' or
# 'flash_attn'; 'sdpa' silently leaves it on flash_attn, which then fails at
# the first forward pass if flash-attn was never built.
os.environ.setdefault("SPCONV_ALGO", "native")
if "ATTN_BACKEND" not in os.environ:
    try:
        import flash_attn  # noqa: F401
        os.environ["ATTN_BACKEND"] = "flash_attn"
    except ImportError:
        os.environ["ATTN_BACKEND"] = "xformers"

RESULT_MARKER = "__TRELLIS_RESULT__"


def emit(message):
    print(f"[TRELLIS] {message}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("usage: trellis_worker.py <job.json>", file=sys.stderr)
        return 2

    job = json.loads(Path(sys.argv[1]).read_text())

    trellis_src = job.get("trellis_src")
    if trellis_src and trellis_src not in sys.path:
        sys.path.insert(0, trellis_src)

    image_path = Path(job["image"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    export_formats = [f.lower() for f in job.get("export_formats", ["glb"])]

    emit(f"attention backend: {os.environ['ATTN_BACKEND']}")

    from PIL import Image
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    emit(f"loading weights: {job['weights']}")
    t0 = time.perf_counter()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(job["weights"])
    pipeline.cuda()
    emit(f"pipeline loaded in {time.perf_counter() - t0:.1f}s")

    image = Image.open(image_path).convert("RGBA")
    emit(f"input: {image_path.name} ({image.width}x{image.height})")

    # 'radiance_field' is skipped deliberately: its decoder needs
    # diffoctreerast, which the mesh path does not, and Phase 1 ships mesh
    # output only. 'gaussian' is still required — it carries the appearance
    # that gets baked into the GLB's texture.
    formats = ["mesh", "gaussian"]

    emit("running SLAT generation…")
    t0 = time.perf_counter()
    outputs = pipeline.run(
        image,
        seed=int(job.get("seed", 42)),
        formats=formats,
        sparse_structure_sampler_params={
            "steps": int(job.get("steps", 12)),
            "cfg_strength": float(job.get("cfg_strength", 7.5)),
        },
        slat_sampler_params={
            "steps": int(job.get("steps", 12)),
            "cfg_strength": float(job.get("slat_cfg_strength", 3.0)),
        },
    )
    inference_seconds = time.perf_counter() - t0
    emit(f"generation complete in {inference_seconds:.1f}s")

    result = {
        "outputs": {},
        "inference_seconds": round(inference_seconds, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }

    if "glb" in export_formats or "obj" in export_formats:
        # The raw SLAT mesh drives everything that follows, and its size is
        # what makes this stage take 20s or 200s. Report it before starting so
        # a slow run is explainable rather than mysterious.
        raw_mesh = outputs["mesh"][0]
        raw_verts = int(len(raw_mesh.vertices))
        raw_faces = int(len(raw_mesh.faces))
        fill_holes = bool(job.get("fill_holes", True))
        emit(f"raw mesh: {raw_verts:,} vertices, {raw_faces:,} faces")
        if raw_verts > 150_000:
            emit(f"WARNING: large mesh ({raw_verts:,} verts) — postprocessing "
                 f"may take several minutes")
        emit(f"postprocessing mesh (simplify={job.get('simplify', 0.95)}, "
             f"fill_holes={fill_holes}, texture={job.get('texture_size', 1024)}px)…")

        t0 = time.perf_counter()
        # verbose=True on purpose: to_glb's internal tqdm is the only progress
        # signal during a stage that can run for minutes, and the host relays
        # it straight into the console panel. Silence here is what made this
        # look like a hang.
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            raw_mesh,
            simplify=float(job.get("simplify", 0.95)),
            fill_holes=fill_holes,
            texture_size=int(job.get("texture_size", 1024)),
            verbose=True,
        )
        if "glb" in export_formats:
            path = output_dir / "mesh.glb"
            glb.export(str(path))
            result["outputs"]["glb"] = str(path)
            emit(f"wrote {path.name}")
        if "obj" in export_formats:
            path = output_dir / "mesh.obj"
            glb.export(str(path))
            result["outputs"]["obj"] = str(path)
            emit(f"wrote {path.name}")
        result["vertex_count"] = int(len(glb.vertices))
        result["face_count"] = int(len(glb.faces))
        emit(f"mesh postprocess took {time.perf_counter() - t0:.1f}s")

    if "ply" in export_formats:
        # TRELLIS's PLY is the Gaussian splat representation, not a plain
        # point cloud — the splat viewer is a Phase 3 item, but the file is
        # still the most faithful point export the model can produce.
        path = output_dir / "pointcloud.ply"
        # Orientation fix. Both TRELLIS export paths claim to rotate z-up to
        # y-up with the same matrix, but apply it transposed relative to one
        # another: to_glb does `vertices @ M`, while save_ply does
        # `xyz @ M.T` (and `M @ rotation`). M is orthogonal, so M.T is its
        # inverse — the splat lands 180° about X from the GLB, i.e. upside
        # down. Passing M.T here cancels save_ply's own transpose, so the
        # PLY comes out in the same frame as the mesh.
        z_up_to_y_up_T = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        outputs["gaussian"][0].save_ply(str(path), transform=z_up_to_y_up_T)
        result["outputs"]["ply"] = str(path)
        emit(f"wrote {path.name} (gaussian splat)")

    result["has_mesh"] = "glb" in result["outputs"] or "obj" in result["outputs"]
    print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
