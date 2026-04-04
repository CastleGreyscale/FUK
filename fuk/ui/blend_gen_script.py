"""
Blender headless script — generates a configured .blend template file.

Called by the FUK server as:
    blender --background --python blend_gen_script.py -- <json_args>

JSON args:
    output_path  str   Full path to save the .blend file
    width        int   Render resolution X
    height       int   Render resolution Y
    frame_count  int   scene.frame_end (frame_start is always 1)
    fps          int   Render FPS (integer; 23.976/29.97 handled via fps_base)
"""

import bpy
import sys
import json
import os

# ── Parse args after '--' ──────────────────────────────────────────────────
argv = sys.argv
try:
    args_start = argv.index('--') + 1
    args = json.loads(argv[args_start])
except (ValueError, IndexError, json.JSONDecodeError) as e:
    print(f"[blend_gen] ERROR: Could not parse args — {e}", flush=True)
    sys.exit(1)

output_path  = args.get('output_path')
width        = int(args.get('width',        1920))
height       = int(args.get('height',       1080))
frame_count  = int(args.get('frame_count',  25))
fps_value    = args.get('fps', 24)          # May be float like 23.976

if not output_path:
    print("[blend_gen] ERROR: output_path is required", flush=True)
    sys.exit(1)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ── Configure scene ───────────────────────────────────────────────────────
scene = bpy.context.scene

# Resolution
scene.render.resolution_x          = width
scene.render.resolution_y          = height
scene.render.resolution_percentage = 100

# Frame range
scene.frame_start = 1
scene.frame_end   = frame_count
scene.frame_current = 1

# FPS — handle drop-frame rates via fps + fps_base
FPS_MAP = {
    23.976: (24, 1001/1000),
    29.97:  (30, 1001/1000),
    59.94:  (60, 1001/1000),
}
if fps_value in FPS_MAP:
    scene.render.fps,  scene.render.fps_base = FPS_MAP[fps_value]
else:
    scene.render.fps      = int(fps_value)
    scene.render.fps_base = 1.0

# Output format
scene.render.image_settings.file_format        = 'PNG'
scene.render.image_settings.color_mode         = 'RGBA'
scene.render.image_settings.color_depth        = '8'

# Use relative paths inside the blend
scene.render.filepath = '//'

print(f"[blend_gen] Saving {width}x{height} | {frame_count}f | {fps_value}fps → {output_path}", flush=True)

bpy.ops.wm.save_as_mainfile(filepath=output_path)

print("[blend_gen] Done", flush=True)
