# FUK Bridge — Blender addon

Drive the FUK generation server from inside Blender. Render a shot's structural
control passes in Blender and round-trip them through FUK to get a generated
still back — without leaving the viewport.

**Status:** images first. Video is a later milestone.

## How it works

```
Blender scene ─► beauty.png (write_still)
              └► depth / normals / openpose  (native render passes → OpenImageIO → PNG)
                                   │
                                   ▼
        POST /api/generate/image  (model = qwen_image_control_union_2512,
                                    control_image_paths = [<control map>])
                                   │
                                   ▼
        result.png  ◄── downloaded and loaded into Blender's Image Editor
```

- **Control passes are Blender-native.** Depth comes from the exact Z pass, normals
  from the Normal pass, and OpenPose from a rig view layer you set up. **Canny** is
  the one map FUK derives from the beauty render. (If OpenPose has no rig layer, FUK
  estimates it from the beauty render as a fallback.)
- **The shot `.json` is the source of truth.** *Load Shot* mirrors `tabs.image`
  (prompt, seed, model, steps, guidance…) into the panel; *Save to Shot* writes it
  back. Blender-only fields are stored under `tabs.image.blender_*`.
- Blender and FUK run on the **same machine**, so renders are exchanged by path —
  no uploads. Only Blender's bundled Python is used (`urllib`, `OpenImageIO`, `numpy`).

## Install

1. Start the FUK server: `python fuk/start_web_ui.py` (listens on `http://localhost:8000`).
2. In Blender: *Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk…* and pick a zip of the
   `fuk_blender/` folder (or point your scripts path at `fuk/plugins/blender`). Enable
   **FUK Bridge**.
3. Set the **Server URL** in the addon preferences if it isn't the default.

## Use

1. Open the **FUK** tab in the 3D viewport sidebar (press `N`).
2. Pick your **project folder** (the one holding the shot `.json` files) and press
   **Connect**, then choose a shot and **Load Shot**.
3. Edit the prompt / seed / settings, choose a **Control** (depth, normals, openpose,
   canny), frame your camera, and press **Quick Preview** (fast, low-res) or
   **Render Full**.
4. The result loads into an Image Editor; *Save to Shot* persists your settings.
   Press **Esc** during a run to cancel.

## Requirements / notes

- Generation with a control map needs the **control-union model weights**
  (`qwen_image_control_union_2512` or `qwen_image_control_union`) present on the
  server — see `config/models.json` / `download_models.sh`. Without them, pick a
  plain `qwen_image*` model (control map is ignored).
- Targets **Blender 4.2+** and is validated on **5.1**. The compositor is rebuilt and
  torn down per render, and all scene/render/compositor settings are restored
  afterward, so your scene is left untouched.
- **OpenPose rig layer:** create a view layer that renders only your OpenPose
  skeleton rig, then select it under *Control ▸ OpenPose ▸ Rig Layer*. Leave it blank
  to use FUK's estimator instead.
- Normal-map encoding (`n*0.5+0.5`) is a first pass; normal space may need tuning per
  scene.
