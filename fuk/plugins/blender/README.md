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
4. The result is shown per the **Result** mode (see below); *Save to Shot* persists
   your settings. Press **Esc** during a run to cancel.

### History entries (FUK side)

- **Quick Preview is ephemeral** — it comes back to Blender but does *not* leave a
  permanent FUK history entry (no clutter while iterating).
- **Render Full auto-saves** a complete, native-looking entry: `generated.png` +
  the control map (`control.png`) + the Blender beauty as `source.png` + metadata.
  The control map is also registered as a preprocess entry, so it shows in FUK's
  control/history panel and is draggable into the control input.
- **Save to History** promotes the *current* result (e.g. a preview you liked) into
  that same complete entry on demand. The button reads **In History** once saved.
- The FUK web history **auto-refreshes** when Blender saves (no manual refresh).
- The latest beauty/control/result + a `meta.json` live in
  `<project>/cache/_blender_io/<shot>/` as the working set.

### Live mode (auto-update)

Toggle **Live** for an IPR-style loop: after you move the camera or edit objects and
the scene settles for **Delay** seconds, it auto-runs a Quick Preview and updates the
viewport — like Cycles/Redshift interactive rendering (gated by generation time, not
real-time). It reacts only to real **transform/geometry** edits — selection clicks are
ignored — and **prompt/seed/setting edits stay manual**. Edits made *while* a preview
is rendering aren't lost: they coalesce into a single **trailing** render of the latest
state once the current one finishes (no growing backlog). Live previews are ephemeral
(no history entries).

Shortcuts (3D View): **Ctrl+Shift+P** runs a Quick Preview, **Ctrl+Shift+L** toggles
Live. (Rebind in Blender's Keymap editor under *3D View*.)

### `#tags` (prompt expansion)

The prompt supports FUK's `#markers`, expanded server-side at generation time — the
same vocabulary the FUK web UI uses.

- **Insert #tag** opens a searchable list of the available markers (pulled on Connect,
  or via the refresh button) and drops the chosen `#marker` into the prompt.
- **Preview** resolves the prompt exactly as generation will (expanding `#markers` and
  appending the storyboard mood) so you can see the final string; unknown markers are
  flagged in a warning.

### Result display

Pick how the generated image comes back under **Result**:

- **Viewport** *(default)* — set as the active camera's background overlay (shown in
  front) and switch the 3D view to camera view, so the result lands registered to your
  framing. Tune **Overlay Opacity** to flip between the render and your scene. Falls
  back to an Image Editor if the scene has no camera.
- **New Window** — open the result in a standalone Image Editor window.
- **Image Editor** — reuse an open Image Editor (or a new window if none).

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
