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
  - **Depth + OpenPose** uses both in one scene: depth for the set, the skeleton for
    the figure. The control-union model reads a single control image, so the two
    are composited — the rig layer's skeleton keyed over the depth map — into
    `depth_openpose.png`. `depth.png` and `openpose.png` are written beside it so
    either half can be inspected. It needs a rig layer (there is no estimator
    fallback) and works for video too, composited per frame.
  - With a rig layer set, the collections that draw the skeleton are **excluded
    from the active view layer for the render** (and restored after), whichever
    control is selected. Otherwise the emission sticks are real geometry in the
    main layer and end up in the depth map and the beauty. Only collections the rig
    layer actually renders are touched — anything else that should stay out of the
    depth map (an older rig, viewport helpers) has to be excluded from the main
    view layer by hand.
  - Renders temporarily force **Sequencer off**. Blender renders Render Layers →
    Compositor → Sequencer, and *any* strip in the sequencer replaces the render
    output: the compositor is skipped (no control pass) and the beauty PNG becomes a
    frame of that strip. Since the video flow parks a `FUK_result` strip there, one
    video generation would otherwise silently feed every later still a stale beauty
    and no control. The original setting is restored afterwards.
- **Image and Video are separate modes.** The toggle under the shot header swaps the
  whole panel stack. Each mode has its **own prompt, negative and seed** — the video
  side has a *Copy from Image* button when you want them to match. Connection, shot
  binding and **Control** are shared; one *Load Shot* / *Save to Shot* covers both.
- **The shot `.json` is the source of truth.** *Load Shot* mirrors `tabs.image` and
  `tabs.video` (prompt, seed, model, steps, guidance…) into the panel; *Save to Shot*
  writes both back. Blender-only fields are stored under `tabs.<tab>.blender_*`.
  - Settings are read from and written to the slot of **the model Blender generates
    with** — `tabs.image.modelSettings[<control model>]` and
    `tabs.video.modelSettings["wan_vace_a14b"]` — not whatever the web UI has active,
    so the seed Blender sends is the seed recorded against that model. Neither tab's
    `activeModel` is ever rewritten. A model with no slot yet inherits the active
    model's prompt/seed on first Load.
  - Seeds are uint32 (0–4294967295), so the panel's seed is a **text field** —
    Blender's integer properties top out at 2147483647 and cannot hold half of them.
  - A full generation writes the seed it actually used (and the control source) back
    into the shot immediately. Prompt, steps and guidance still need *Save to Shot*.
  - **Control** and the OpenPose layer persist in the `.blend` and are only overridden
    by a Load when the shot actually carries a `blender_*` key.
- Blender and FUK run on the **same machine**, so renders are exchanged by path —
  no uploads. Only Blender's bundled Python is used (`urllib`, `OpenImageIO`, `numpy`).

## Install

1. Start the FUK server: `python fuk/start_web_ui.py` (listens on `http://localhost:8000`).
2. In Blender: *Edit ▸ Preferences ▸ Add-ons ▸ Install from Disk…* and pick a zip of the
   `fuk_blender/` folder (or point your scripts path at `fuk/plugins/blender`). Enable
   **FUK Bridge**.
3. Set the **Server URL** in the addon preferences if it isn't the default.

## Use

1. Open the **FUK** tab in the 3D viewport sidebar (press `N`). The shot binding, the
   **Image | Video** toggle and the run status sit on the **FUK** panel itself;
   *Prompt*, *Generation*, *Control*, *Result* and *Render* are collapsible sub-panels
   beneath it. Flipping to **Video** swaps in the video's own *Prompt*, *Generation*
   and *Render* panels — *Control* stays put, since it drives both.
2. Pick your **project folder** (the one holding the shot `.json` files) and press
   **Connect**, then choose a shot and **Load Shot**. The shot list is a snapshot
   taken at connect — press the ⟳ button beside the dropdown to re-scan the folder
   after creating shots in FUK (your current selection is kept).
3. Edit the prompt / seed / settings, choose a **Control** (depth, normals, openpose,
   depth + openpose, canny), frame your camera, and press **Quick Preview** (fast, low-res) or
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

**Interrupt** (the small button next to Delay, on by default): in Live mode a new edit
**cancels the in-flight preview and restarts** with the latest state instead of waiting
for it to finish — true IPR responsiveness. With it off, edits made mid-render coalesce
into one trailing render instead.

Press **Esc** (or the **Cancel** button that appears while generating) to stop a
generation for real — it aborts at the next diffusion step, not just stops waiting.

The addon **auto-connects** on launch / file open when the project folder is saved in
the .blend; the **Connect** button reads *Connected* once it's reached the server.

Shortcuts (3D View): **Ctrl+Shift+P** runs a Quick Preview, **Ctrl+Shift+L** toggles
Live. (Rebind in Blender's Keymap editor under *3D View*.)

### Video (Wan-VACE)

Animate a still with your scene's motion. Workflow:

1. Generate a **still** first (image flow) — it becomes the VACE **reference** (the look).
2. Set the scene **frame range** (the motion) and pick a **depth / normals / openpose**
   control.
3. Flip the header toggle to **Video** and write the video's **own prompt and seed**
   (*Copy from Image* if you want the still's). These live in `tabs.video`, so they are
   completely independent of the image tab's — changing one never disturbs the other.
4. Press **Generate Video**. The addon renders the control pass over the whole frame
   range into a folder (depth is normalized over a *global* range so it doesn't
   flicker) and runs `wan_vace_a14b` with `vace_video` = that sequence and
   `vace_reference_image` = your still.
5. The result follows the **Result** mode, same as stills: in *Viewport* the mp4 becomes
   the camera's background clip, so scrubbing the timeline plays the generated video
   over your scene and the opacity slider works on it exactly as it does for an image.
   The other modes load it into the **Video Sequencer**. The seed it used is written
   straight back into `tabs.video`, as with stills.

Notes: video is slow (minutes) and runs on demand — no live mode. Control must be a
native pass (depth/normals/openpose-rig); canny/estimated openpose aren't supported for
sequences. Requires the **wan_vace_a14b** weights on the server. Esc stops *waiting* but
video isn't abortable mid-render yet (unlike images).

**Frame count is trimmed to 4n+1.** Wan only runs at those lengths and rounds up
internally; a control sequence of any other length lands on a different temporal grid
than the latents, and VACE silently zero-pads it rather than erroring. Up to 3 trailing
frames are dropped so the control matches exactly. The server applies the same rule to
width/height (multiples of 16) — **50% of 720p is 360, which rounds to 368**, and that
mismatch used to shear the control across the frame until it read as noise.

### Writing prompts

Blender has no multi-line text field — a `StringProperty` is always one line, with no
wrapping — so the prompt UI works around it two ways:

- The **pencil** button beside either field opens a wide dialog holding both the
  prompt and the negative. Same single line, roughly three times the width. It edits
  the scene properties directly, so changes apply as you type and dismissing the
  dialog does **not** revert them (there's no cancel; it's a bigger window onto the
  same data, not a buffered editor).
- Below each field, anything too long to fit is echoed word-wrapped in a read-only
  box, so you can at least *read* the whole prompt in the sidebar.

### `#tags` (prompt expansion)

The prompt supports FUK's `#markers`, expanded server-side at generation time — the
same vocabulary the FUK web UI uses.

- **Insert #tag** opens a searchable list of the available markers (pulled on Connect,
  or via the refresh button) and drops the chosen `#marker` into the prompt.
- **Preview** resolves the prompt exactly as generation will (expanding `#markers` and
  appending the storyboard mood) so you can see the final string; unknown markers are
  flagged in a warning.
- **Framing variants.** A LoRA in `defaults_loras.json` may carry `inject_variants`
  (`close`, `medium`, `wide`) beside its `inject_text`. With **Describe Shot** on, the
  measured shot size is sent with the generation and each `#marker` expands to the
  wording that fits: face only and head-and-shoulders use `close`, chest-up to knees-up
  use `medium`, whole body and far use `wide`. `#marker:wide` in the prompt overrides
  the measurement for that marker. A LoRA with no variant for the shot uses its
  `inject_text`, as before.

### Result display

Pick how the generated image comes back under **Result**:

- **Viewport** *(default)* — set as the active camera's background overlay (shown in
  front) and switch the 3D view to camera view, so the result lands registered to your
  framing. Tune **Overlay Opacity** to flip between the render and your scene. Falls
  back to an Image Editor if the scene has no camera.
- **New Window** — open the result in a standalone Image Editor window.
- **Image Editor** — reuse an open Image Editor (or a new window if none).

**Diffusion Preview** *(on by default)* — shows the image **forming** during
generation: the server decodes a few mid-diffusion frames and the addon updates the
display live (like the progressive preview during diffusion). It adds a little time
(each preview is a VAE decode) and is opt-in — turn it off for the fastest renders.
Works in Viewport/Image Editor modes; in New Window mode previews appear once the
window exists. Currently wired for the Qwen (control-union) models.

## OpenPose rigs

Rigs you append into your scene and put on the OpenPose rig layer. Only two
`.blend` files are tracked — the irreplaceable source and the finished rig.
Everything in between is rebuilt by the scripts beside them.

| file | what |
| --- | --- |
| `Body25_COCO18.blend` | body rig, recoloured to the COCO-18 convention |
| `Hand.blend` | **source** hand rig, as downloaded — never edit in place |
| `Hand_OP21_posable.blend` | **finished** hand rig: correct colours, anatomical proportions, a forearm, joint limits and per-finger curl controls |

### Rebuilding the hand rig

Run from this folder. **Order matters** — the arm step needs the wrist bone that
step 2 adds, and the curl handles have to be placed *after* the bones are
rescaled or they land against the old proportions:

```bash
# 1. colours: the palette the pose encoder actually reads
blender Hand.blend --background --factory-startup \
    --python openpose_hand21.py -- --out Hand_OP21.blend

# 2. wrist bone only (the four finger chains ship unparented)
blender Hand_OP21.blend --background --factory-startup \
    --python hand_rig_constraints.py -- --no-limits --out step1.blend

# 3. anatomical proportions + forearm
blender step1.blend --background --factory-startup \
    --python hand_rig_anatomy.py -- --arm forearm --out step2.blend

# 4. joint limits + per-finger curl controls
blender step2.blend --background --factory-startup \
    --python hand_rig_constraints.py -- --preset anatomical --curl \
    --out Hand_OP21_posable.blend
```

Each script takes `--out PATH` or `--inplace`, and a relative `--out` resolves
against the *source* `.blend`, not the working directory.

`hand_rig_constraints.py` also has a `--preset hinge` (safety rails only, no
anatomical ranges) if the full preset fights a particular pose. Dial a bone's
constraint **Influence** down rather than deleting it.

### Experimental

`openpose_gn_prototype.py` builds the skeleton with Geometry Nodes on top of an
Auto-Rig Pro character instead of using a purpose-built diagram rig. Nothing is
baked, so it regenerates per frame — the shape a control *video* needs. It wants
an ARP rig that is not in this repo, and it covers body plus one hand, no face.

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
  to use FUK's estimator instead (plain OpenPose only — *Depth + OpenPose* requires it).
- Normal-map encoding (`n*0.5+0.5`) is a first pass; normal space may need tuning per
  scene.
