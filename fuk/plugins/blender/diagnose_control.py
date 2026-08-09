"""
Why is no control EXR being written?

Run this from Blender's Scripting tab with the .blend that reproduces the problem
open (Text > Open, pick this file, then Run Script).

You do NOT need a terminal. The report is written to:
  * a Text datablock called "FUK_diagnosis" — the Text Editor switches to it
    automatically when the script finishes, so just read it there; and
  * a file next to your .blend (or /tmp if the file is unsaved), path noted at the end.

It reports the scene state the control render depends on, then performs the same
render the addon does — in BOTH Quick Preview and Full modes, into a throwaway
directory — and reports which files actually appeared in each.
"""

import os
import glob
import tempfile

import bpy

LINES = []
SUSPECTS = []


def _p(msg=""):
    LINES.append(msg)
    print(msg, flush=True)


def _fail(msg):
    SUSPECTS.append(msg)
    _p("  *** SUSPECT ***  " + msg)


def _test_render(mode_name, preview):
    """Run render_passes into a temp dir; return True if a control EXR appeared."""
    from fuk_blender import render as R

    scene = bpy.context.scene
    # Preview swaps to EEVEE; Full leaves the scene's engine alone.
    engine = (R._eevee_engine() or scene.render.engine) if preview else scene.render.engine
    out = tempfile.mkdtemp(prefix="fuk_diag_")
    try:
        result = R.render_passes(bpy.context, out, "depth",
                                 preview=preview, preview_percentage=25)
        produced = sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, "*")))
        exr = [f for f in produced if f.endswith(".exr")]
        _p(f"  {mode_name:14s} engine={engine} files={produced or 'NOTHING'}")
        _p(f"  {mode_name:14s} control -> {result.get('control')}")
        if not exr:
            _fail(f"{mode_name}: render completed but NO control .exr was written")
        return bool(exr)
    except Exception as e:
        _p(f"  {mode_name:14s} RAISED {type(e).__name__}: {e}")
        _fail(f"{mode_name}: render_passes raised {type(e).__name__}")
        return False
    finally:
        for f in glob.glob(os.path.join(out, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(out)
        except OSError:
            pass


def report():
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    _p("=" * 68)
    _p("FUK control-pass diagnosis")
    _p("=" * 68)
    _p(f"Blender        : {bpy.app.version_string}")
    _p(f"Scene engine   : {scene.render.engine}")
    _p(f"Scene          : {scene.name}")
    _p(f"Camera         : {scene.camera.name if scene.camera else '<NONE>'}")
    _p(f"Resolution     : {scene.render.resolution_x}x{scene.render.resolution_y}"
       f" @ {scene.render.resolution_percentage}%")
    if scene.camera is None:
        _fail("no scene camera — nothing to render")

    _p()
    _p("--- View layers ---")
    _p(f"Active view layer: {view_layer.name!r}")
    for vl in scene.view_layers:
        marks = []
        if vl is view_layer:
            marks.append("ACTIVE")
        if not vl.use:
            marks.append("DISABLED FOR RENDER")
        _p(f"  {vl.name!r:26s} use={vl.use!s:5s} pass_z={vl.use_pass_z!s:5s} "
           f"pass_normal={vl.use_pass_normal!s:5s} {' '.join(marks)}")
    if not view_layer.use:
        _fail("the ACTIVE view layer is disabled, so it never renders and its "
              "Render Layers node feeds the File Output node nothing")
    if not any(vl.use for vl in scene.view_layers):
        _fail("NO view layer is enabled for render at all")

    _p()
    _p("--- Compositor state (before the addon touches it) ---")
    _p(f"scene.render.use_compositing: {scene.render.use_compositing}")
    grp = scene.compositing_node_group
    _p(f"scene.compositing_node_group: {grp.name if grp else None}")
    leftovers = [g.name for g in bpy.data.node_groups if g.name.startswith("FUK_BLENDER_COMP")]
    _p(f"leftover FUK node groups    : {leftovers or 'none'}")
    if leftovers:
        _p("     (means a previous run didn't reach its cleanup)")

    _p()
    _p("--- Render Layers 'Depth' socket ---")
    probe = bpy.data.node_groups.new("FUK_DIAG_PROBE", "CompositorNodeTree")
    try:
        was_z = view_layer.use_pass_z
        view_layer.use_pass_z = True
        rl = probe.nodes.new("CompositorNodeRLayers")
        rl.scene = scene
        rl.layer = view_layer.name
        _p(f"available outputs: {[s.name for s in rl.outputs]}")
        depth = rl.outputs.get("Depth")
        if depth is None:
            _fail("no 'Depth' output on this engine/layer")
        else:
            _p(f"Depth socket enabled = {depth.enabled}")
            if not depth.enabled:
                _fail("Depth exists but is disabled — the Z pass isn't producing data")
        view_layer.use_pass_z = was_z
    finally:
        bpy.data.node_groups.remove(probe)

    _p()
    _p("--- Live test renders (temp dir; your project is untouched) ---")
    try:
        import fuk_blender.render  # noqa: F401
    except ImportError:
        _p("Could not import fuk_blender.render — is the FUK Bridge addon enabled?")
        return

    # The failing run in the log was a Quick Preview, which swaps the engine to EEVEE
    # and drops the resolution — test both paths so a preview-only failure is visible.
    full_ok = _test_render("FULL", preview=False)
    prev_ok = _test_render("QUICK PREVIEW", preview=True)

    _p()
    _p("--- Conclusion ---")
    if full_ok and prev_ok and not SUSPECTS:
        _p("Both modes wrote a control EXR and nothing looks wrong.")
        _p("The control is fine; if generation still repeats the cause is downstream.")
    elif full_ok and not prev_ok:
        _p("FULL works but QUICK PREVIEW does not — the failure is preview-specific")
        _p("(engine swap / resolution change). That narrows it a lot.")
    elif prev_ok and not full_ok:
        _p("QUICK PREVIEW works but FULL does not.")
    elif not full_ok and not prev_ok:
        _p("Neither mode wrote a control EXR — the compositor's File Output node is")
        _p("not writing in this scene at all, though the beauty render succeeds.")
    if SUSPECTS:
        _p()
        _p("Suspects found:")
        for s in SUSPECTS:
            _p(f"  - {s}")


def _deliver():
    text = "\n".join(LINES)

    # Text datablock — readable without a terminal.
    name = "FUK_diagnosis"
    txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    txt.clear()
    txt.write(text)

    # File on disk, so it can be copied out easily.
    blend = bpy.data.filepath
    folder = os.path.dirname(blend) if blend else tempfile.gettempdir()
    path = os.path.join(folder, "fuk_diagnosis.txt")
    try:
        with open(path, "w") as f:
            f.write(text + "\n")
    except OSError as e:
        path = f"<could not write: {e}>"

    txt.write(f"\n\nSaved to: {path}\n")

    # Swap the Text Editor this was run from over to the report.
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "TEXT_EDITOR":
                area.spaces.active.text = txt
                area.tag_redraw()
    print(f"\n[FUK] diagnosis written to Text datablock {name!r} and {path}", flush=True)


try:
    report()
finally:
    _deliver()
