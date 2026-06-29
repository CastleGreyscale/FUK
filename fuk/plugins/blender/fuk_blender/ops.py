"""Operators: connect, load/save shot, and the render+generate round trip."""

from __future__ import annotations

import os
import time
import bpy

from . import props as props_mod
from . import shot as shot_mod
from . import render as render_mod
from . import controls as controls_mod
from .client import FukClient, FukError
from .prefs import get_server_url


def _client(context) -> FukClient:
    return FukClient(get_server_url(context))


def _abs_folder(props) -> str:
    return bpy.path.abspath(props.project_folder) if props.project_folder else ""


def _show_image(context, img) -> bool:
    for area in context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            area.spaces.active.image = img
            return True
    return False


class FUK_OT_connect(bpy.types.Operator):
    bl_idname = "fuk.connect"
    bl_label = "Connect"
    bl_description = "Reach the FUK server, set the project folder, and list shots"

    def execute(self, context):
        props = context.scene.fuk
        folder = _abs_folder(props)
        if not folder or not os.path.isdir(folder):
            self.report({"ERROR"}, "Pick a valid project folder first")
            return {"CANCELLED"}
        client = _client(context)
        try:
            client.health()
            client.set_project_folder(folder)
            resp = client.list_shots()
        except FukError as e:
            props.status = "Not connected"
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        props_mod.SHOT_CACHE.clear()
        props_mod.SHOT_CACHE.extend(resp.get("files", []))
        count = len(props_mod.SHOT_CACHE)
        if count and props_mod.SHOT_CACHE[0].get("name"):
            props.shot_file = props_mod.SHOT_CACHE[0]["name"]
        props.status = f"Connected — {count} shot(s)"
        self.report({"INFO"}, props.status)
        return {"FINISHED"}


class FUK_OT_load_shot(bpy.types.Operator):
    bl_idname = "fuk.load_shot"
    bl_label = "Load Shot"
    bl_description = "Load the selected shot JSON into the panel (shot is the source of truth)"

    def execute(self, context):
        props = context.scene.fuk
        if not props.shot_file:
            self.report({"ERROR"}, "No shot selected — Connect first")
            return {"CANCELLED"}
        try:
            shot_mod.pull(_client(context), props.shot_file, props)
        except FukError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, props.status)
        return {"FINISHED"}


class FUK_OT_save_shot(bpy.types.Operator):
    bl_idname = "fuk.save_shot"
    bl_label = "Save to Shot"
    bl_description = "Write the panel settings back into the shot JSON"

    def execute(self, context):
        props = context.scene.fuk
        if not props.shot_file:
            self.report({"ERROR"}, "No shot selected")
            return {"CANCELLED"}
        try:
            shot_mod.push(_client(context), props.shot_file, props)
        except FukError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, props.status)
        return {"FINISHED"}


class FUK_OT_generate(bpy.types.Operator):
    bl_idname = "fuk.generate"
    bl_label = "Generate"
    bl_description = "Render the scene's control passes and send them to FUK"

    mode: bpy.props.EnumProperty(
        items=[("preview", "Preview", ""), ("full", "Full", "")],
        default="full",
        options={"HIDDEN"},
    )

    _timer = None
    _client = None
    _gen_id = ""
    _out_dir = ""
    _t0 = 0.0

    def invoke(self, context, event):
        props = context.scene.fuk
        if props.busy:
            self.report({"WARNING"}, "A generation is already running")
            return {"CANCELLED"}
        if not props.shot_file:
            self.report({"ERROR"}, "No shot selected — Connect and Load a shot first")
            return {"CANCELLED"}

        folder = _abs_folder(props)
        client = _client(context)
        shot_stem = os.path.splitext(props.shot_file)[0]
        out_dir = os.path.join(folder, "cache", "_blender_io", shot_stem, self.mode)

        try:
            # Make sure the server is pointed at this shot so outputs land in its cache.
            client.set_project_folder(folder)
            client.load_shot(props.shot_file)

            props.status = "Rendering passes..."
            preview = self.mode == "preview"
            result = render_mod.render_passes(
                context, out_dir, props.control_source,
                preview=preview,
                preview_percentage=props.preview_percentage,
                openpose_view_layer=props.openpose_view_layer,
            )

            props.status = "Preparing control map..."
            control_path = controls_mod.derive_control(client, result, props.control_source)
            if props.control_source not in render_mod.FUK_DERIVED and not control_path:
                self.report({"WARNING"}, f"No {props.control_source} map produced — running without control")

            steps = props.preview_steps if preview else props.steps
            payload = {
                "prompt": props.prompt,
                "negative_prompt": props.negative_prompt or None,
                "model": props.model,
                "steps": int(steps),
                "guidance_scale": float(props.guidance_scale),
                "seed": None if props.seed_mode == "random" else int(props.seed),
                "width": result["width"],
                "height": result["height"],
                "output_format": props.output_format,
            }
            if control_path:
                payload["control_image_paths"] = [control_path]

            props.status = "Submitting to FUK..."
            resp = client.generate_image(payload)
            self._gen_id = resp.get("generation_id", "")
            if not self._gen_id:
                raise FukError("Server did not return a generation id")
        except (FukError, RuntimeError, KeyError, OSError) as e:
            props.busy = False
            props.status = "Failed"
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        self._client = client
        self._out_dir = out_dir
        self._t0 = time.time()
        props.busy = True
        props.status = "Queued..."

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.6, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = context.scene.fuk
        if event.type == "ESC":
            try:
                self._client.cancel(self._gen_id)
            except FukError:
                pass
            return self._finish(context, "Cancelled")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            st = self._client.status(self._gen_id)
        except FukError as e:
            self.report({"ERROR"}, str(e))
            return self._finish(context, "Failed")

        status = st.get("status", "")
        phase = st.get("phase", "")
        progress = st.get("progress", 0.0) or 0.0
        elapsed = time.time() - self._t0

        if status in ("running", "queued"):
            props.status = f"{phase or status} {progress*100:.0f}% ({elapsed:.0f}s)"
            return {"PASS_THROUGH"}

        if status == "complete":
            png_url = (st.get("outputs", {}) or {}).get("png")
            if not png_url:
                return self._finish(context, "Complete (no image returned)")
            dest = os.path.join(self._out_dir, "result.png")
            try:
                self._client.download(png_url, dest)
                img = bpy.data.images.load(dest, check_existing=True)
                img.reload()
                props.last_result = dest
                shown = _show_image(context, img)
            except (FukError, RuntimeError) as e:
                self.report({"WARNING"}, f"Generated, but display failed: {e}")
                shown = False
            tail = "" if shown else " (open it in an Image Editor)"
            return self._finish(context, f"Done in {elapsed:.0f}s{tail}")

        if status in ("failed", "cancelled"):
            return self._finish(context, f"{status.capitalize()}: {st.get('error','')}")

        return {"PASS_THROUGH"}

    def _finish(self, context, message):
        props = context.scene.fuk
        props.busy = False
        props.status = message
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self.report({"INFO"}, message)
        return {"FINISHED"}


CLASSES = (
    FUK_OT_connect,
    FUK_OT_load_shot,
    FUK_OT_save_shot,
    FUK_OT_generate,
)
