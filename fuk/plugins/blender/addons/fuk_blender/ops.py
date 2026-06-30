"""Operators: connect, load/save shot, and the render+generate round trip."""

from __future__ import annotations

import os
import json
import time
import bpy

from . import props as props_mod
from . import shot as shot_mod
from . import render as render_mod
from . import controls as controls_mod
from . import viewer as viewer_mod
from .client import FukClient, FukError
from .prefs import get_server_url


def _client(context) -> FukClient:
    return FukClient(get_server_url(context))


def _abs_folder(props) -> str:
    return bpy.path.abspath(props.project_folder) if props.project_folder else ""


def _refresh_tokens(client, props) -> int:
    """Pull prompt tokens into the cache; best-effort. Returns token count."""
    resp = client.prompt_tokens(model=props.model)
    props_mod.TOKEN_CACHE.clear()
    props_mod.TOKEN_CACHE.extend(resp.get("tokens", []))
    return len(props_mod.TOKEN_CACHE)


# Live-mode shared state.
#   count    monotonic counter the handler bumps on real edits
#   rendered count snapshot taken when the last render started
#   suspend  set by the generate operator around its OWN render/display mutations
#            (those emit geometry updates too, so we must gate them out)
# A trailing render is owed whenever count != rendered — this coalesces any number
# of edits made during a render down to a single follow-up render of the latest state.
_LIVE = {"count": 0, "rendered": 0, "last_change": 0.0, "suspend": False}

# The generation currently in flight (so Live can interrupt it). Set by the generate
# operator on submit, cleared when it finishes.
_ACTIVE_GEN = {"id": "", "mode": ""}


# Object types whose geometry updates are genuine edits. The render + display
# re-evaluations show up as geometry updates on the CAMERA and on collections/scene
# (which aren't Objects) — those must be ignored or Live re-triggers itself forever.
_GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT",
                   "VOLUME", "GREASEPENCIL", "GPENCIL", "POINTCLOUD"}


def _depsgraph_handler(scene, depsgraph):
    """Count only genuine transform/geometry edits.

    Filtered out: selection/active-object clicks (no transform/geometry flag), and
    our OWN render/display, which emit `is_updated_geometry` on the camera and the
    scene collections. A camera *move* is a transform; the render artifact is
    geometry-on-camera — so we count object transforms, and geometry only on real
    geometry objects.
    """
    if _LIVE["suspend"]:
        return
    for upd in depsgraph.updates:
        idd = getattr(upd, "id", None)
        is_object = isinstance(idd, bpy.types.Object)
        if upd.is_updated_transform and is_object:
            _LIVE["count"] += 1
            _LIVE["last_change"] = time.time()
            return
        if upd.is_updated_geometry and is_object and idd.type in _GEOMETRY_TYPES:
            _LIVE["count"] += 1
            _LIVE["last_change"] = time.time()
            return


def _live_suspend(on: bool):
    _LIVE["suspend"] = on


def _live_snapshot_rendered():
    """Mark the current edit count as 'rendered' (called when a render starts)."""
    _LIVE["rendered"] = _LIVE["count"]


def _gen_id_from_url(png_url: str) -> str:
    """api/project/cache/<rel>/generated.png -> <rel> (cache-relative entry id)."""
    s = png_url
    for pref in ("/api/project/cache/", "api/project/cache/"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    return s.rsplit("/", 1)[0]


def _resolved_seed(props, seed_used=None):
    """The concrete seed to record (actual used > fixed value > none for random)."""
    if seed_used is not None:
        return int(seed_used)
    if props.last_used_seed:
        return int(props.last_used_seed)
    return None if props.seed_mode == "random" else int(props.seed)


def _enrich_payload(props, render, control_path, gen_id):
    """save-entry payload that enriches the just-generated entry in place."""
    return {
        "generation_id": gen_id,
        "control_path": control_path or None,
        "beauty_path": (render or {}).get("beauty"),
        "control_kind": props.control_source,
        "register_control": True,
        "prompt": props.prompt,
        "negative_prompt": props.negative_prompt or "",
        "model": props.model,
        "seed": _resolved_seed(props),
        "width": (render or {}).get("width", 0),
        "height": (render or {}).get("height", 0),
    }


def _write_io_meta(out_dir, props, render, control_path, result_path, seed_used):
    """Persist the working set in blender_io so Save-to-History can package it."""
    meta = {
        "prompt": props.prompt,
        "negative_prompt": props.negative_prompt,
        "model": props.model,
        "seed": _resolved_seed(props, seed_used),
        "control_kind": props.control_source,
        "control": control_path or "",
        "beauty": (render or {}).get("beauty", ""),
        "result": result_path,
        "width": (render or {}).get("width", 0),
        "height": (render or {}).get("height", 0),
    }
    try:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass


def _create_payload_from_meta(meta):
    """save-entry payload that creates a fresh entry from a working-set meta dict."""
    return {
        "result_path": meta.get("result"),
        "control_path": meta.get("control") or None,
        "beauty_path": meta.get("beauty") or None,
        "control_kind": meta.get("control_kind", "control"),
        "register_control": True,
        "prompt": meta.get("prompt", ""),
        "negative_prompt": meta.get("negative_prompt", ""),
        "model": meta.get("model", ""),
        "seed": meta.get("seed"),
        "width": meta.get("width", 0),
        "height": meta.get("height", 0),
    }


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

        tag_count = 0
        try:
            tag_count = _refresh_tokens(client, props)
        except FukError:
            pass  # tags are optional; don't block connect
        props.status = f"Connected — {count} shot(s), {tag_count} tag(s)"
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


class FUK_OT_refresh_tags(bpy.types.Operator):
    bl_idname = "fuk.refresh_tags"
    bl_label = "Refresh Tags"
    bl_description = "Reload the available #marker prompt tags from FUK"

    def execute(self, context):
        props = context.scene.fuk
        try:
            n = _refresh_tokens(_client(context), props)
        except FukError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"{n} tag(s) available")
        return {"FINISHED"}


class FUK_OT_insert_tag(bpy.types.Operator):
    bl_idname = "fuk.insert_tag"
    bl_label = "Insert Tag"
    bl_description = "Search the FUK vocabulary and insert a #marker into the prompt"
    bl_property = "token"

    token: bpy.props.EnumProperty(name="Tag", items=props_mod.token_enum_items)

    def invoke(self, context, event):
        if not props_mod.TOKEN_CACHE:
            try:
                _refresh_tokens(_client(context), context.scene.fuk)
            except FukError as e:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.fuk
        marker = self.token
        if not marker:
            return {"CANCELLED"}
        sep = "" if (not props.prompt or props.prompt.endswith((" ", "\n", ","))) else " "
        props.prompt = f"{props.prompt}{sep}{marker} "
        return {"FINISHED"}


class FUK_OT_resolve_preview(bpy.types.Operator):
    bl_idname = "fuk.resolve_preview"
    bl_label = "Preview Expansion"
    bl_description = "Resolve #markers (and mood) exactly as generation will — preview the final prompt"

    def execute(self, context):
        props = context.scene.fuk
        try:
            res = _client(context).prompt_resolve(props.prompt, model=props.model, apply_mood=True)
        except FukError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        props.resolved_preview = res.get("resolved", "")
        unknown = res.get("unknown_markers", [])
        if unknown:
            self.report({"WARNING"}, f"Unknown markers: {', '.join(unknown)}")
        else:
            self.report({"INFO"}, "Prompt resolved")
        return {"FINISHED"}


class FUK_OT_use_last_seed(bpy.types.Operator):
    bl_idname = "fuk.use_last_seed"
    bl_label = "Reuse Last Seed"
    bl_description = "Switch to Fixed mode using the last seed FUK actually used"

    def execute(self, context):
        props = context.scene.fuk
        if not props.last_used_seed:
            self.report({"WARNING"}, "No seed recorded yet")
            return {"CANCELLED"}
        props.seed_mode = "fixed"
        props.seed = props.last_used_seed
        return {"FINISHED"}


class FUK_OT_save_to_history(bpy.types.Operator):
    bl_idname = "fuk.save_to_history"
    bl_label = "Save to History"
    bl_description = "Save the current result as a complete FUK history entry (result + control + source)"

    def execute(self, context):
        props = context.scene.fuk
        if props.result_persisted:
            self.report({"INFO"}, "Already in history")
            return {"CANCELLED"}
        meta_path = os.path.join(props.working_dir, "meta.json") if props.working_dir else ""
        if not meta_path or not os.path.exists(meta_path):
            self.report({"ERROR"}, "Nothing to save — run a Quick Preview first")
            return {"CANCELLED"}
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            self.report({"ERROR"}, f"Could not read working set: {e}")
            return {"CANCELLED"}
        if not meta.get("result") or not os.path.exists(meta["result"]):
            self.report({"ERROR"}, "Result image missing from working set")
            return {"CANCELLED"}

        client = _client(context)
        try:
            # Make sure the server is on this shot so the entry lands in its cache.
            client.set_project_folder(_abs_folder(props))
            if props.shot_file:
                client.load_shot(props.shot_file)
            client.save_entry(_create_payload_from_meta(meta))
        except FukError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        props.result_persisted = True
        props.status = "Saved to history"
        self.report({"INFO"}, "Saved to history")
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
    _render = None        # render_passes result (beauty, control, kind, w, h)
    _control_path = ""    # the structural map actually sent to FUK
    _last_preview = ""    # last diffusion-preview URL displayed
    _cancelling = False   # an Esc-stop is in progress

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
        # One working dir per shot — preview and full overwrite it; "current result".
        out_dir = os.path.join(folder, "cache", "_blender_io", shot_stem)

        # Snapshot the live edit count now, and suspend live change-detection: the
        # render itself emits geometry updates that must not look like user edits.
        _live_snapshot_rendered()
        _live_suspend(True)

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
                "live_preview": bool(props.show_diffusion),
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
            _live_suspend(False)
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        # Render + submit done; resume live detection during the diffusion wait so
        # edits made now are noticed and a trailing render runs when this finishes.
        _live_suspend(False)

        self._client = client
        self._out_dir = out_dir
        self._render = result
        self._control_path = control_path or ""
        self._last_preview = ""
        self._cancelling = False
        self._t0 = time.time()
        props.busy = True
        _ACTIVE_GEN.update({"id": self._gen_id, "mode": self.mode})
        props.status = "Queued..."

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.6, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = context.scene.fuk
        if event.type == "ESC" and not self._cancelling:
            # Request a real stop, but keep the modal alive (busy=True) until the
            # server confirms the abort finished — so nothing starts concurrently.
            try:
                self._client.cancel(self._gen_id)
            except FukError:
                pass
            self._cancelling = True
            props.status = "Cancelling…"
            return {"PASS_THROUGH"}
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
            # Live diffusion view: show each mid-denoise preview as it arrives.
            preview_url = (st.get("outputs", {}) or {}).get("preview")
            if preview_url and preview_url != self._last_preview:
                self._last_preview = preview_url
                try:
                    pdest = os.path.join(self._out_dir, "preview.png")
                    self._client.download(preview_url, pdest)
                    pimg = viewer_mod.load_result_image(pdest)
                    viewer_mod.show_result(context, pimg, props.result_display,
                                           props.bg_alpha, reuse_only=True)
                except (FukError, RuntimeError):
                    pass
            return {"PASS_THROUGH"}

        if status == "complete":
            png_url = (st.get("outputs", {}) or {}).get("png")
            if not png_url:
                return self._finish(context, "Complete (no image returned)")

            # Capture the seed FUK actually used (esp. for random mode) BEFORE we
            # touch the entry, since previews delete it.
            seed_used = None
            try:
                meta = self._client.generation_metadata(png_url)
                sd = meta.get("seed")
                if sd not in (None, "", "null"):
                    seed_used = max(0, int(sd))
                    props.last_used_seed = seed_used
                    props.seed = seed_used
            except (FukError, ValueError, TypeError):
                pass

            dest = os.path.join(self._out_dir, "result.png")
            try:
                self._client.download(png_url, dest)
                img = viewer_mod.load_result_image(dest)
                props.last_result = dest
                # Displaying mutates the camera (background image) and emits geometry
                # updates — suspend live detection across it so it isn't seen as an edit.
                _live_suspend(True)
                shown = viewer_mod.show_result(context, img, props.result_display, props.bg_alpha)
            except (FukError, RuntimeError) as e:
                self.report({"WARNING"}, f"Generated, but display failed: {e}")
                shown = None
            finally:
                _live_suspend(False)

            # Record the working set so Save-to-History can package it later.
            props.working_dir = self._out_dir
            _write_io_meta(self._out_dir, props, self._render, self._control_path, dest, seed_used)

            # Apply the persistence policy: previews are ephemeral, Full persists.
            gen_id = _gen_id_from_url(png_url)
            note = ""
            try:
                if self.mode == "preview":
                    self._client.delete_generation(gen_id)  # drop the auto-created entry
                    props.result_persisted = False
                    note = " (preview — not saved)"
                else:
                    self._client.save_entry(_enrich_payload(props, self._render, self._control_path, gen_id))
                    props.result_persisted = True
                    note = " — saved to history"
            except FukError as e:
                self.report({"WARNING"}, f"History update failed: {e}")

            tail = (f" — shown in {shown}" if shown else "") + note
            return self._finish(context, f"Done in {elapsed:.0f}s{tail}")

        if status == "failed":
            return self._finish(context, f"Failed: {st.get('error','')}")

        if status == "cancelled":
            # Hold busy until the server's diffusion thread has actually aborted
            # (phase flips 'cancelling' -> 'cancelled'), so a restart can't overlap
            # the still-running pipeline (DiffSynth isn't thread-safe).
            if phase == "cancelled":
                return self._finish(context, "Cancelled")
            props.status = "Cancelling…"
            return {"PASS_THROUGH"}

        return {"PASS_THROUGH"}

    def _finish(self, context, message):
        props = context.scene.fuk
        props.busy = False
        props.status = message
        _ACTIVE_GEN.update({"id": "", "mode": ""})
        _live_suspend(False)  # safety: never leave live detection suspended
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self.report({"INFO"}, message)
        return {"FINISHED"}


class FUK_OT_live(bpy.types.Operator):
    bl_idname = "fuk.live"
    bl_label = "Live"
    bl_description = "Toggle live mode: auto-run a Quick Preview after camera/object edits settle"

    _timer = None

    def invoke(self, context, event):
        props = context.scene.fuk
        # Button acts as a toggle — a second press turns the running watcher off.
        if props.live_mode:
            props.live_mode = False
            return {"FINISHED"}
        if not props.shot_file:
            self.report({"ERROR"}, "Connect and load a shot first")
            return {"CANCELLED"}

        # Clear any stale lock (e.g. left set by an interrupted run) — enabling Live
        # is a deliberate fresh start, so nothing should be mid-generation here.
        props.busy = False
        props.live_mode = True
        # Start clean: nothing owed until the next real edit.
        _LIVE.update({"count": 0, "rendered": 0, "last_change": time.time(), "suspend": False})
        if _depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_depsgraph_handler)
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        props.status = "Live: watching"
        self.report({"INFO"}, "Live mode on")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = context.scene.fuk
        if not props.live_mode:
            return self._stop(context, "Live: off")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        # A generation is running (this one's, or a manual Render Full). The generate
        # operator owns suspend + the rendered snapshot; edits made during its
        # diffusion wait bump `count` past `rendered`, owing one trailing render.
        if props.busy:
            owed = _LIVE["count"] != _LIVE["rendered"]
            settled = (time.time() - _LIVE["last_change"]) >= props.live_delay
            # Interrupt & restart: a new edit settled while a Live *preview* is running.
            if (props.live_interrupt and owed and settled
                    and _ACTIVE_GEN.get("mode") == "preview" and _ACTIVE_GEN.get("id")):
                try:
                    _client(context).cancel(_ACTIVE_GEN["id"])
                except FukError:
                    pass
                _ACTIVE_GEN["id"] = ""  # one request; the generate op handles the rest
                props.status = "Live: restarting…"
            return {"PASS_THROUGH"}

        # Visible state so the loop is observable: did it see your edit? is it settling?
        owed = _LIVE["count"] != _LIVE["rendered"]
        if not owed:
            props.status = "Live: watching"
        elif (time.time() - _LIVE["last_change"]) < props.live_delay:
            props.status = "Live: change — settling…"
        else:
            props.status = "Live: rendering…"
            try:
                bpy.ops.fuk.generate("INVOKE_DEFAULT", mode="preview")
            except RuntimeError as e:
                self.report({"WARNING"}, f"Live preview failed: {e}")
        return {"PASS_THROUGH"}

    def _stop(self, context, message):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
        _LIVE["suspend"] = False
        context.scene.fuk.status = message
        return {"FINISHED"}


CLASSES = (
    FUK_OT_connect,
    FUK_OT_load_shot,
    FUK_OT_save_shot,
    FUK_OT_refresh_tags,
    FUK_OT_insert_tag,
    FUK_OT_resolve_preview,
    FUK_OT_use_last_seed,
    FUK_OT_save_to_history,
    FUK_OT_generate,
    FUK_OT_live,
)
