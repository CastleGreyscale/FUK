"""
FUK panels in the 3D viewport sidebar (N-panel, 'FUK' tab).

FUK_PT_main holds the connection/shot binding plus the run status; everything else
hangs off it as a collapsible sub-panel. The split exists mostly for the prompt: a
long one wraps to a dozen echo rows, and before the split that pushed the render
buttons off the bottom of the sidebar.
"""

from __future__ import annotations

import textwrap
import bpy


def wrap_text(text, width_px):
    """Word-wrap `text` to a given pixel width (approx chars-per-line from pixels)."""
    cpl = max(16, int((width_px - 24) / 7))
    lines = []
    for para in (text or "").split("\n"):
        lines.extend(textwrap.wrap(para, cpl) or [""])
    return lines


def draw_wrapped(layout, text, width_px):
    """Echo `text` word-wrapped in a sunken box below its field.

    Blender has no multi-line string widget — the field itself only ever shows one
    line — so this read-only echo is the only way to see a long prompt in full.
    Skipped when the text already fits on the single line.
    """
    lines = wrap_text(text, width_px)
    if not text or len(lines) <= 1:
        return
    col = layout.box().column(align=True)
    col.scale_y = 0.85
    for line in lines:
        col.label(text=line)


class _FukSubPanel:
    """Shared bl_ attributes for the sub-panels under FUK_PT_main."""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FUK"
    bl_parent_id = "FUK_PT_main"


class FUK_PT_main(bpy.types.Panel):
    bl_label = "FUK"
    bl_idname = "FUK_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FUK"

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk

        col = layout.column(align=True)
        col.prop(props, "project_folder", text="")
        row = col.row(align=True)
        row.operator("fuk.connect",
                     text="Connected" if props.connected else "Connect",
                     icon="LINKED" if props.connected else "UNLINKED",
                     depress=props.connected)
        row.prop(props, "shot_file", text="")
        # Re-lists the folder: shots created in FUK after connecting aren't in the
        # dropdown until this is pressed.
        row.operator("fuk.refresh_shots", text="", icon="FILE_REFRESH")
        row = col.row(align=True)
        row.operator("fuk.load_shot", icon="IMPORT")
        row.operator("fuk.save_shot", icon="EXPORT")

        # Status and Cancel live on the parent so they stay visible no matter how the
        # sub-panels below are folded.
        if props.busy:
            row = layout.row()
            row.scale_y = 1.2
            row.operator("fuk.cancel", text="Cancel", icon="CANCEL")
        layout.label(text=props.status, icon="SORTTIME" if props.busy else "INFO")
        if props.last_result:
            layout.label(text=bpy.path.basename(props.last_result), icon="IMAGE_DATA")


class FUK_PT_prompt(_FukSubPanel, bpy.types.Panel):
    bl_label = "Prompt"
    bl_idname = "FUK_PT_prompt"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk
        width = context.region.width

        row = layout.row(align=True)
        row.prop(props, "prompt", text="")
        row.operator("fuk.edit_prompt", text="", icon="GREASEPENCIL")
        draw_wrapped(layout, props.prompt, width)

        row = layout.row(align=True)
        row.operator("fuk.insert_tag", text="Insert #tag", icon="ADD")
        row.operator("fuk.resolve_preview", text="Preview", icon="VIEWZOOM")
        row.operator("fuk.refresh_tags", text="", icon="FILE_REFRESH")
        if props.resolved_preview:
            sub = layout.box().column(align=True)
            sub.scale_y = 0.85
            sub.label(text="Resolves to:", icon="SORTALPHA")
            for line in wrap_text(props.resolved_preview, width):
                sub.label(text=line)

        layout.separator()
        layout.label(text="Negative")
        row = layout.row(align=True)
        row.prop(props, "negative_prompt", text="")
        row.operator("fuk.edit_prompt", text="", icon="GREASEPENCIL")
        draw_wrapped(layout, props.negative_prompt, width)


class FUK_PT_generation(_FukSubPanel, bpy.types.Panel):
    bl_label = "Generation"
    bl_idname = "FUK_PT_generation"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk

        layout.prop(props, "model", text="")
        row = layout.row(align=True)
        row.prop(props, "steps")
        row.prop(props, "guidance_scale")
        row = layout.row(align=True)
        row.prop(props, "seed_mode", text="")
        sub = row.row(align=True)
        # Increment needs a start value too — only Random has nothing to type.
        sub.enabled = props.seed_mode != "random"
        sub.prop(props, "seed", text="")
        if props.last_used_seed:
            row = layout.row(align=True)
            row.label(text=f"Last seed: {props.last_used_seed}", icon="KEYINGSET")
            row.operator("fuk.use_last_seed", text="Reuse", icon="FILE_REFRESH")
        layout.prop(props, "output_format", text="Format")


class FUK_PT_control(_FukSubPanel, bpy.types.Panel):
    bl_label = "Control"
    bl_idname = "FUK_PT_control"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk

        layout.prop(props, "control_source", text="")
        if props.control_source == "openpose":
            layout.prop_search(props, "openpose_view_layer", context.scene,
                               "view_layers", text="Rig Layer")


class FUK_PT_result(_FukSubPanel, bpy.types.Panel):
    bl_label = "Result"
    bl_idname = "FUK_PT_result"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk

        layout.prop(props, "result_display", text="")
        if props.result_display == "viewport":
            layout.prop(props, "bg_alpha", slider=True)
        layout.prop(props, "show_diffusion")


class FUK_PT_render(_FukSubPanel, bpy.types.Panel):
    bl_label = "Render"
    bl_idname = "FUK_PT_render"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk

        row = layout.row(align=True)
        row.prop(props, "preview_percentage")
        row.prop(props, "preview_steps")

        col = layout.column(align=True)
        sub = col.column(align=True)
        sub.enabled = not props.busy
        sub.scale_y = 1.3
        sub.operator("fuk.generate", text="Quick Preview (not saved)", icon="HIDE_OFF").mode = "preview"
        sub.operator("fuk.generate", text="Render Full (saves)", icon="RENDER_STILL").mode = "full"

        # Live (IPR-style) auto-update on camera/object edits.
        row = col.row(align=True)
        row.operator("fuk.live", text="Live", icon="REC", depress=props.live_mode)
        row.prop(props, "live_delay", text="Delay")
        row.prop(props, "live_interrupt", text="", icon="TRACKING_CLEAR_FORWARDS")

        # Save the current (e.g. preview) result to FUK history on demand.
        if props.last_result:
            row = layout.row()
            row.enabled = (not props.busy) and (not props.result_persisted)
            icon = "CHECKMARK" if props.result_persisted else "EXPORT"
            label = "In History" if props.result_persisted else "Save to History"
            row.operator("fuk.save_to_history", text=label, icon=icon)


class FUK_PT_video(_FukSubPanel, bpy.types.Panel):
    bl_label = "Video (VACE)"
    bl_idname = "FUK_PT_video"
    bl_order = 5
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk
        scene = context.scene

        frames = scene.frame_end - scene.frame_start + 1
        layout.label(text=f"Range {scene.frame_start}–{scene.frame_end} "
                          f"({frames}f @ {scene.render.fps}fps)")
        if props.control_source not in ("depth", "normals", "openpose"):
            layout.label(text="Control must be depth / normals / openpose", icon="ERROR")
        # effective output size at the chosen scale
        vw = scene.render.resolution_x * props.video_percentage // 100
        vh = scene.render.resolution_y * props.video_percentage // 100
        row = layout.row(align=True)
        row.prop(props, "video_percentage")
        row.label(text=f"{vw}x{vh}")
        row = layout.row(align=True)
        row.prop(props, "video_steps")
        row.prop(props, "video_guidance")
        col = layout.column(align=True)
        col.enabled = (not props.busy) and bool(props.last_result)
        col.scale_y = 1.2
        col.operator("fuk.generate_video", text="Generate Video", icon="RENDER_ANIMATION")
        if not props.last_result:
            layout.label(text="Generate a still first (VACE reference)", icon="INFO")


CLASSES = (
    FUK_PT_main,
    FUK_PT_prompt,
    FUK_PT_generation,
    FUK_PT_control,
    FUK_PT_result,
    FUK_PT_render,
    FUK_PT_video,
)
