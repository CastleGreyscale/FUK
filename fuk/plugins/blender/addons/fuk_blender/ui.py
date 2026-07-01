"""FUK panel in the 3D viewport sidebar (N-panel, 'FUK' tab)."""

from __future__ import annotations

import textwrap
import bpy


def _wrap(text, width_px):
    """Word-wrap `text` to the panel width (approx chars-per-line from pixels)."""
    cpl = max(16, int((width_px - 24) / 7))
    lines = []
    for para in (text or "").split("\n"):
        lines.extend(textwrap.wrap(para, cpl) or [""])
    return lines


def _draw_growing_text(layout, text, region_width):
    """Draw `text` word-wrapped over as many rows as needed (auto-expands)."""
    lines = _wrap(text, region_width)
    if not text or len(lines) <= 1:
        return
    col = layout.column(align=True)
    col.scale_y = 0.7
    for line in lines:
        col.label(text=line)


class FUK_PT_main(bpy.types.Panel):
    bl_label = "FUK"
    bl_idname = "FUK_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FUK"

    def draw(self, context):
        layout = self.layout
        props = context.scene.fuk

        # --- connection / shot ---
        box = layout.box()
        box.label(text="Shot", icon="FILE_BLEND")
        box.prop(props, "project_folder", text="")
        row = box.row(align=True)
        row.operator("fuk.connect",
                     text="Connected" if props.connected else "Connect",
                     icon="LINKED" if props.connected else "UNLINKED",
                     depress=props.connected)
        row.prop(props, "shot_file", text="")
        row = box.row(align=True)
        row.operator("fuk.load_shot", icon="IMPORT")
        row.operator("fuk.save_shot", icon="EXPORT")

        # --- generation settings ---
        box = layout.box()
        box.label(text="Generation", icon="RENDER_STILL")

        box.label(text="Prompt")
        box.prop(props, "prompt", text="")
        _draw_growing_text(box, props.prompt, context.region.width)
        row = box.row(align=True)
        row.operator("fuk.insert_tag", text="Insert #tag", icon="ADD")
        row.operator("fuk.resolve_preview", text="Preview", icon="VIEWZOOM")
        row.operator("fuk.refresh_tags", text="", icon="FILE_REFRESH")
        if props.resolved_preview:
            sub = box.box()
            sub.scale_y = 0.7
            sub.label(text="Resolves to:", icon="SORTALPHA")
            for line in _wrap(props.resolved_preview, context.region.width):
                sub.label(text=line)

        box.separator()
        box.label(text="Negative")
        box.prop(props, "negative_prompt", text="")
        _draw_growing_text(box, props.negative_prompt, context.region.width)

        box.separator()
        box.prop(props, "model", text="")
        row = box.row(align=True)
        row.prop(props, "steps")
        row.prop(props, "guidance_scale")
        row = box.row(align=True)
        row.prop(props, "seed_mode", text="")
        sub = row.row(align=True)
        sub.enabled = props.seed_mode == "fixed"
        sub.prop(props, "seed", text="")
        if props.last_used_seed:
            r = box.row(align=True)
            r.label(text=f"Last seed: {props.last_used_seed}", icon="KEYINGSET")
            r.operator("fuk.use_last_seed", text="Reuse", icon="FILE_REFRESH")
        box.prop(props, "output_format", text="Format")

        # --- control ---
        box = layout.box()
        box.label(text="Control", icon="MODIFIER")
        box.prop(props, "control_source", text="")
        if props.control_source == "openpose":
            box.prop_search(props, "openpose_view_layer", context.scene, "view_layers", text="Rig Layer")

        # --- preview tuning ---
        box = layout.box()
        box.label(text="Quick Preview", icon="HIDE_OFF")
        row = box.row(align=True)
        row.prop(props, "preview_percentage")
        row.prop(props, "preview_steps")

        # --- result display ---
        box = layout.box()
        box.label(text="Result", icon="IMAGE_DATA")
        box.prop(props, "result_display", text="")
        if props.result_display == "viewport":
            box.prop(props, "bg_alpha", slider=True)
        box.prop(props, "show_diffusion")

        # --- actions ---
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

        # Cancel the in-flight generation (also Esc in the viewport).
        if props.busy:
            row = layout.row()
            row.scale_y = 1.2
            row.operator("fuk.cancel", text="Cancel", icon="CANCEL")

        # Save the current (e.g. preview) result to FUK history on demand.
        if props.last_result:
            row = layout.row()
            row.enabled = (not props.busy) and (not props.result_persisted)
            icon = "CHECKMARK" if props.result_persisted else "EXPORT"
            label = "In History" if props.result_persisted else "Save to History"
            row.operator("fuk.save_to_history", text=label, icon=icon)

        # --- status ---
        row = layout.row()
        row.label(text=props.status, icon="INFO" if not props.busy else "SORTTIME")
        if props.last_result:
            layout.label(text=bpy.path.basename(props.last_result), icon="IMAGE_DATA")
