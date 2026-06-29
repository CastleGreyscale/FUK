"""FUK panel in the 3D viewport sidebar (N-panel, 'FUK' tab)."""

from __future__ import annotations

import bpy


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
        row.operator("fuk.connect", icon="LINKED")
        row.prop(props, "shot_file", text="")
        row = box.row(align=True)
        row.operator("fuk.load_shot", icon="IMPORT")
        row.operator("fuk.save_shot", icon="EXPORT")

        # --- generation settings ---
        box = layout.box()
        box.label(text="Generation", icon="RENDER_STILL")
        box.prop(props, "prompt", text="Prompt")
        box.prop(props, "negative_prompt", text="Negative")
        box.prop(props, "model", text="")
        row = box.row(align=True)
        row.prop(props, "steps")
        row.prop(props, "guidance_scale")
        row = box.row(align=True)
        row.prop(props, "seed_mode", text="")
        sub = row.row(align=True)
        sub.enabled = props.seed_mode == "fixed"
        sub.prop(props, "seed", text="")
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

        # --- actions ---
        col = layout.column(align=True)
        col.enabled = not props.busy
        col.scale_y = 1.3
        col.operator("fuk.generate", text="Quick Preview", icon="HIDE_OFF").mode = "preview"
        col.operator("fuk.generate", text="Render Full", icon="RENDER_STILL").mode = "full"

        # --- status ---
        row = layout.row()
        row.label(text=props.status, icon="INFO" if not props.busy else "SORTTIME")
        if props.last_result:
            layout.label(text=bpy.path.basename(props.last_result), icon="IMAGE_DATA")
