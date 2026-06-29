"""Addon preferences — currently just the FUK server URL."""

from __future__ import annotations

import bpy

# Top-level addon package name ("fuk_blender"), used as the AddonPreferences id.
ADDON_ID = __package__.split(".")[0]


class FukAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    server_url: bpy.props.StringProperty(
        name="Server URL",
        description="Base URL of the running FUK web server",
        default="http://localhost:8000",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "server_url")
        layout.label(text="Start the FUK server with: python fuk/start_web_ui.py", icon="INFO")


def get_prefs(context) -> "FukAddonPreferences":
    return context.preferences.addons[ADDON_ID].preferences


def get_server_url(context) -> str:
    try:
        return get_prefs(context).server_url
    except (KeyError, AttributeError):
        return "http://localhost:8000"
