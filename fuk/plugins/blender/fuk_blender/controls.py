"""
Resolve the structural control map to send to FUK.

Native maps (depth/normals/openpose-from-rig) come straight from render_passes().
For FUK-derived maps (canny, or openpose without a rig layer) we call the server's
/api/preprocess on the beauty render and use the absolute output path it returns
(Blender and FUK share the filesystem).
"""

from __future__ import annotations


def _output_path(resp: dict) -> str | None:
    return resp.get("output_path") or resp.get("url")


def derive_control(client, render_result: dict, control_source: str) -> str | None:
    """Return an absolute path to the control map, or None for plain t2i."""
    native = render_result.get("control")
    if native:
        return native

    beauty = render_result.get("beauty")
    if not beauty:
        return None

    if control_source == "canny":
        resp = client.preprocess({"image_path": beauty, "method": "canny"})
        return _output_path(resp)

    if control_source == "openpose":
        resp = client.preprocess({
            "image_path": beauty,
            "method": "openpose",
            "detect_body": True,
            "detect_hand": True,
            "detect_face": True,
        })
        return _output_path(resp)

    return None
