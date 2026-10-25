#!/usr/bin/env python3
"""
Self-check the PhotoCraft bridge against a running PhotoCraft.

Run it after a PhotoCraft update: the bridge leans on behaviour of the desktop
control server that the docs do not promise (see the README's probe notes),
and this exercises exactly that path on a generated test document. It never
touches your own documents; the test document is closed at the end.

    launch_photocraft.sh &
    venv/bin/python fuk/plugins/photocraft/pc_probe.py [--port 47823] [--root DIR] [--keep]
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))
import photocraft_bridge as pb  # noqa: E402
from photocraft_bridge import Exchange, PhotoCraftClient, PhotoCraftError  # noqa: E402

W, H = 3000, 2000   # over 2048 on purpose: doc.render-style size limits must not bite


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def write_test_ora(path: Path) -> None:
    """Plate + an offset visible @-layer + a hidden @-layer + a hidden _mask."""
    ramp = np.tile(np.linspace(0, 255, W).astype(np.uint8), (H, 1))
    plate = np.dstack([ramp, ramp[:, ::-1], np.full_like(ramp, 90)])
    car = np.full((300, 400, 4), (255, 0, 0, 255), np.uint8)
    dog = np.full((600, 600, 4), (0, 0, 255, 255), np.uint8)
    mask = np.full((600, 600, 4), 255, np.uint8)
    stack = f"""<?xml version='1.0' encoding='UTF-8'?>
<image w="{W}" h="{H}"><stack>
  <layer name="_mask" src="data/mask.png" x="1200" y="200" visibility="hidden"/>
  <layer name="@a red car" src="data/car.png" x="500" y="700" visibility="visible"/>
  <layer name="@a dog" src="data/dog.png" x="1800" y="900" visibility="hidden"/>
  <layer name="plate" src="data/plate.png" x="0" y="0" visibility="visible"/>
</stack></image>"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "image/openraster", compress_type=zipfile.ZIP_STORED)
        z.writestr("stack.xml", stack)
        for name, arr in (("mask", mask), ("car", car), ("dog", dog), ("plate", plate)):
            z.writestr(f"data/{name}.png", _png(arr))
        z.writestr("mergedimage.png", _png(plate))


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=47823)
    pictures = subprocess.run(["xdg-user-dir", "PICTURES"], capture_output=True, text=True).stdout.strip()
    ap.add_argument("--root", default=str(Path(pictures or Path.home() / "Pictures") / "fuk_photocraft"))
    ap.add_argument("--keep", action="store_true", help="leave the test document open")
    args = ap.parse_args()

    ex = Exchange(Path(args.root))
    ex.ensure()
    tok = ex.token_file
    results = []
    results.append(check("token file is private", tok.exists() and (tok.stat().st_mode & 0o077) == 0,
                         f"{tok}"))

    test_doc = ex.inbox / "fuk_selfcheck.ora"
    write_test_ora(test_doc)
    try:
        with PhotoCraftClient(args.port, tok) as pc:
            before = pc.execute("session.inspect")
            pc.call("app.open", path=ex.rel(test_doc))
            session = pc.execute("session.inspect")
            index = session["active"]
            pc.execute("select.rect", x=100, y=100, width=500, height=400)
            # the selection is itself a history step, so snapshot the revision after it
            session = pc.execute("session.inspect")
            doc = next(d for d in session["documents"] if d["index"] == index)

            export = pb.export_document(pc, ex, index, selection_mask=True)
            ora = export.ora
            names = [l.name for l in ora.layers]
            results.append(check("ORA lists layers top-first",
                                 names[-4:] == ["_mask", "@a red car", "@a dog", "plate"] and names[0] == pb.SELECTION_LAYER,
                                 str(names)))
            car = ora.find("@a red car")
            results.append(check("layers carry their offsets", car is not None and (car.x, car.y) == (500, 700)))
            results.append(check("utility layers are left out of the composite",
                                 tuple(ora.composite[800, 600][:3]) != (255, 0, 0),
                                 f"pixel under the car: {tuple(ora.composite[800, 600])}"))
            sel = pb.mask_from_layer(ora.find(pb.SELECTION_LAYER).canvas(W, H)) if ora.find(pb.SELECTION_LAYER) else None
            if sel is not None:
                ys, xs = np.nonzero(sel > 127)
                bbox = (xs.min(), ys.min(), xs.max(), ys.max())
            results.append(check("selection travels as a layer", sel is not None and bbox == (100, 100, 599, 499),
                                 f"bbox {bbox if sel is not None else None}"))
            after = pc.execute("session.inspect")
            ours = next(d for d in after["documents"] if d["index"] == index)
            results.append(check("export leaves the document untouched",
                                 ours["revision"] == doc["revision"] and ours["path"] == doc["path"]
                                 and len(after["documents"]) == len(session["documents"])))

            result = np.zeros((1376, 2064, 3), np.uint8)   # a 2048-pull rounded to 16s
            result[..., 1] = 200
            placed = pb.place_array(pc, ex, result, "FUK selfcheck", index)
            rows = pc.execute("layer.list", document=index)["layers"]
            top = rows[0]
            results.append(check("place adds a named, canvas-aligned layer on top",
                                 top["name"] == "FUK selfcheck" and top["bounds"] == [0, 0, W, H] and placed["resized"],
                                 str(top)))
            final = pc.execute("session.inspect")
            results.append(check("staging documents are closed",
                                 len(final["documents"]) == len(session["documents"])))
            if not args.keep:
                pc.execute("file.close", document=index)
                results.append(check("test document closed",
                                     len(pc.execute("session.inspect")["documents"]) == len(before["documents"])))
    except PhotoCraftError as e:
        print(f"  FAIL  [{e.kind}] {e}")
        return 1

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
