# PhotoCraft bridge

Moves images between FUK and [PhotoCraft](https://getartcraft.com/apps), ArtCraft's
MIT/Apache image editor:

- **Pull:** a document's composite, one of its layers, a selection or `_mask`, or its
  `@` entity layers, into the Image tab's inputs.
- **Place:** any FUK result or history entry back into the document as a new layer.
- **Depth:** a depth map of the composite, placed back as a layer for slap comps.

PhotoCraft has no in-app plugin API and can't call out to other programs, so the
controls live in the FUK web UI. The FUK server drives PhotoCraft over its
control channel.

| Piece | Where |
|---|---|
| Control-channel client, ORA reader, image helpers | `fuk/ui/photocraft_bridge.py` |
| Routes (`/api/photocraft/*`) | `fuk/ui/photocraft_endpoints.py` |
| Web UI | `ui/src/components/PhotoCraft{Card,Button}.jsx`, `ui/src/utils/photocraftApi.js`, buttons in `ImageTab.jsx` and `GenerationHistory.jsx` |
| Launcher | `launch_photocraft.sh` |
| Self-check against the real app | `pc_probe.py` |
| Tests (fake PhotoCraft) | `fuk/testing/test_photocraft_bridge.py` |

## Setup

PhotoCraft is installed as the Flatpak `ai.storyteller.photocraft`. Start it with
the launcher. You can also use **Launch PhotoCraft** on the Image tab's PhotoCraft
card, which runs the same script.

```sh
fuk/plugins/photocraft/launch_photocraft.sh            # this launch only
fuk/plugins/photocraft/launch_photocraft.sh --persist  # also: flatpak override --share=network
```

The launcher adds three things a plain `flatpak run` doesn't have:

- **`--share=network`.** Without it the sandbox gets its own network namespace, and
  the control port (`127.0.0.1:47823`) is invisible to FUK.
- **`--present-mode mailbox`.** Control requests run on PhotoCraft's frame loop. Under
  the default vsync present mode, a hidden window stops drawing, so requests queue
  up and time out after 60 s ("bring its window into view"). If it happens anyway,
  bring the window into view and retry.
- **The control token and automation roots.** Everything is exchanged through
  `<Pictures>/fuk_photocraft/exchange/`, with the token in `control.token` (0600).
  The sandbox already sees the Pictures folder, at the same absolute path.

Config lives in `defaults.json → tools.photocraft`:

| Key | Meaning |
|---|---|
| `port` | Control port. |
| `root` | Exchange root. Empty means `<Pictures>/fuk_photocraft`. |
| `app_id` | Flatpak app id. |
| `max_pull_side` | Default 2048. Pulls are downscaled to this size so a 6K document doesn't become a 6K generation. Placing a result scales it back up to the document size. |

## Layer conventions

| Layer | Used for |
|---|---|
| `@<prompt>` (e.g. `@a red car`) | One EliGen entity. Its painted pixels (alpha, or luminance if the layer is opaque) are the region. The layer nearest the top of the Layers panel becomes entity 01. Hidden or visible both work. |
| `_mask` | Inpaint mask when there's no selection. White or painted means repaint. |
| a selection | Inpaint mask. A selection wins over `_mask`. |

`@` and `_` layers are always left out of the composite FUK receives. The bridge
hides them in a scratch copy, so your document isn't changed. Layers inside
groups are read too, including `@` layers.

## What each action does

- **From PhotoCraft (control/edit input):** a menu with Composite, Active layer, or any
  layer. The result goes into `control_image_paths`.
- **From PhotoCraft (inpaint):** sets the composite as the inpaint image and the
  selection or `_mask` as its mask, in one step. "Mask from PhotoCraft" fills only the
  mask.
- **@ layers → EliGen:** writes `import_###/eligen/NN_<prompt>.png` and switches the tab
  to the EliGen model with that folder as `eligen_source`. The server sizes the
  generation from the mask folder, a fix added with this bridge.
- **Depth → layer:**
  - Runs Depth Anything 3 (`da3_mono_large`, Apache-2.0) on the composite.
    Non-commercial depth models are refused.
  - Normalises FUK's raw Z buffer itself, min–max with near = white, so the result
    is deterministic and doesn't depend on how the model writes its display PNG.
  - Places the result as `FUK depth preprocess_depth_NNN`.
  - In an 8-bit document the layer has 256 levels. Pick "+ 16-bit doc" or
    "+ 32-bit EXR doc" to also open the full-precision map as its own document.
  - `depth16.png` and `depth.exr` stay in the history entry, which also works as a
    control map.
- **→ PhotoCraft** (result preview, history entries): places the image as
  `FUK img_gen_012 s1234` (generations) or `FUK <entry> <file>` (everything else).
  It is aligned to the canvas and scaled back to the document size when only scale or
  rounding to multiples of 16 differs.

Pulls never start a generation. Each one creates an `import_###` history entry
(a copy, since the exchange folder is reused), and you press Generate.

## Probe notes (PhotoCraft 0.16, desktop control server)

The protocol docs describe the headless `photocraft-cli serve` server too, and
the desktop app supports much less. What the bridge relies on (re-check with
`pc_probe.py` after a PhotoCraft update):

- **Unavailable on the desktop server:**
  - no `doc.*` methods (`doc.inspect`, `doc.place`, `doc.exportLayers`, `doc.render`
    all return "unknown method")
  - engine commands that take a path or load profiles are refused (`file.saveACopy`,
    `layer.exportAs`, `image.mode.*`), so the bridge **can't change a document's bit
    depth**
- **File I/O:** only `app.open {path}` and `app.save {path}`, with paths relative to the
  automation roots. `app.save` **re-targets** the document it saves, so the bridge
  only saves an `image.duplicate` copy.
- **Duplicates keep the selection.** The bridge fills it white on a `_fuk_selection`
  layer in the copy, and that is how the selection reaches FUK.
- **Saved `.ora` files carry everything needed:**
  - layers listed top-first, trimmed, with offsets
  - hidden layers included
  - `mergedimage.png` as the visible composite
- **Bit depth on import:** `layer.copyToDocument {document, source, layers, name}` keeps
  position and bit depth up to the destination's depth. A 16-bit layer copied into an
  8-bit document is quantised.
- **Commands:** `layer.list` is top-down with `bounds: [x, y, w, h]`. The
  `engine.commands` filter parameter is ignored.
- **Closing:** `file.close` closes a modified document without asking.

## Security

The control port is bound to loopback and needs the 256-bit token, which is in a
0600 file. Network access is only shared with PhotoCraft instances the launcher
starts, unless you used `--persist`.
