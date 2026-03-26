"""
Latent Layer Stack

Non-destructive image editing via latent delta composition.

Each edit is stored as a delta: (edit_latent - base_latent).
Flattening sums the base + all enabled deltas, then VAE-decodes to a preview.

This means toggling, reordering, or re-running any layer never touches
the others — it's purely additive tensor math until the final decode.

Manifest lives on disk as stack.json alongside the .pt files so sessions
survive server restarts and can be reloaded at any time.
"""

import json
import uuid
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat()


def _log(msg: str, level: str = "info"):
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%H:%M:%S.%f")[:-3]
    colors = {
        "info":    "\033[96m",
        "success": "\033[92m",
        "warning": "\033[93m",
        "error":   "\033[91m",
    }
    print(f"{colors.get(level, colors['info'])}[{ts}] [LAYERS] {msg}\033[0m", flush=True)


# ---------------------------------------------------------------------------
# LatentLayerStack
# ---------------------------------------------------------------------------

class LatentLayerStack:
    """
    Manages a stack of latent edit layers for a single base image.

    Directory layout (all under stack_dir/):
        stack.json          — manifest (source of truth)
        base.pt             — base image latent
        base_preview.png    — decoded base preview
        layer_{id}/
            v{n}.pt         — edit latent for version n
            v{n}_delta.pt   — precomputed delta (edit - base)

    Usage:
        stack = LatentLayerStack.create(stack_dir, base_latent_tensor, base_image_path)
        stack.add_layer("orange cat", latent_tensor, params)
        stack.add_layer("lamp warmer", latent_tensor, params)
        flat_array = stack.flatten(vae)           # → float32 [H,W,C] numpy
        stack.set_enabled("layer_id", False)
        flat_array = stack.flatten(vae)           # lamp excluded
    """

    MANIFEST_FILE = "stack.json"

    def __init__(self, stack_dir: Path):
        self.stack_dir = Path(stack_dir)
        self._manifest: dict = {}

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        stack_dir: Path,
        base_latent: torch.Tensor,
        base_image_path: Optional[Path] = None,
    ) -> "LatentLayerStack":
        """
        Create a new stack from a base latent tensor.

        Args:
            stack_dir:       Where to store all stack files.
            base_latent:     The latent of the original (unedited) image.
            base_image_path: Optional path to the source PNG for display.
        """
        stack_dir = Path(stack_dir)
        stack_dir.mkdir(parents=True, exist_ok=True)

        # Save base latent
        base_pt = stack_dir / "base.pt"
        torch.save(base_latent.cpu(), base_pt)

        manifest = {
            "stack_id":        str(uuid.uuid4()),
            "created_at":      _now(),
            "updated_at":      _now(),
            "base_latent_path": str(base_pt),
            "base_image_path": str(base_image_path) if base_image_path else None,
            "layers":          [],
        }

        instance = cls(stack_dir)
        instance._manifest = manifest
        instance._save_manifest()

        _log(f"Created stack: {stack_dir.name}", "success")
        return instance

    @classmethod
    def load(cls, stack_dir: Path) -> "LatentLayerStack":
        """Load an existing stack from disk."""
        stack_dir = Path(stack_dir)
        manifest_path = stack_dir / cls.MANIFEST_FILE

        if not manifest_path.exists():
            raise FileNotFoundError(f"No stack manifest at: {manifest_path}")

        instance = cls(stack_dir)
        with open(manifest_path) as f:
            instance._manifest = json.load(f)

        _log(f"Loaded stack: {stack_dir.name} ({len(instance._manifest['layers'])} layers)")
        return instance

    # ------------------------------------------------------------------
    # Manifest persistence
    # ------------------------------------------------------------------

    def _save_manifest(self):
        self._manifest["updated_at"] = _now()
        path = self.stack_dir / self.MANIFEST_FILE
        with open(path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def to_dict(self) -> dict:
        """Return manifest as a JSON-serialisable dict (for API responses)."""
        return dict(self._manifest)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_layer(
        self,
        name: str,
        edit_latent: torch.Tensor,
        params: dict,
    ) -> str:
        """
        Add a new layer (with its first version) to the stack.

        Args:
            name:         Human-readable label (e.g. "orange cat").
            edit_latent:  Full edit latent from the generation.
            params:       Generation params to store (prompt, seed, model, …).

        Returns:
            layer_id of the new layer.
        """
        layer_id = str(uuid.uuid4())[:8]
        layer_dir = self.stack_dir / f"layer_{layer_id}"
        layer_dir.mkdir(exist_ok=True)

        delta = self._compute_delta(edit_latent)
        v_index = 0

        # Save edit latent and delta
        latent_pt = layer_dir / f"v{v_index}.pt"
        delta_pt  = layer_dir / f"v{v_index}_delta.pt"
        torch.save(edit_latent.cpu(), latent_pt)
        torch.save(delta.cpu(),       delta_pt)

        version = {
            "v":            v_index,
            "latent_path":  str(latent_pt),
            "delta_path":   str(delta_pt),
            "prompt":       params.get("prompt", ""),
            "seed":         params.get("seed"),
            "model":        params.get("model", ""),
            "steps":        params.get("steps"),
            "cfg_scale":    params.get("cfg_scale"),
            "denoising_strength": params.get("denoising_strength"),
            "lora":         params.get("lora"),
            "timestamp":    _now(),
        }

        layer = {
            "layer_id":       layer_id,
            "name":           name,
            "enabled":        True,
            "active_version": v_index,
            "versions":       [version],
        }

        self._manifest["layers"].append(layer)
        self._save_manifest()

        _log(f"Added layer '{name}' (id={layer_id})", "success")
        return layer_id

    def add_version(
        self,
        layer_id: str,
        edit_latent: torch.Tensor,
        params: dict,
    ) -> int:
        """
        Add a new version to an existing layer (re-run with different params).

        Returns the new version index.
        """
        layer = self._get_layer(layer_id)
        v_index = len(layer["versions"])

        layer_dir = self.stack_dir / f"layer_{layer_id}"
        layer_dir.mkdir(exist_ok=True)

        delta = self._compute_delta(edit_latent)

        latent_pt = layer_dir / f"v{v_index}.pt"
        delta_pt  = layer_dir / f"v{v_index}_delta.pt"
        torch.save(edit_latent.cpu(), latent_pt)
        torch.save(delta.cpu(),       delta_pt)

        version = {
            "v":            v_index,
            "latent_path":  str(latent_pt),
            "delta_path":   str(delta_pt),
            "prompt":       params.get("prompt", ""),
            "seed":         params.get("seed"),
            "model":        params.get("model", ""),
            "steps":        params.get("steps"),
            "cfg_scale":    params.get("cfg_scale"),
            "denoising_strength": params.get("denoising_strength"),
            "lora":         params.get("lora"),
            "timestamp":    _now(),
        }

        layer["versions"].append(version)
        layer["active_version"] = v_index   # Auto-activate new version
        self._save_manifest()

        _log(f"Added v{v_index} to layer '{layer['name']}'", "success")
        return v_index

    def set_enabled(self, layer_id: str, enabled: bool):
        """Toggle a layer on or off."""
        layer = self._get_layer(layer_id)
        layer["enabled"] = enabled
        self._save_manifest()
        _log(f"Layer '{layer['name']}' enabled={enabled}")

    def set_active_version(self, layer_id: str, v_index: int):
        """Switch a layer to a different version."""
        layer = self._get_layer(layer_id)
        if v_index < 0 or v_index >= len(layer["versions"]):
            raise ValueError(f"Version {v_index} out of range for layer {layer_id}")
        layer["active_version"] = v_index
        self._save_manifest()
        _log(f"Layer '{layer['name']}' switched to v{v_index}")

    def reorder_layers(self, layer_ids: list):
        """Reorder layers by providing a new ordered list of layer_ids."""
        id_to_layer = {l["layer_id"]: l for l in self._manifest["layers"]}
        if set(layer_ids) != set(id_to_layer.keys()):
            raise ValueError("layer_ids must contain exactly all existing layer IDs")
        self._manifest["layers"] = [id_to_layer[lid] for lid in layer_ids]
        self._save_manifest()

    def remove_layer(self, layer_id: str):
        """Remove a layer from the stack (files remain on disk)."""
        layers = self._manifest["layers"]
        self._manifest["layers"] = [l for l in layers if l["layer_id"] != layer_id]
        self._save_manifest()
        _log(f"Removed layer {layer_id}")

    # ------------------------------------------------------------------
    # Flatten
    # ------------------------------------------------------------------

    def flatten(self, vae, device: str = "cuda") -> np.ndarray:
        """
        Flatten all enabled layers into a decoded float32 image.

        Algorithm:
            result_latent = base + sum(active_delta for each enabled layer)
            image = VAE.decode(result_latent)

        Args:
            vae:    VAE decoder (borrowed from the cached Qwen pipeline).
            device: Torch device for computation.

        Returns:
            float32 numpy array [H, W, C], scene-linear, range ~[0,1].
        """
        base_latent = self._load_base_latent(device)

        # Accumulate enabled deltas
        result = base_latent.clone()
        active_count = 0

        for layer in self._manifest["layers"]:
            if not layer["enabled"]:
                continue
            v = layer["active_version"]
            version = layer["versions"][v]
            delta_path = Path(version["delta_path"])

            if not delta_path.exists():
                _log(f"Delta missing for layer '{layer['name']}' v{v} — recomputing", "warning")
                latent = torch.load(version["latent_path"], map_location=device)
                delta = self._compute_delta(latent, device=device)
                torch.save(delta.cpu(), delta_path)
            else:
                delta = torch.load(delta_path, map_location=device)

            result = result + delta
            active_count += 1

        _log(f"Flattening {active_count} active layer(s)")

        # VAE decode
        with torch.inference_mode():
            result = result.to(device)
            decoded = vae.decode(result)

        # Normalise to [0, 1] and convert to numpy [H, W, C]
        if isinstance(decoded, torch.Tensor):
            arr = decoded
        else:
            arr = decoded.sample if hasattr(decoded, "sample") else decoded

        arr = arr.squeeze(0).cpu().float()          # [C, H, W]
        arr = (arr + 1.0) / 2.0                     # [-1,1] → [0,1]
        arr = arr.clamp(0.0, 1.0)
        arr = arr.permute(1, 2, 0).numpy()           # [H, W, C]

        _log("Flatten complete", "success")
        return arr.astype(np.float32)

    def save_flatten_preview(self, vae, preview_path: Path, device: str = "cuda") -> Path:
        """
        Flatten and save a PNG preview for display in the UI.

        Returns:
            Path to the saved PNG.
        """
        from PIL import Image

        arr = self.flatten(vae, device=device)
        preview_path = Path(preview_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 [0,1] → uint8 for PNG
        img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(preview_path)

        _log(f"Preview saved: {preview_path.name}", "success")
        return preview_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layer(self, layer_id: str) -> dict:
        for layer in self._manifest["layers"]:
            if layer["layer_id"] == layer_id:
                return layer
        raise KeyError(f"Layer not found: {layer_id}")

    def _load_base_latent(self, device: str = "cuda") -> torch.Tensor:
        base_path = Path(self._manifest["base_latent_path"])
        if not base_path.exists():
            raise FileNotFoundError(f"Base latent missing: {base_path}")
        return torch.load(base_path, map_location=device)

    def _compute_delta(
        self,
        edit_latent: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        delta = edit_latent - base_latent

        If the edit latent has different spatial dimensions (e.g. the edit
        pipeline quantised to a different VAE multiple), we centre-crop or
        zero-pad the edit to match the base before subtracting.

        Both tensors kept in float32 for accurate delta math.
        """
        base = self._load_base_latent(device)
        edit = edit_latent.to(device)

        # Shape: typically [B, C, H, W] for images
        if edit.shape != base.shape:
            _log(f"Shape mismatch: base={list(base.shape)} edit={list(edit.shape)} — aligning edit to base", "warning")
            edit = self._match_spatial(edit, base)

        return (edit.float() - base.float())

    @staticmethod
    def _match_spatial(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Centre-crop or zero-pad *src* so its spatial dims match *target*.

        Works for both [B, C, H, W] (image) and [B, C, F, H, W] (video)
        latents — spatial dims are always the last two.
        """
        # Spatial dims are the last two
        src_h, src_w = src.shape[-2], src.shape[-1]
        tgt_h, tgt_w = target.shape[-2], target.shape[-1]

        if src_h == tgt_h and src_w == tgt_w:
            return src

        result = torch.zeros_like(target)

        # Copy region is the overlap
        copy_h = min(src_h, tgt_h)
        copy_w = min(src_w, tgt_w)

        # Centre offsets
        src_y = (src_h - copy_h) // 2
        src_x = (src_w - copy_w) // 2
        tgt_y = (tgt_h - copy_h) // 2
        tgt_x = (tgt_w - copy_w) // 2

        result[..., tgt_y:tgt_y + copy_h, tgt_x:tgt_x + copy_w] = \
            src[..., src_y:src_y + copy_h, src_x:src_x + copy_w]

        _log(f"Aligned: ({src_h}×{src_w}) → ({tgt_h}×{tgt_w}) via centre crop/pad")
        return result