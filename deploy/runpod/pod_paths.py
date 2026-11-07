"""
Rewrite FUK config paths for the pod. Run at every boot and after a config push.

Configs pushed from a workstation carry that machine's absolute paths —
models_root, the LoRA roots, vendor paths in tools/*.json. Rather than know
every key that holds a path, this re-roots any string that starts with a known
home root:

    home models_root  ->  <workspace>/models
    home repo root    ->  /opt/fuk

The home roots come from .home_paths.json, which `fukpod push-config` writes
next to the configs. Without it (a volume seeded from the image's templates)
models_root is still forced to <workspace>/models, so the pod always runs.

Idempotent: a rewritten config contains no home roots, so a second run changes
nothing. Stdlib only — it runs before anything else is known to work.

    python3 pod_paths.py /workspace/fuk
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

POD_REPO = "/opt/fuk"


def _reroot(value, roots):
    if isinstance(value, str):
        for old, new in roots:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value
    if isinstance(value, list):
        return [_reroot(v, roots) for v in value]
    if isinstance(value, dict):
        # "_..." keys are prose comments; a path quoted in one is not a setting.
        return {k: (v if k.startswith("_") else _reroot(v, roots)) for k, v in value.items()}
    return value


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(workspace: str) -> int:
    ws = Path(workspace)
    config = ws / "config"
    models = str(ws / "models")

    home = {}
    home_file = config / ".home_paths.json"
    if home_file.exists():
        home = json.loads(home_file.read_text())

    roots = []
    if home.get("models_root"):
        roots.append((home["models_root"].rstrip("/"), models))
    if home.get("repo_root"):
        roots.append((home["repo_root"].rstrip("/"), POD_REPO))
    # Longest first, so a root nested inside another is matched before its parent.
    roots.sort(key=lambda r: len(r[0]), reverse=True)

    changed = []
    for path in sorted(config.rglob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[pod_paths] skipped {path.name}: {exc}")
            continue

        new = _reroot(data, roots)
        if path.name == "defaults.json" and isinstance(new, dict):
            new["models_root"] = models
        # Compare data, not text: a file with no paths in it keeps its own
        # formatting instead of being re-serialised.
        if new != data:
            _write(path, new)
            changed.append(str(path.relative_to(config)))

    print(f"[pod_paths] models_root = {models}")
    print(f"[pod_paths] rewritten: {', '.join(changed) if changed else 'nothing to change'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/fuk"))
