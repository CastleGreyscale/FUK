"""
FukClient — minimal HTTP client for the FUK server.

Uses only Python's stdlib (urllib) because Blender's bundled Python does not
ship `requests`. Blender and FUK run on the same machine, so control images are
exchanged by *path* (the addon writes renders to disk and passes absolute paths
that the server reads directly) — no multipart upload is needed.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
import urllib.parse


class FukError(Exception):
    """Raised for any FUK server / transport error, with a human message."""


class FukClient:
    def __init__(self, base_url: str):
        self.base = (base_url or "http://localhost:8000").rstrip("/")

    # -- low level -----------------------------------------------------------
    def _request(self, method: str, path: str, data=None, timeout: float = 120.0):
        url = path if path.startswith("http") else f"{self.base}/{path.lstrip('/')}"
        body = None
        headers = {"Accept": "application/json"}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")
                parsed = json.loads(detail)
                detail = parsed.get("detail", detail)
            except Exception:
                pass
            raise FukError(f"HTTP {e.code} on {url}: {detail or e.reason}")
        except urllib.error.URLError as e:
            raise FukError(f"Cannot reach FUK server at {url} — {e.reason}. Is it running?")
        except (TimeoutError, OSError) as e:
            raise FukError(f"Request to {url} failed: {e}")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw.decode("utf-8", "replace")}

    def get(self, path: str, timeout: float = 120.0):
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, data=None, timeout: float = 120.0):
        return self._request("POST", path, data=data, timeout=timeout)

    def download(self, url_or_path: str, dest: str) -> str:
        """Download a server file (URL or `api/...` relative path) to `dest`."""
        url = url_or_path if url_or_path.startswith("http") else f"{self.base}/{url_or_path.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=120.0) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
        except (urllib.error.URLError, OSError) as e:
            raise FukError(f"Failed to download {url}: {e}")
        return dest

    # -- convenience ---------------------------------------------------------
    def health(self):
        return self.get("/health", timeout=10.0)

    def set_project_folder(self, folder: str):
        return self.post("/api/project/set-folder", {"path": folder})

    def list_shots(self):
        return self.get("/api/project/list")

    def load_shot(self, filename: str):
        return self.get(f"/api/project/load/{urllib.parse.quote(filename)}")

    def save_shot(self, filename: str, state: dict):
        return self.post(f"/api/project/save/{urllib.parse.quote(filename)}", state)

    def preprocess(self, payload: dict, timeout: float = 600.0):
        return self.post("/api/preprocess", payload, timeout=timeout)

    def prompt_tokens(self, model: str | None = None, active_loras: str | None = None):
        """List available prompt tokens (#markers) for tag autocomplete."""
        query = {}
        if model:
            query["model"] = model
        if active_loras:
            query["active_loras"] = active_loras
        path = "/api/prompt/tokens"
        if query:
            path += "?" + urllib.parse.urlencode(query)
        return self.get(path, timeout=30.0)

    def prompt_resolve(self, text: str, model: str | None = None, apply_mood: bool = True):
        """Resolve #markers (+ storyboard mood) — the same expansion used at gen time."""
        return self.post("/api/prompt/resolve", {
            "text": text,
            "model": model,
            "apply_mood": apply_mood,
        }, timeout=30.0)

    def generate_image(self, payload: dict):
        return self.post("/api/generate/image", payload)

    def status(self, generation_id: str):
        return self.get(f"/api/status/{generation_id}", timeout=30.0)

    def cancel(self, generation_id: str):
        return self.post(f"/api/cancel/{generation_id}", {})
