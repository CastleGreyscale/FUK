#!/bin/bash
# Start model downloads through FUK's own Models-tab API, then follow them.
#
#   prefetch.sh ltx2 ltx2:loras tool:ltx_hdr
#
#   <model_key>         a models.json entry (POST /api/models/manage/<key>/download)
#   <model_key>:loras   its add-on LoRAs   (POST .../<key>/loras/download)
#   tool:<key>          a tool's weights   (POST .../tools/<key>/download)
#
# Going through the server rather than download_models.py means the same
# component patterns, HF/ModelScope fallback and LoRA placement as the UI, and
# the jobs show up in the Models panel while they run.
set -uo pipefail

API=http://localhost:8000/api/models/manage

for _ in $(seq 1 120); do
    curl -s -m 2 localhost:8000/health > /dev/null && break
    sleep 2
done
if ! curl -s -m 2 localhost:8000/health > /dev/null; then
    echo "[prefetch] server never came up — nothing started"
    exit 1
fi

for item in "$@"; do
    case "$item" in
        tool:*)  url="$API/tools/${item#tool:}/download" ;;
        *:loras) url="$API/${item%:loras}/loras/download" ;;
        *)       url="$API/$item/download" ;;
    esac
    # The download routes are queued as background threads; starting them all
    # at once lets small LoRA jobs finish while the base model is still coming.
    resp="$(curl -s -m 30 -X POST -H 'Content-Type: application/json' -d '{}' "$url")"
    echo "[prefetch] $item → $resp"
done

echo "[prefetch] following jobs (GET $API/jobs)"
while true; do
    # One line per job: model status completed/total [current file]. Exit code
    # says whether any job is still running.
    if curl -s -m 10 "$API/jobs" | python3 -c '
import json, sys, time
jobs = json.load(sys.stdin).get("jobs", [])
stamp = time.strftime("%H:%M:%S")
for j in jobs:
    extra = j.get("current") or j.get("error") or ", ".join(map(str, j.get("failed") or []))
    print(f"[prefetch] {stamp} {j[\"model\"]}: {j[\"status\"]} {j[\"completed\"]}/{j[\"total\"]} {extra or \"\"}")
sys.exit(1 if any(j["status"] == "running" for j in jobs) else 0)
'; then
        echo "[prefetch] all jobs finished"
        break
    fi
    sleep 30
done
