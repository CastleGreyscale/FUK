# FUK on RunPod

Runs the whole of FUK, backend and UI, on a rented GPU. You drive it from your
own browser over an SSH tunnel. Use it for the video work a 24GB card is too
slow for: LTX-2.5, Wan, MiniMax, SDR→HDR and SeedVR2. Image generation stays local.

```
your machine                               RunPod pod
────────────                               ──────────
browser → localhost:3001 ── ssh tunnel ──→ vite :3000 → FUK backend :8000
fukpod push / pull ──────── rsync/ssh ───→ /workspace/fuk  (volume)
```

Nothing is exposed through RunPod's HTTP proxy. FUK has no login, and anyone
who has a proxy URL can reach it. That matters for client footage.

## Files

| File | Runs where | Does |
|---|---|---|
| `Dockerfile` | build | Ubuntu 24.04 + `setup.sh` (pinned vendors, patches, torch, npm) |
| `build.sh` | your machine | Builds the image from git-visible files only; tags `<sha>` / `<sha>-dirty` |
| `entrypoint.sh` | pod | Lays out the volume, links FUK onto it, starts sshd and FUK |
| `services.sh` | pod | `start` / `stop` / `restart` / `status` for backend + frontend |
| `pod_paths.py` | pod | Rewrites your home paths in pushed configs to pod paths |
| `prefetch.sh` | pod | Starts downloads through FUK's Models-tab API |
| `fukpod` | your machine | connect, tunnel, push/pull, logs, restart, prefetch |

## One-time setup

1. **Build the image**

   ```bash
   FUK_IMAGE=ghcr.io/castlegreyscale/fuk-runpod deploy/runpod/build.sh
   ```

   The image is built for torch `cu132`, which needs a host driver that
   supports CUDA 13.2. If too few RunPod hosts have that, rebuild with
   `FUK_TORCH_INDEX=cu126`.

2. **Push it to a private registry.** For example GHCR:
   `docker login ghcr.io`, then `docker push`. The image contains FUK's
   source, so keep it private, and add the registry login under RunPod →
   Settings → Container Registry Credentials.

3. **Add your SSH public key** under RunPod → Settings → SSH Public Keys.
   RunPod passes it to the pod as `PUBLIC_KEY`.

4. **Add a Hugging Face token** as a RunPod secret named `hf_token`. Use a
   read-only token from an account that has accepted the licences of any gated
   repos.

5. **Create a network volume**, about 150GB, in a datacenter that has the GPUs
   you want. Weights, configs and projects live on it between pods.

6. **Create a pod template**

   | Setting | Value |
   |---|---|
   | Image | `ghcr.io/castlegreyscale/fuk-runpod:latest` |
   | Container disk | 40 GB *(the image unpacks to ~21GB)* |
   | Volume mount path | `/workspace` |
   | Expose TCP ports | `22` |
   | Expose HTTP ports | *(none)* |
   | Env `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
   | Env `FUK_PREFETCH` | `ltx2 ltx2:loras tool:ltx_hdr` *(optional)* |

7. **Put `fukpod` on your PATH**:
   `ln -s ~/ai/fuk/deploy/runpod/fukpod ~/.local/bin/fukpod`

## Each session

```bash
# Deploy a pod from the template with the volume attached, then copy the
# IP and port from "SSH over exposed TCP" on the pod's Connect panel.
fukpod connect 203.0.113.7 40123

fukpod push-config        # once per volume (also sets up the $HOME mirror), and after config changes
fukpod logs prefetch      # if FUK_PREFETCH is set: watch the first download
fukpod ui                 # → http://localhost:3001

fukpod push ~/Projects/projects/clientX    # inputs up
#   ...open the project in the UI at the same path as at home, and work...
fukpod pull ~/Projects/projects/clientX    # results back, same path
```

The pod mirrors your `$HOME`. `push-config` sends its path, and the pod makes
that path a symlink onto the volume, so `/home/you/...` exists on both machines.
Project files store absolute paths, and this keeps them valid in both
directions without rewriting anything. Only paths under `$HOME` can be pushed
or pulled.

Terminate the pod when you're done. Everything that matters is on the network
volume, which bills monthly whether or not a pod is attached. Nothing on the
pod itself needs keeping.

## What lives where on the pod

```
/workspace/fuk/
  config/    ← fuk/config     your configs, after pod_paths.py
  data/      ← fuk/ui/data    tags, seeds; perf history is the pod's own
  models/    models_root      weights; LoRAs under models/loras
  cache/     ← fuk/cache
  outputs/   ← fuk/outputs
  home/      your $HOME       symlinked from /home/you; projects go here
  hf/        HF_HOME
  logs/      server.log, ui.log, prefetch.log
```

`/opt/fuk` is the image's code. Anything you change there is lost when the pod
goes away.

## Notes

- **`push-config` overwrites the pod's configs** with your local ones. That
  includes the Models tab's enable checkboxes. Your workstation is the source
  of truth.
- **The LTX set is roughly 75–95GB**, depending on which add-on LoRAs you take.
  All of it comes from Lightricks repos on Hugging Face; nothing personal has
  to be uploaded for video work. On a fast host the first download takes
  minutes, and after that it's on the volume.
- **VRAM presets need no change.** `low` sets the limit to total VRAM minus the
  buffer, so on an 80GB card the weights stay resident instead of offloading.
  SDR→HDR's `auto` ladder tries resident FP8 first.
- **Changing FUK code** means rebuilding and pushing the image, then deploying a
  new pod. A `-dirty` tag means uncommitted changes went into it.
- **Outputs in the default cache** (`fuk/cache`, with no project open) land in
  `/workspace/fuk/cache`, not under `$HOME`. Open a project before generating
  anything you want `fukpod pull` to bring back.
