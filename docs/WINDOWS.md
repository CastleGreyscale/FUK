# FUK on Windows — WSL2 Setup Guide

FUK runs on Windows via **WSL2** (Windows Subsystem for Linux). You get a real Linux kernel, native CUDA access, and a proper bash environment — `setup.sh` runs without modification.

> **WSL2 basics not covered here.** If you haven't installed WSL2 yet, Microsoft's official two-step guide covers it:
> [https://documentation.ubuntu.com/](https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/)
> Once you have a working Ubuntu terminal, come back here.

---

## The I/O problem you need to know about

WSL2 stores its Linux filesystem inside a virtual disk (`.vhdx` file) that lives on your **C: drive by default**. This works, but creates two performance issues for FUK specifically:

1. **C: drive space** — FUK models run 15–30 GB each. A modest working set (3–4 models) is 60–100 GB before outputs. Most people's C: drives can't absorb that comfortably.

2. **Cross-filesystem I/O penalty** — if your models are on a Windows drive (say `D:\ai\models`) and you access them from inside WSL via `/mnt/d/ai/models`, every file read crosses the WSL translation layer. For normal files this is unnoticeable. For loading 14B model shards at startup it is very noticeable — expect 2–4x slower cold load times compared to native.

The fix is to move WSL itself onto your larger drive so `~/` is physically on D: (or wherever your space is) at full native speed.

---

## Relocating WSL to another drive

Run the following in **PowerShell as Administrator**. Replace `D:\WSL\Ubuntu` with whatever path makes sense for your setup.

```powershell
# 1. Shut down WSL cleanly
wsl --shutdown

# 2. Export your current Ubuntu installation to a backup tarball
#    (this can take a few minutes depending on what you have installed)
wsl --export Ubuntu C:\Temp\ubuntu-backup.tar

# 3. Remove the current registration (does NOT delete the tarball you just made)
wsl --unregister Ubuntu

# 4. Re-import from the tarball into your preferred location
wsl --import Ubuntu D:\WSL\Ubuntu C:\Temp\ubuntu-backup.tar

# 5. Restore your default user (replace 'brad' with your WSL username)
ubuntu config --default-user [USERNAME]
```

After this, launch Ubuntu normally. Your home directory (`~/`) is now physically on D: and accessed at full speed. You can delete `C:\Temp\ubuntu-backup.tar` once you've confirmed everything works.

---

## Recommended FUK layout after relocation

With WSL on your large drive, keep everything inside the WSL filesystem:

```
~/
├── Projects/FUK/          ← git repo, setup.sh, venv
└── ai/
    └── models/            ← models_root in defaults.json
        ├── Qwen/
        ├── Wan-AI/
        └── loras/
```

Set `models_root` in `fuk/config/defaults.json` to an absolute Linux path:

```json
"models_root": "/home/brad/ai/models"
```

**Avoid `/mnt/c/` or `/mnt/d/` paths for models.** Those paths work but you'll pay the cross-filesystem penalty every time the pipeline loads a model.

---

## VRAM note for laptops

On systems with integrated + discrete GPU, make sure WSL2 is using the discrete card. Check with:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If it shows the wrong GPU, set the preferred GPU for WSL in Windows **Graphics Settings → Hardware-accelerated GPU scheduling**.

---

## Everything else

Once WSL is set up and relocated, the standard install flow applies:

```bash
cd ~/Projects/FUK
bash setup.sh
```

No path translation, no Windows-specific flags, no changes to the codebase needed.
