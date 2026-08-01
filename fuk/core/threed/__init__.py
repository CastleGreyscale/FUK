"""
FUK 3D reconstruction backends.

Two model tiers, both non-DiffSynth — they load from their own vendored repos
rather than through the pipeline hub:

    vggt_backend    multi-view → point cloud + mesh, runs in the main venv
    trellis_worker  single image → mesh, runs in an isolated venv (subprocess)
    mesh_export     shared GLB / PLY / OBJ writers

See docs/3D_RECONSTRUCTION_SYSTEM.md
"""
