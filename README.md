# ViewSlice Progressive CleanRoom — Modly Process Extension

**Experimental alpha M0.2.1**. Designed to run as a mesh-to-mesh process extension inside Modly, with local/offline-first static mesh refinement.

## Install (GitHub)

1. Upload **all files from this ZIP** to the root of your public GitHub repository. **Do not upload this ZIP as a single file**; extract it first. `manifest.json`, `generator.py` and `setup.py` must be at the top level.
2. Commit the files to `main`.
3. In Modly, find its **Install from GitHub** flow (if supported by your installed Modly version) and enter: `https://github.com/enterryena/viewslice-progressive-cleanroom`.
4. Add the process node named **Adaptive Progressive / Rig Planning**, connect a 3D mesh input, and start with `Operation = progressive`, `Iterations = 1`, `Priority region = Full model`.

## Functionality and safety

The alpha operator increases polygon density in an existing static mesh; it does **not** reconstruct an entire character from images or automatically improve facial detail. The safe GLB path preserves supported vertex attributes (UV, normal, custom IDs) and materials while refining geometry; unsupported glTF attributes/extensions may be rejected to avoid silent losses. The separate CLI and Windows tools are for development and verification.

**Windows + Modly end-to-end installation has not yet been tested.** Preview and rigged/animated assets remain experimental. Always preserve original models. The repository does not bundle MakeHuman meshes, model weights, or copied Modly application code.

## Included files

`manifest.json` is the Modly extension manifest. `generator.py` is the process node adapter, `setup.py` provisions an isolated Python environment, `gltf_raw_editor.py` processes static GLB, and the other Python scripts implement supporting mesh/rig/lineage functions.

## Version note

M0.2.1 is a bug-fix iteration of the earlier v0.7 engine, addressing GLB vertex data loss and NumPy 2.x legacy ID migration. It is not the complete M1/M2 character pipeline.
