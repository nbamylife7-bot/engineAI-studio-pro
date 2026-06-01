# Output Formats

CLI generation uses a single **output stem** (`--output`) for all formats (NPZ, AMASS NPZ, CSV, and BVH). It can write either **one file** or **a folder of files**, depending on the number of samples:

- **One sample** (`--num_samples 1`): writes a single file per format at the stem (e.g. `--output test` → `test.npz`, `test.csv`). No folder is created. For SMPLX, AMASS is written to `test_amass.npz`.
- **Multiple samples**: creates a folder with that stem and writes one file per sample with suffixes `_00`, `_01`, etc. (e.g. `--output test` → `test/test_00.npz`, ...).

Use the `--bvh` flag to also export BVH (SOMA only) to the same stem.

## Kimodo NPZ Format

Generated motions are stored as NPZ files (one file per sample, e.g. `motion_00.npz`) containing:

- `posed_joints`: Global joint positions `[T, J, 3]`
- `global_rot_mats`: Global joint rotation matrices `[T, J, 3, 3]`
- `local_rot_mats`: Local (parent-relative) joint rotation matrices `[T, J, 3, 3]`
- `foot_contacts`: Foot contact labels [left heel, left toe, right heel, right toes] `[T, 4]`
- `smooth_root_pos`: Smoothed root representations outputted from the model `[T, 3]`
- `root_positions`: The (non-smoothed) trajectory of the actual root joint (e.g., pelvis) `[T, 3]`
- `global_root_heading`: The heading direction output from the model `[T, 2]`

For SOMA models, the exported NPZ uses the full **`somaskel77`** skeleton even though the model itself operates internally on the reduced **`somaskel30`** skeleton. This means the saved `posed_joints`, `global_rot_mats`, and `local_rot_mats` arrays are written in the 77-joint SOMA layout. Older 30-joint SOMA NPZ files may still exist and remain loadable for backward compatibility.

Where:

- `T`: number of frames
- `J`: number of joints in the exported skeleton representation (`77` for SOMA NPZ exports, `34` for G1, `22` for SMPL-X)

If multiple samples are generated, files are saved with suffixes like `_00`, `_01`, etc.

## BVH Format for Kimodo-SOMA

When using a SOMA model and passing the `--bvh` flag to CLI generation, Kimodo also writes a BVH file alongside the NPZ output.

- BVH export is supported for **SOMA models only**
- the exported hierarchy uses the full **`somaskel77`** skeleton
- if the motion is still in internal `somaskel30` form, Kimodo converts it to `somaskel77` before writing the BVH
- the file stores root translation plus per-joint local rotations for the clip at the generated frame rate

The exporter writes a standard plain-text BVH file and scales joint offsets and root motion from meters to centimeters (same format as the SEED dataset release). If multiple samples are generated, files are saved with suffixes like `_00`, `_01`, etc.

## CSV Format for Kimodo-G1

When using `Kimodo-G1` models and providing `--output` to CLI generation, the exporter writes MuJoCo `qpos`
data to a CSV file. Each row corresponds to a pose in the motion and contains 36 values:

- Root translation `[x, y, z]`
- Root rotation quaternion `[w, x, y, z]`
- 29 joint 1-DoF values (in G1 joint order)

The CSV uses the MuJoCo coordinate system (z-up, +x forward). If multiple samples are generated, files are saved with suffixes like `_00`, `_01`, etc.


## AMASS NPZ Format for Kimodo-SMPLX

When using the `Kimodo-SMPLX-RP` model and `--output` is specified to CLI generation, the exporter writes an
AMASS-style SMPL-X `.npz` file. Keys include:

- `trans`: Root translation `[T, 3]`
- `root_orient`: Root orientation axis-angle `[T, 3]`
- `pose_body`: Body pose axis-angle `[T, 63]` (21 joints x 3)
- `pose_hand`: Hand pose axis-angle `[T, 90]` (15 joints x 2 hands x 3)
- `pose_jaw`: Jaw pose axis-angle `[T, 3]`
- `pose_eye`: Eye pose axis-angle `[T, 6]`
- `betas`: Shape coefficients
- `num_betas`: Number of shape coefficients
- `gender`: `neutral`
- `surface_model_type`: `smplx`
- `mocap_frame_rate`: Frame rate (fps)
- `mocap_time_length`: Motion duration in seconds

The exporter converts from the Kimodo coordinate system (y-up, +z forward)
to AMASS coordinates (z-up, +y forward). If multiple samples are generated, files are saved with suffixes like `_00`, `_01`, etc.
