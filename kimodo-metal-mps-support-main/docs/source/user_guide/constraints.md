# Constraints JSON Format

The `--constraints` flag in the CLI expects a JSON file containing a list of constraint objects.
It is easiest to look at the examples provided with the demo to see how thse are formatted. These can be seen for various model types in `kimodo/assets/demo/examples`.

> Tip: the easiest way to get a valid constraints file is to create constraints in the interactive demo and to click on `Save Constraints`.

## High-Level Structure

- The file is a JSON array: `[{...}, {...}, ...]`
- Each element is an object with at least:
  - `type` (string)
    - `root2d`, `fullbody`, `left-hand`, `right-hand`, `left-foot`, `right-foot`, `end-effector`
  - `frame_indices` (array of integers): 0-based frame indices within the generated clip.


```{note}
For SOMA models, constraints may be authored or displayed on the full `somaskel77` skeleton, but Kimodo converts them to the reduced `somaskel30` representation before passing them to the model. See the [skeleton](../key_concepts/skeleton.md) section for more details.
```

## Constraint Types
Depending on `type`, additional fields are required or optional. All numeric arrays are plain nested JSON lists. In the following definitions `T` is the number of frames and `J` is the number of skeleton joints.


### `root2d`
This captures 2D root waypoints and 2D root paths. It requires:

- `smooth_root_2d` (array shapes `[T, 2]`): Smoothed root positions `[x, z]` on the ground plane at the given `frame_indices`.

and optionally:
- `global_root_heading` (array shapes `[T, 2]`): Global root heading direction `[cos, sin]` at the given `frame_indices`.

### `fullbody`
This captures full-body keyframe constraints on joint positions. It includes:

- `local_joints_rot` (array shaped `[T, J, 3]`): Per-frame per-joint **axis-angle** local rotations (radians). Constraint joint positions will be derived from these.
- `root_positions` (array shaped `[T, 3]`): Root (hips) translation `[x, y, z]`.
- `smooth_root_2d` (optional; array of `[T, 2]`): Smoothed root positions `[x, z]`. If omitted, it is taken as the `[x, z]` components of `root_positions`.

### `left-hand` / `right-hand` / `left-foot` / `right-foot`
Captures end-effector constraints on the hand/feet joint positions and rotations.

These use the same fields as `fullbody`. However, under the hood these will only affect the corresponding end-effectors and hips.

## Minimal Example

```json
[
  {
    "type": "root2d",
    "frame_indices": [0, 30, 60],
    "smooth_root_2d": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.1]]
  }
]
```
