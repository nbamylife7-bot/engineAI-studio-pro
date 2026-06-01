# Generation Parameters

In the demo UI, command-line tool (`kimodo_gen` / `python -m kimodo.scripts.generate`), and low-level Python API, Kimodo allows some advanced configuration for motion generation.

## Classifier-Free Guidance

Control the strength of text and constraint guidance:

```python
output = model(
    prompt="A person jumps",
    num_frames=150,
    cfg_weight=[2.0, 2.0],  # [text_weight, constraint_weight]
    cfg_type="separated",  # Options: "nocfg", "regular", "separated"
    num_denoising_steps=100,
)
```

These are helpful when there is a tradeoff between following the prompt and hitting constraints.

The CFG options are:
- `cfg_type="nocfg"`: No guidance (faster, less controllable)
- `cfg_type="regular"`: "Standard" classifier-free guidance
    - Equation: `out_uncond + w * (out_text_and_constraint - out_uncond)`
- `cfg_type="separated"`: Separate weights for text and constraints
    - Equation: `out_uncond + w_text * (out_text - out_uncond) + w_constraint * (out_constraint - out_uncond)`

### CLI

The same options are available from the command line as `--cfg_type` and `--cfg_weight`. See the {ref}`CLI user guide (CFG) <classifier-free-guidance-cfg>` for examples, validation rules, and how `meta.json` interacts with explicit flags when using `--input_folder`.

## Denoising Steps
The number of denoising steps used in DDIM sampling can be used to control the speed vs. quality trade-off:
- Fewer steps (50-100): Faster inference, slightly lower quality
- More steps (100-200): Higher quality, slower inference
