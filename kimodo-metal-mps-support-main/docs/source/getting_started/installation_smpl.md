# Using Kimodo-SMPLX Model

Using the [Kimodo-SMPLX-RP-v1](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) model requires a few extra installation steps.

## Request Model Access

The SMPL-X version of Kimodo is gated, so before trying to generate motions with it in the CLI or demo, go to the [Hugging Face model page](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) and request access. As described in the [installation](./installation.md) process, make sure your HF token is properly set up so your access to the model can be authenticated.

## Download SMPL-X Body Model
If you want to visualize generated SMPL-X motions in the demo, you will need to download the SMPL-X body model.
Go to the [SMPL-X](https://smpl-x.is.tue.mpg.de/) webpage and then sign in or create an account and go to the "Download" page.
Click "Download SMPL-X with removed head bun (NPZ)" and then copy the `SMPLX_NEUTRAL.npz` file to the Kimodo codebase to be at `kimodo/kimodo/assets/skeletons/smplx22/SMPLX_NEUTRAL.npz`.

Note that if you installed Kimodo as a package without downloading the codebase, you'll need to find where the assets directory is located by running:
```bash
python -c "from kimodo.assets import skeleton_asset_path; print(skeleton_asset_path('smplx22'))"
```
