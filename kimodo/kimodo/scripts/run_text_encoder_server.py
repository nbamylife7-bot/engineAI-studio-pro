# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import gradio as gr
import numpy as np

from kimodo.device_utils import preferred_text_encoder_dtype, resolve_text_encoder_device
from kimodo.model import resolve_target
from kimodo.model.text_encoder_quantization import (
    resolve_effective_quantization,
    resolve_text_encoder_load_device,
)

from .gradio_theme import get_gradio_theme

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"
DEFAULT_TEXT = "A person walks and falls to the ground."
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 9550
DEFAULT_TMP_FOLDER = "/tmp/text_encoder/"
DEFAULT_TEXT_ENCODER = "llm2vec"
TEXT_ENCODER_PRESETS = {
    "llm2vec": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dtype": "bfloat16",
            "llm_dim": 4096,
        },
        "display_name": "LLM2Vec",
    },
    "llm2vec_nf4": {
        "target": "kimodo.model.LLM2VecEncoder",
        "kwargs": {
            "base_model_name_or_path": "matbee/kimodo-llm2vec-nf4",
            "peft_model_name_or_path": "matbee/kimodo-llm2vec-nf4/supervised_adapter",
            "dtype": "bfloat16",
            "llm_dim": 4096,
        },
        "display_name": "LLM2Vec NF4 (matbee)",
    },
}


class DemoWrapper:
    def __init__(self, text_encoder, tmp_folder):
        self.text_encoder = text_encoder
        self.tmp_folder = tmp_folder

    def __call__(self, text, filename, progress=gr.Progress()):
        # Compute text embedding
        tensor, length = self.text_encoder(text)
        if isinstance(length, list):
            length = int(length[0]) if length else 1
        else:
            length = int(length)
        if tensor.dim() == 3:
            embedding = tensor[0, :length]
        elif tensor.dim() == 2:
            embedding = tensor[:length]
        else:
            embedding = tensor.unsqueeze(0)
        embedding = embedding.detach().cpu().numpy()

        # Save text embedding
        path = os.path.join(self.tmp_folder, filename)
        np.save(path, embedding)

        output_title = gr.Markdown(visible=True)
        output_text = gr.Markdown(visible=True, value=f"Text: {text}")
        download = gr.DownloadButton(visible=True, value=path)
        return download, output_title, output_text


def _get_env(name: str, default):
    return os.getenv(name, default)


def _build_text_encoder(name: str):
    if name not in TEXT_ENCODER_PRESETS:
        available = ", ".join(sorted(TEXT_ENCODER_PRESETS))
        raise ValueError(f"Unknown TEXT_ENCODER='{name}'. Available: {available}")
    preset = TEXT_ENCODER_PRESETS[name]
    target_cls = resolve_target(preset["target"])
    device = resolve_text_encoder_device()
    quantization = resolve_effective_quantization()
    load_device = resolve_text_encoder_load_device(device, quantization)
    kwargs = dict(preset["kwargs"])
    kwargs["dtype"] = preferred_text_encoder_dtype(device, override=_get_env("TEXT_ENCODER_DTYPE", None))
    text_encoder = target_cls(**kwargs)
    text_encoder.to(load_device)
    return text_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Run text encoder Gradio server.")
    parser.add_argument(
        "--text-encoder",
        default=_get_env("TEXT_ENCODER", DEFAULT_TEXT_ENCODER),
        choices=sorted(TEXT_ENCODER_PRESETS.keys()),
        help="Text encoder preset.",
    )
    parser.add_argument(
        "--tmp-folder",
        default=_get_env("TEXT_ENCODER_TMP_FOLDER", DEFAULT_TMP_FOLDER),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server_name = _get_env("GRADIO_SERVER_NAME", DEFAULT_SERVER_NAME)
    server_port = int(_get_env("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT))
    theme, css = get_gradio_theme()
    os.makedirs(args.tmp_folder, exist_ok=True)
    text_encoder = _build_text_encoder(args.text_encoder)
    display_name = TEXT_ENCODER_PRESETS[args.text_encoder]["display_name"]
    demo_wrapper_fn = DemoWrapper(text_encoder, args.tmp_folder)

    with gr.Blocks(title="Text encoder", css=css, theme=theme) as demo:
        gr.Markdown(f"# Text encoder: {display_name}")
        gr.Markdown("## Description")
        gr.Markdown("Get a embeddings from a text.")

        gr.Markdown("## Inputs")
        with gr.Row():
            text = gr.Textbox(
                placeholder="Type the motion you want to generate with a sentence",
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
                type="text",
            )
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                btn = gr.Button("Encode", variant="primary")
            with gr.Column(scale=1):
                clear = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=3):
                pass

        output_title = gr.Markdown("## Outputs", visible=False)
        output_text = gr.Markdown("", visible=False)
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                download = gr.DownloadButton("Download", variant="primary", visible=False)
            with gr.Column(scale=4):
                pass

        filename = gr.Textbox(
            visible=False,
            value="embedding.npy",
        )

        def clear_fn():
            return [
                gr.DownloadButton(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
            ]

        outputs = [download, output_title, output_text]

        gr.on(
            triggers=[text.submit, btn.click],
            fn=clear_fn,
            inputs=None,
            outputs=outputs,
        ).then(
            fn=demo_wrapper_fn,
            inputs=[text, filename],
            outputs=outputs,
        )

        def download_file():
            return gr.DownloadButton()

        download.click(
            fn=download_file,
            inputs=None,
            outputs=[download],
        )
        clear.click(fn=clear_fn, inputs=None, outputs=outputs)

    demo.launch(server_name=server_name, server_port=server_port, show_error=True)


if __name__ == "__main__":
    main()
