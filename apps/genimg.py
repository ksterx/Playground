from tkinter import filedialog

import gradio as gr
from torchvision.utils import save_image

from deepx.algos import ImageGen
from deepx.nn import registered_models


def generate(ckpt_path, model_name, tgt_shape, backbone):
    tgt_shape = tuple(map(int, tgt_shape.split("x")))

    model = registered_models[model_name]
    model = model(backbone=backbone, tgt_shape=tgt_shape)
    model = ImageGen.load_from_checkpoint(ckpt_path, model=model)
    model.eval()
    z = model.generate_noize(16)
    generated = model.model.generator(z)
    save_image(generated, "generated.png", normalize=True)
    return "generated.png"


with gr.Blocks("Model") as app:
    gr.Markdown(
        """
        # Image Classification App
    """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_name = gr.Dropdown(list(registered_models.keys()), label="Model")
                backbone = gr.Dropdown(list(registered_models.keys()), label="Backbone")
            with gr.Row():
                target_shape = gr.Dropdown(
                    [
                        "1x8x8",
                        "3x32x32",
                        "3x256x256",
                    ],
                    label="Target shape",
                )
                with gr.Column():
                    ckpt_path = gr.Textbox(label="Checkpoint path")
                    ckpt_btn = gr.Button(value="Select checkpoint")

            def set_ckpt_path():
                path = filedialog.askopenfilename(
                    initialdir="/workspace/experiments",
                    title="Select checkpoint file",
                )
                return path

            ckpt_btn.click(fn=set_ckpt_path, outputs=ckpt_path)

            genarate_btn = gr.Button(value="Generate")

        with gr.Column():
            result = gr.Image(label="Result")
    genarate_btn.click(
        fn=generate, inputs=[ckpt_path, model_name, target_shape, backbone], outputs=result
    )

app.launch()
