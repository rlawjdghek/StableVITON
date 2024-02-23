import gradio as gr
import cv2

import os

from inference_app import init, predict, args


def _predict(path, cloth):
    (_, im_name) = os.path.split(path)
    im_name = im_name.split(".")[0]
    im_name += '.jpg'

    cloth = cv2.cvtColor(cloth, cv2.COLOR_RGB2BGR)
    return predict(im_name, cloth)


def build_ui():
    models_path = os.path.join(args.data_root_dir, "test", "image")
    models_files = list(map(lambda item: os.path.join(models_path, item), os.listdir(models_path)))
    cloth_path = os.path.join(args.data_root_dir, "test", "cloth")
    cloth_files = list(map(lambda item: os.path.join(cloth_path, item), os.listdir(cloth_path)))
    init_model = models_files[0]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                init_image = gr.Image(sources=["clipboard"], type="filepath", label="Model", value=init_model)
                gr.Examples(inputs=init_image,
                            examples_per_page=4,
                            examples=models_files)
            with gr.Column():
                with gr.Row():
                    garment = gr.Image(sources=["upload"], label="Cloth")
                with gr.Row():
                    gr.Examples(inputs=[garment],
                                examples_per_page=4,
                                examples=cloth_files)
            with gr.Column():
                run_button = gr.Button(value="Run")
                output_img = gr.Image()

        # 运行
        run_button.click(fn=_predict,
                         inputs=[init_image, garment, ],
                         outputs=[output_img])
    return demo


def launch():
    init()
    build_ui().launch()
    pass


if __name__ == '__main__':
    launch()
