import os
import cv2
import glob
import numpy as np
import gradio as gr
from PIL import Image
from src.process_gradio import main_process

title = 'Avocado Ripe Classification'
# css = ".image-preview {height: auto !important;}"

inputs = [gr.inputs.Image(source='upload')]
outputs = [gr.outputs.Image(label='image output'), gr.Label(label='Classification')]
examples = [[f'{i}'] for i in glob.glob('images/*.jpg')]

iface = gr.Interface(
    title   = title,
    fn      = main_process, 
    inputs  = inputs, 
    outputs = outputs,
    examples= examples,
    # css=css
)

iface.launch(share=True, server_port=8081)