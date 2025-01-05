import gradio as gr
from model import model

def predict(input_text):
    return model.predict(input_text)

interface = gr.Interface(fn=predict, inputs="text", outputs="text")
interface.launch(server_port=8080)
