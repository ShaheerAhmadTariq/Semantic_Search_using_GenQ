import gradio as gr
from retriever import get_relavent_docs


with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("## Search Engine")
    with gr.Column():
        inp = gr.Textbox(lines=2, placeholder="Enter your search query here...")
        out = gr.Dataframe(headers=["Content"])
    btn = gr.Button("Search")
    btn.click(get_relavent_docs, inputs=inp, outputs=out)

demo.launch()