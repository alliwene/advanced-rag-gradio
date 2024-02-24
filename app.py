from tempfile import _TemporaryFileWrapper
from typing import List, Tuple, Generator, Literal

import gradio as gr


css = """
h1 {
    text-align: center;
    display:block;
}

#box { height: 420px; overflow-y: scroll !important}
"""


def add_text(chat_history: List[Tuple[str, str]], query: str):
    chat_history += [(query, "")]
    return chat_history, gr.update(value="", interactive=False)


# def generate_response(
#     file: _TemporaryFileWrapper,
#     chat_history: List[Tuple[str, str]],
#     rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
# ) -> Generator[str, List[Tuple[str, str]], str]:
#     chat_engine = execute(file, rag_type)

#     with Capturing() as output:
#         response = chat_engine.stream_chat(chat_history[-1][0])

#     ansi = "\n========\n".join(output)
#     html_output = Ansi2HTMLConverter().convert(ansi)
#     for token in response.response_gen:
#         chat_history[-1][1] += token
#         yield chat_history, str(html_output)


#! Take care of multiple files
with gr.Blocks(css=css) as demo:
    gr.Markdown("# ADVANCED RAG GPT")

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Message History",
            scale=2,
        )
        console = gr.HTML(elem_id="box")

    with gr.Row():
        query = gr.Textbox(placeholder="Input your question...", scale=4)

        file = gr.File(
            type="filepath",
            label="Upload a file",
            height=0.1,
            file_types=["text", ".pdf"],
        )

        rag_type = gr.Dropdown(
            ["basic", "sentence_window", "auto_merging"],
            value="basic",
            label="RAG Type",
            max_choices=1,
            info="testing testing",
        )

    with gr.Row():
        btn = gr.Button(value="Submit")
        clear = gr.ClearButton([query, chatbot, console], variant="primary")

    # file.upload(fn=lambda x: x.name, inputs=file, outputs=query)
    response = gr.on(
        triggers=[btn.click, query.submit],
        fn=add_text,
        inputs=[chatbot, query],
        outputs=[chatbot, query],
    ).then(
        fn=generate_response,
        inputs=[file, chatbot, rag_type],
        outputs=[chatbot, console],
    )

    response.then(lambda: gr.update(interactive=True), None, [query], queue=False)

if __name__ == "__main__":
    demo.queue().launch()
