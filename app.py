from typing import List, Tuple, Dict

import gradio as gr
from backend import ChatbotInterface


css = """
h1 {
    text-align: center;
    display:block;
}

#upload { height: 120px; overflow-y: scroll !important}
"""


def add_text(
    chat_history: List[Tuple[str, str]], query: str
) -> Tuple[List[Tuple[str, str]], Dict]:
    chat_history += [(query, "")]
    return chat_history, gr.update(value="", interactive=False)


chatbot_interface = ChatbotInterface()

#! Take care of multiple files
with gr.Blocks(css=css) as demo:
    gr.Markdown("# ADVANCED RAG GPT")

    with gr.Row():
        chatbot = gr.Chatbot(label="Message History", scale=2, height="460px")
        console = gr.TextArea(
            label="Similarity Search Results",
            show_copy_button=True,
            lines=20,
            max_lines=20,
        )

    with gr.Row():
        query = gr.Textbox(
            placeholder="Input your question...", scale=4, show_copy_button=True
        )

        file = gr.File(
            type="filepath",
            label="Upload a file",
            height=0.1,
            file_types=["text", ".pdf"],
            elem_id="upload",
        )

        rag_type = gr.Dropdown(
            chatbot_interface.rag_types,
            value="basic",
            label="RAG Type",
            max_choices=1,
            info="Select the type of RAG to use",
        )

    with gr.Row():
        btn = gr.Button(value="Submit")
        clear_btn = gr.ClearButton([query, chatbot, console], variant="primary")

    clear_btn.click(chatbot_interface.reset_chat, None, [chatbot, query, console])

    response = gr.on(
        triggers=[btn.click, query.submit],
        fn=add_text,
        inputs=[chatbot, query],
        outputs=[chatbot, query],
    ).then(
        fn=chatbot_interface.generate_response,
        inputs=[file, chatbot, rag_type],
        outputs=[chatbot, console],
    )

    response.then(lambda: gr.update(interactive=True), None, [query], queue=False)

if __name__ == "__main__":
    demo.queue().launch()
