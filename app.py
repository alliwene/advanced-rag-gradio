from typing import List, Tuple
import gradio as gr

from backend import ChabotInterface


def add_text(chat_history: List[Tuple[str, str]], query: str):
    chat_history += [(query, "")]
    return chat_history, gr.update(value="", interactive=False)


chatbot_interface = ChabotInterface()


css = """
h1 {
    text-align: center;
    display:block;
}

#upload { height: 120px; overflow-y: scroll !important; }
"""

#! Take care of multiple files
with gr.Blocks(css=css) as demo:
    gr.Markdown("# ADVANCED RAG GPT")

    with gr.Row():
        chatbot = gr.Chatbot(
            show_copy_button=True,
            scale=2,
            height="460px",
            avatar_images=("images/user.jpg", "images/bot.png"),
        )
        console = gr.TextArea(
            label="Console",
            info="Contains token usage, similarity search result and other info from RAG",
            show_copy_button=True,
            lines=18,
            max_lines=18,
        )

    with gr.Row():
        query = gr.Textbox(
            label="Question",
            placeholder="Input your question...",
            show_copy_button=True,
            scale=4,
        )

        file = gr.File(
            type="filepath",
            label="Upload a file",
            file_types=["text", ".pdf"],
            elem_id="upload",
        )

        rag_type = gr.Dropdown(
            chatbot_interface.rag_types,
            value="basic",
            label="RAG Type",
            info="Choose the type of RAG to use",
            max_choices=1,
        )

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.ClearButton([query, chatbot, console], variant="primary")

    with gr.Accordion(
        label="Addditional Parameters for Sentence Window RAG", open=False
    ):
        with gr.Row():
            window_size = gr.Slider(
                value=3,
                minimum=1,
                maximum=5,
                step=1,
                label="Window Size",
                info="Number of Sentences to retrieve around text chunk",
            )
            top_n = gr.Slider(
                value=2,
                minimum=2,
                maximum=4,
                step=1,
                label="Rerank Top N",
                info="Top N for reranking the results from similarity search",
            )

    gr.on(
        triggers=[clear_btn.click, file.clear],
        fn=chatbot_interface.reset_chat,
        inputs=None,
        outputs=[chatbot, query, console],
    )

    response = gr.on(
        triggers=[submit_btn.click, query.submit],
        fn=add_text,
        inputs=[chatbot, query],
        outputs=[chatbot, query],
    ).then(
        fn=chatbot_interface.generate_response,
        inputs=[file, chatbot, rag_type, window_size, top_n],
        outputs=[chatbot, console],
    )

    response.then(lambda: gr.update(interactive=True), None, [query], queue=False)

    # file.upload(fn=lambda x: x.name, inputs=file, outputs=text)

if __name__ == "__main__":
    demo.queue().launch()
