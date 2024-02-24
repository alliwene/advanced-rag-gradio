import os
from os import PathLike
from typing import List, Literal, Tuple, cast
from tempfile import _TemporaryFileWrapper

from scripts.utils import get_openai_api_key, hash_file, Capturing
from scripts.chat_engine_builder import ChatEngineBuilder

import openai
from ansi2html import Ansi2HTMLConverter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter


openai.api_key = get_openai_api_key()

llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
embed_model = OpenAIEmbedding()

api_keys: List[str] = ["OPENAI_API_KEY"]

assert any(os.getenv(api_key, None) for api_key in api_keys), (
    "Add " + " ".join(api_keys) + " in your environment variables"
)


class ChatbotInterface(ChatEngineBuilder):
    def __init__(self):
        super().__init__(llm, embed_model)

    def generate_response(
        self,
        file: _TemporaryFileWrapper,
        chat_history: List[Tuple[str, str]],
        rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
    ):
        """Generate the response from rag, and capture the stdout (similarity search result) 
        of the rag.
        """
        file_path: PathLike[str] = cast(PathLike[str], file.name)
        save_dir: PathLike[str] = cast(PathLike[str], f"saved_index/{hash_file(file)}")

        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        nodes = parser.get_nodes_from_documents(documents)

        chat_engine = self.build_chat_engine(
            cast(List[Document], nodes), save_dir, rag_type
        )
        self.chat_engine = chat_engine

        with Capturing() as output:
            response = chat_engine.stream_chat(chat_history[-1][0])

        ansi = "\n========\n".join(output)
        html_output = Ansi2HTMLConverter().convert(ansi)
        for token in response.response_gen:
            chat_history[-1][1] += token  # type: ignore
            yield chat_history, str(html_output)

    def reset_chat(self) -> Tuple[List, str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        self.chat_engine.reset()
        return [], "", ""
