import os
from os import PathLike
from typing import List, cast, Tuple
from tempfile import _TemporaryFileWrapper

from scripts.utils import get_openai_api_key, hash_file, RAGType, Capturing
from scripts.chat_engine_builder import ChatEngineBuilder

import openai
import tiktoken
from llama_index.core import Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager


openai.api_key = get_openai_api_key()

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    verbose=True,
)
Settings.callback_manager = CallbackManager([token_counter])

embed_model = OpenAIEmbedding()
llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)

api_keys: List[str] = ["OPENAI_API_KEY"]

assert all(os.getenv(api_key, None) for api_key in api_keys), (
    "Add " + ", ".join(api_keys) + " in your environment variables"
)


class ChabotInterface(ChatEngineBuilder):
    def __init__(self):
        super().__init__(llm=llm, embed_model=embed_model)

    def generate_response(
        self,
        file: _TemporaryFileWrapper,
        chat_history: List[Tuple[str, str]],
        rag_type: RAGType = "basic",
    ):
        file_path: PathLike[str] = cast(PathLike[str], file.name)
        save_dir: PathLike[str] = cast(
            PathLike[str], f"saved_index/{hash_file(file)}/{rag_type}"
        )

        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        nodes = parser.get_nodes_from_documents(documents)

        chat_engine = self.build_chat_engine(
            cast(List[Document], nodes), save_dir, rag_type
        )
        self.chat_engine = chat_engine

        with Capturing() as output:
            response = self.chat_engine.stream_chat(chat_history[-1][0])

        output_text = "\n".join(output)
        for token in response.response_gen:
            chat_history[-1][1] += token  # type: ignore
            return chat_history, str(output_text)

    def reset_chat(self) -> Tuple[List, str, str]:
        self.chat_engine.reset()
        return [], "", ""

    def _token_count(self) -> None:
        print(
            "Embedding Tokens: ",
            token_counter.total_embedding_token_count,
            "\n",
            "LLM Prompt Tokens: ",
            token_counter.prompt_llm_token_count,
            "\n",
            "LLM Completion Tokens: ",
            token_counter.completion_llm_token_count,
            "\n",
            "Total LLM Token Count: ",
            token_counter.total_llm_token_count,
        )
