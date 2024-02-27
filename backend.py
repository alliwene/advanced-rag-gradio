import os
from os import PathLike
from typing import List, Tuple, cast
from tempfile import _TemporaryFileWrapper

from scripts.utils import get_openai_api_key, hash_file, Capturing, RAGType
from scripts.chat_engine_builder import ChatEngineBuilder

import openai
import tiktoken
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings, set_global_handler, global_handler

set_global_handler("wandb", run_args={"project": "llamaindex-advanced-rag"})
wandb_callback = global_handler

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    verbose=True,
)

Settings.callback_manager = CallbackManager([token_counter])

openai.api_key = get_openai_api_key()

llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
embed_model = OpenAIEmbedding()

api_keys: List[str] = ["OPENAI_API_KEY"]

assert any(os.getenv(api_key, None) for api_key in api_keys), (
    "Add " + " ".join(api_keys) + " in your environment variables"
)
assert get_openai_api_key().startswith(
    "sk-"
), "This doesn't look like a valid OpenAI API key"


class ChatbotInterface(ChatEngineBuilder):
    def __init__(self):
        super().__init__(llm, embed_model)
        self.token_counter = token_counter

    def generate_response(
        self,
        file: _TemporaryFileWrapper,
        chat_history: List[Tuple[str, str]],
        rag_type: RAGType = "basic",
    ):
        """Generate the response from rag, and capture the stdout (similarity search result)
        of the rag.
        """
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
            yield chat_history, str(output_text)

    def reset_chat(self) -> Tuple[List, str, str]:
        """Reset the chat history. And clear all dialogue boxes."""
        self.chat_engine.reset()
        return [], "", ""

    def _token_usage(self) -> None:
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
