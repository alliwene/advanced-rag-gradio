import os
from os import PathLike
from typing import List, Literal
from tempfile import _TemporaryFileWrapper

from scripts.utils import get_openai_api_key, hash_file
from scripts.chat_engine_builder import ChatEngineBuilder

import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine.types import BaseChatEngine


openai.api_key = get_openai_api_key()

model_name = "gpt-3.5-turbo-0125"
llm = OpenAI(model=model_name, temperature=0.1)
embed_model = OpenAIEmbedding()

api_keys: List[str] = ["OPENAI_API_KEY"]

assert any(
    os.getenv(api_key, None) for api_key in api_keys
), "Add 'OPENAI_API_KEY' or 'COHERE_API_KEY' in your environment variables"


def execute(
    file: _TemporaryFileWrapper,
    rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
) -> BaseChatEngine:
    file_path: PathLike[str] = file.name
    save_dir: PathLike[str] = hash_file(file)

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)

    engine_builder = ChatEngineBuilder(nodes, llm, embed_model, save_dir, rag_type)
    chat_engine = engine_builder.build_chat_engine()

    return chat_engine
