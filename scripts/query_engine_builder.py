import warnings
from os import PathLike
from typing import List, Literal

import utils
from basic_rag import build_basic_rag_index, get_basic_rag_query_engine
from sentence_window import (
    build_sentence_window_index,
    get_sentence_window_query_engine,
)
from auto_merging import build_automerging_index, get_automerging_query_engine

import nest_asyncio
import openai
from llama_index.llms import OpenAI
from llama_index import Document, VectorStoreIndex
from llama_index.indices.base import BaseIndex
from llama_index.query_engine import RetrieverQueryEngine, BaseQueryEngine


nest_asyncio.apply()

warnings.filterwarnings("ignore")

openai.api_key = utils.get_openai_api_key()

model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name, temperature=0.1)


def build_index(
    documents: List[Document],
    # save_dir: PathLike,
    llm=llm,
    embed_model: str = "local:BAAI/bge-small-en-v1.5",
    window_size: int = 3,
    chunk_sizes: List[int] | None = None,
    rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
):
    index_params = {
        "documents": documents,
        "llm": llm,
        "embed_model": embed_model,
        # "save_dir": save_dir,
    }

    index_builders = {
        "basic": build_basic_rag_index(**index_params),
        "sentence_window": build_sentence_window_index(
            **index_params,
            window_size=window_size,
        ),
        "auto_merging": build_automerging_index(
            **index_params,
            chunk_sizes=chunk_sizes,
        ),
    }

    try:
        return index_builders[rag_type]
    except KeyError:
        raise ValueError(f"Invalid rag_type: {rag_type}")


def get_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
    rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
) -> BaseQueryEngine | RetrieverQueryEngine:
    query_params = {
        "index": index,
        "similarity_top_k": similarity_top_k,
    }
    query_engines = {
        "basic": get_basic_rag_query_engine(**query_params),
        "sentence_window": get_sentence_window_query_engine(
            **query_params, rerank_top_n=rerank_top_n
        ),
        "auto_merging": get_automerging_query_engine(
            **query_params, rerank_top_n=rerank_top_n
        ),
    }

    try:
        return query_engines[rag_type]
    except KeyError:
        raise ValueError(f"Invalid rag_type: {rag_type}")
