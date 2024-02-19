import warnings
from os import PathLike

from scripts import utils
import openai
from llama_index.llms.openai import OpenAI

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine, BaseQueryEngine

from typing import List, Literal

import nest_asyncio

nest_asyncio.apply()

warnings.filterwarnings("ignore")

openai.api_key = utils.get_openai_api_key()

model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name, temperature=0.1)

from scripts.basic_rag import build_basic_rag_index, get_basic_rag_query_engine
from scripts.sentence_window import (
    build_sentence_window_index,
    get_sentence_window_query_engine,
)
from scripts.auto_merging import build_automerging_index, get_automerging_query_engine


# def build_index(
#     documents: List[Document],
#     llm,
#     embed_model: str,
#     save_dir: PathLike,
#     window_size: int,
#     chunk_sizes: List[int] | None,
#     rag_type: Literal["basic", "sentence_window", "auto_merging"],
# ):
#     if rag_type == "basic":
#         return build_basic_rag_index(documents=documents, llm=llm, save_dir=save_dir, embed_model=embed_model)
#     elif rag_type == "sentence_window":
#         return build_sentence_window_index(
#             documents=documents,
#             llm=llm,
#             save_dir=save_dir,
#             embed_model=embed_model,
#             window_size=window_size
#         )
#     elif rag_type == "auto_merging":
#         return build_automerging_index(
#             documents=documents,
#             llm=llm,
#             save_dir=save_dir,
#             embed_model=embed_model,
#             chunk_sizes=chunk_sizes
#         )

# use a dictionary to fix the code with no if else


def build_index(
    documents: List[Document],
    llm,
    embed_model: str,
    save_dir: PathLike,
    window_size: int,
    chunk_sizes: List[int] | None,
    rag_type: Literal["basic", "sentence_window", "auto_merging"],
):
    index_builders = {
        "basic": build_basic_rag_index,
        "sentence_window": build_sentence_window_index,
        "auto_merging": build_automerging_index,
    }

    if rag_type in index_builders:
        return index_builders[rag_type](
            documents=documents,
            llm=llm,
            save_dir=save_dir,
            embed_model=embed_model,
            window_size=window_size,
            chunk_sizes=chunk_sizes,
        )

    # Define a function for get query engine
    # def get_query_engine (
    # index: VectorStoreIndex | BaseIndex, similarity_top_k: int, rerank_top_n: int, rag_type: Literal["basic", "sentence_window", "auto_merging"]
    # ) -> BaseQueryEngine | RetrieverQueryEngine:
    #     if rag_type == "basic":
    #         return get_basic_rag_query_engine(index, similarity_top_k=similarity_top_k)
    #     elif rag_type == "sentence_window":
    #         return get_sentence_window_query_engine(index, similarity_top_k=similarity_top_k, rerank_top_n=rerank_top_n)
    #     elif rag_type == "auto_merging":
    #         return get_automerging_query_engine(index, similarity_top_k=similarity_top_k, rerank_top_n=rerank_top_n)


def get_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k: int,
    rerank_top_n: int,
    rag_type: Literal["basic", "sentence_window", "auto_merging"],
) -> BaseQueryEngine | RetrieverQueryEngine:
    query_engines = {
        "basic": get_basic_rag_query_engine,
        "sentence_window": get_sentence_window_query_engine,
        "auto_merging": get_automerging_query_engine,
    }

    if rag_type in query_engines:
        return query_engines[rag_type](
            index, similarity_top_k=similarity_top_k, rerank_top_n=rerank_top_n
        )
