import warnings
from scripts import utils
from typing import List, Literal
from os import PathLike

warnings.filterwarnings("ignore")

import openai
import nest_asyncio
from llama_index import Document, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.indices.base import BaseIndex

from scripts.basic_rag import build_basic_rag_index, get_basic_rag_query_engine
from scripts.sentence_window import (
    build_sentence_window_index,
    get_sentence_window_query_engine,
)
from scripts.auto_merging import build_automerging_index, get_automerging_query_engine

nest_asyncio.apply()

openai.api_key = utils.get_openai_api_key()

model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name, temperature=0.1)


def build_index(
    documents: List[Document],
    llm,
    save_dir: PathLike,
    chunk_sizes: List[int] | None = None,
    window_size: int = 3,
    rag_type: Literal["basic", "automerging", "sentence_window"] = "basic",
):
    build_index_dict = {
        "basic": build_basic_rag_index(documents=documents, llm=llm),
        "automerging": build_automerging_index(
            documents=documents, llm=llm, chunk_sizes=chunk_sizes
        ),
        "sentence_window": build_sentence_window_index(
            documents=documents, llm=llm, window_size=window_size
        ),
    }

    return build_index_dict[rag_type]


def get_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
    rag_type: Literal["basic", "automerging", "sentence_window"] = "basic",
):
    get_query_engine_dict = {
        "basic": get_basic_rag_query_engine(
            index,
            similarity_top_k=similarity_top_k,
        ),
        "automerging": get_automerging_query_engine(
            automerging_index=index,
            similarity_top_k=similarity_top_k,
            rerank_top_n=rerank_top_n,
        ),
        "sentence_window": get_sentence_window_query_engine(
            sentence_index=index,
            similarity_top_k=similarity_top_k,
            rerank_top_n=rerank_top_n,
        ),
    }

    return get_query_engine_dict[rag_type]
