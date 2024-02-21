import warnings
from os import PathLike
from typing import List, Literal, TypedDict


from scripts.basic_rag.build_index import build_basic_rag_index
from scripts.basic_rag.query_engine import get_basic_rag_query_engine

from scripts.sentence_window.build_index import build_sentence_window_index
from scripts.sentence_window.query_engine import get_sentence_window_query_engine

from scripts.auto_merging.build_index import build_automerging_index
from scripts.auto_merging.query_engine import get_automerging_query_engine

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine, BaseQueryEngine
from llama_index.core.embeddings.utils import EmbedType


import nest_asyncio

nest_asyncio.apply()
warnings.filterwarnings("ignore")


class IndexParams(TypedDict):
    documents: List[Document]
    embed_model: EmbedType
    save_dir: PathLike[str]


class QueryParams(TypedDict):
    index: VectorStoreIndex | BaseIndex
    similarity_top_k: int


class QueryEngineBuilder:
    def __init__(
        self,
        documents: List[Document],
        embed_model: EmbedType,
        save_dir: PathLike[str],
        rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
    ):
        self.embed_model = embed_model
        self.documents = documents
        self.rag_type = rag_type
        self.save_dir = save_dir

        self.index_params: IndexParams = {
            "documents": self.documents,
            "embed_model": self.embed_model,
            "save_dir": self.save_dir,
        }

    def build_index(
        self,
        window_size: int = 3,
        chunk_sizes: List[int] | None = None,
    ) -> VectorStoreIndex | BaseIndex:
        index_builders = {
            "basic": lambda: build_basic_rag_index(**self.index_params),
            "sentence_window": lambda: build_sentence_window_index(
                **self.index_params,
                window_size=window_size,
            ),
            "auto_merging": lambda: build_automerging_index(
                **self.index_params,
                chunk_sizes=chunk_sizes,
            ),
        }

        try:
            return index_builders[self.rag_type]()
        except KeyError:
            raise ValueError(f"Invalid rag_type: {self.rag_type}")

    def get_query_engine(
        self,
        similarity_top_k: int = 6,
        rerank_top_n: int = 2,
    ) -> BaseQueryEngine | RetrieverQueryEngine:

        query_params: QueryParams = {
            "index": self.build_index(),
            "similarity_top_k": similarity_top_k,
        }

        query_engines = {
            "basic": lambda: get_basic_rag_query_engine(**query_params),
            "sentence_window": lambda: get_sentence_window_query_engine(
                **query_params, rerank_top_n=rerank_top_n
            ),
            "auto_merging": lambda: get_automerging_query_engine(
                **query_params, rerank_top_n=rerank_top_n
            ),
        }

        try:
            return query_engines[self.rag_type]()
        except KeyError:
            raise ValueError(f"Invalid rag_type: {self.rag_type}")
