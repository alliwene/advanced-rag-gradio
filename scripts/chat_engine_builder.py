import warnings
from os import PathLike
from typing import List

from scripts.utils import IndexParams, QueryParams, RAGType

from scripts.basic_rag.build_index import build_basic_rag_index
from scripts.basic_rag.chat_engine import build_basic_rag_chat_engine

from scripts.sentence_window.build_index import build_sentence_window_index
from scripts.sentence_window.chat_engine import build_sentence_window_chat_engine

from scripts.auto_merging.build_index import build_automerging_index
from scripts.auto_merging.chat_engine import build_automerging_chat_engine

from llama_index.core import Document
from llama_index.core.indices.base import BaseIndex
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.llms.utils import LLMType


# import nest_asyncio

# nest_asyncio.apply()
warnings.filterwarnings("ignore")


class ChatEngineBuilder:
    def __init__(
        self,
        llm: LLMType,
        embed_model: EmbedType,
    ):
        self.embed_model = embed_model
        self.llm = llm

        self.rag_types: RAGType = ["basic", "sentence_window", "auto_merging"]

    def build_index(
        self,
        documents: List[Document],
        save_dir: PathLike[str],
        rag_type: RAGType = "basic",
        window_size: int = 3,
        chunk_sizes: List[int] | None = None,
    ) -> BaseIndex:
        index_params: IndexParams = {
            "documents": documents,
            "embed_model": self.embed_model,
            "save_dir": save_dir,
        }

        index_builders = {
            "basic": lambda: build_basic_rag_index(**index_params),
            "sentence_window": lambda: build_sentence_window_index(
                **index_params,
                window_size=window_size,
            ),
            "auto_merging": lambda: build_automerging_index(
                **index_params,
                chunk_sizes=chunk_sizes,
            ),
        }

        try:
            return index_builders[rag_type]()
        except KeyError:
            raise ValueError(f"Invalid rag_type: {rag_type}")

    def build_chat_engine(
        self,
        documents: List[Document],
        save_dir: PathLike[str],
        rag_type: RAGType = "basic",
        similarity_top_k: int = 6,
        rerank_top_n: int = 2,
    ) -> BaseChatEngine:

        query_params: QueryParams = {
            "index": self.build_index(documents, save_dir, rag_type),
            "similarity_top_k": similarity_top_k,
            "llm": self.llm,
        }

        query_engines = {
            "basic": lambda: build_basic_rag_chat_engine(**query_params),
            "sentence_window": lambda: build_sentence_window_chat_engine(
                **query_params, rerank_top_n=rerank_top_n
            ),
            "auto_merging": lambda: build_automerging_chat_engine(
                **query_params, rerank_top_n=rerank_top_n
            ),
        }

        try:
            return query_engines[rag_type]()
        except KeyError:
            raise ValueError(f"Invalid rag_type: {rag_type}")
