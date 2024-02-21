from typing import cast
from scripts.utils import memory

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine


def get_automerging_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k=12,
    rerank_top_n=2,
) -> BaseChatEngine:
    base_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k, streaming=True
    )
    retriever = AutoMergingRetriever(
        cast(VectorIndexRetriever, base_retriever), index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], streaming=True
    )

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        query_engine=auto_merging_engine,
        verbose=True,
        streaming=True,
    )

    return chat_engine
