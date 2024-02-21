from typing import cast

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)


def get_automerging_query_engine(
    index: BaseIndex,
    similarity_top_k=12,
    rerank_top_n=2,
) -> RetrieverQueryEngine:
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

    return auto_merging_engine
