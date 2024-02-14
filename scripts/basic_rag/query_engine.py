from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex


def get_basic_rag_query_engine(
    index: VectorStoreIndex | BaseIndex, similarity_top_k=6
) -> BaseQueryEngine:
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, streaming=True
    )

    return query_engine
