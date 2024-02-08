from typing import List, Literal

from scripts.query_engine_builder import build_index, get_query_engine

from llama_index import Document


def build_index_get_query_engine(
    documents: List[Document],
    llm,
    embed_model: str = "local:BAAI/bge-small-en-v1.5",
    window_size: int = 3,
    chunk_sizes: List[int] | None = None,
    rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
):
    index = build_index(
        documents=documents,
        llm=llm,
        embed_model=embed_model,
        window_size=window_size,
        chunk_sizes=chunk_sizes,
        rag_type=rag_type,
    )

    query_engine = get_query_engine(
        index=index,
        similarity_top_k=similarity_top_k,
        rerank_top_n=rerank_top_n,
        rag_type=rag_type,
    )

    return query_engine
