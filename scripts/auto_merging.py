import os
from os import PathLike
from typing import List, cast, Optional

from llama_index import (
    Document,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.llms import OpenAI

from scripts.load_index import load_index


def build_automerging_index(
    documents: List[Document],
    llm: OpenAI,
    embed_model: str = "local:BAAI/bge-small-en-v1.5",
    save_dir: PathLike[str] = cast(PathLike[str], "merging_index"),
    chunk_sizes: Optional[List[int]] = None,
) -> VectorStoreIndex:
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index(
            save_dir=save_dir, service_context=merging_context
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index: VectorStoreIndex,
    similarity_top_k: int = 12,
    rerank_top_n: int = 2,
) -> RetrieverQueryEngine:
    base_retriever = automerging_index.as_retriever(
        similarity_top_k=similarity_top_k, streaming=True
    )
    retriever = AutoMergingRetriever(
        cast(VectorIndexRetriever, base_retriever),
        automerging_index.storage_context,
        verbose=True,
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], streaming=True
    )
    return auto_merging_engine
