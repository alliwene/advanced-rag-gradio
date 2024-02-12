import os
from typing import List

from llama_index import (
    Document,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.indices.base import BaseIndex

from scripts.load_index import index_from_storage

def build_automerging_index(
    documents: List[Document],
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
) -> VectorStoreIndex | BaseIndex:
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
        automerging_index = index_from_storage(merging_context, save_dir)
    return automerging_index