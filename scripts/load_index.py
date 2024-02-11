import os
from os import PathLike
from typing import List, Dict, Optional, Literal, cast

from llama_index import (
    Document,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.schema import BaseNode
from llama_index.storage.storage_context import StorageContext
from llama_index.indices.base import BaseIndex


def load_index(
    save_dir: PathLike[str],
    service_context: ServiceContext,
    storage_context: Optional[StorageContext] = None,
    document: Optional[Document] = None,
    leaf_nodes: Optional[List[BaseNode]] = None,
    type: Literal["basic", "auto_merging"] = "basic",
) -> VectorStoreIndex | BaseIndex:
    vector_index_dict: Dict[str, VectorStoreIndex] = {
        "basic": VectorStoreIndex(
            cast(List[BaseNode], [document]), service_context=service_context
        ),
        # "auto_merging": VectorStoreIndex(
        #     nodes=leaf_nodes,
        #     storage_context=storage_context,
        #     service_context=service_context,
        # ),
    }

    if not os.path.exists(save_dir):
        index = vector_index_dict[type]
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = cast(
            VectorStoreIndex,
            load_index_from_storage(
                StorageContext.from_defaults(persist_dir=cast(str, save_dir)),
                service_context=service_context,
            ),
        )
    return index
