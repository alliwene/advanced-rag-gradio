import os
from os import PathLike

from llama_index.core import (
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex


def load_index(
    document: Document, service_context: ServiceContext, save_dir: PathLike[str]
) -> VectorStoreIndex | BaseIndex:
    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents(
            [document], service_context=service_context
        )
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = index_from_storage(service_context, save_dir)

    return index


def index_from_storage(
    service_context: ServiceContext, save_dir: PathLike[str]
) -> BaseIndex:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=save_dir),
        service_context=service_context,
    )

    return index
