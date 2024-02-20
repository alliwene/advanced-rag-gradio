import os
from os import PathLike
from typing import cast

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.embeddings.utils import EmbedType


def load_index(
    document: Document, embed_model: EmbedType, save_dir: PathLike[str]
) -> VectorStoreIndex | BaseIndex:
    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = cast(VectorStoreIndex, index_from_storage(save_dir))

    return index


def index_from_storage(save_dir: PathLike[str]) -> BaseIndex:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=cast(str, save_dir)),
    )

    return index
