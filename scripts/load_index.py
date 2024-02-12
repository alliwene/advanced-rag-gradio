from os import PathLike
from typing import cast

from llama_index import (
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.storage.storage_context import StorageContext


def load_index(
    save_dir: PathLike[str],
    service_context: ServiceContext,
) -> VectorStoreIndex:
    index = cast(
        VectorStoreIndex,
        load_index_from_storage(
            StorageContext.from_defaults(persist_dir=cast(str, save_dir)),
            service_context=service_context,
        ),
    )
    return index
