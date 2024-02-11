import os
from os import PathLike
from typing import List, cast

from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.query_engine import BaseQueryEngine
from llama_index.indices.base import BaseIndex
from llama_index.llms import OpenAI

def build_basic_rag_index(
    documents: List[Document],
    llm: OpenAI,
    embed_model: str = "local:BAAI/bge-small-en-v1.5",
    save_dir: PathLike[str] = cast(PathLike[str], "basic_rag_index"),
) -> VectorStoreIndex | BaseIndex:
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents(
            [document], service_context=service_context
        )
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


def get_basic_rag_query_engine(
    index: VectorStoreIndex | BaseIndex, similarity_top_k: int = 6
) -> BaseQueryEngine:
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, streaming=True
    )

    return query_engine
