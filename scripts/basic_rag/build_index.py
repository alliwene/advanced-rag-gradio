from typing import List

from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.indices.base import BaseIndex

from scripts.load_index import load_index


def build_basic_rag_index(
    documents: List[Document],
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="basic_rag_index",
) -> VectorStoreIndex | BaseIndex:
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    index = load_index(document, service_context, save_dir)

    return index
