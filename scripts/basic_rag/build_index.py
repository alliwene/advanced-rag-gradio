from typing import List

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex

from scripts.load_index import load_index


def build_basic_rag_index(
    documents: List[Document],
    embed_model,
    save_dir="basic_rag_index",
) -> VectorStoreIndex | BaseIndex:
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    index = load_index(document, embed_model, save_dir)

    return index
