import os
from typing import List

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex

from scripts.load_index import index_from_storage


def build_sentence_window_index(
    documents: List[Document],
    embed_model,
    save_dir="sentence_index",
    window_size=3,
) -> VectorStoreIndex | BaseIndex:
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], embed_model=embed_model, transformations=[node_parser]
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = index_from_storage(save_dir)

    return sentence_index
