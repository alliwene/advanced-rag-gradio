import os
from os import PathLike
from typing import List, cast

from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index import (
    load_index_from_storage,
    Document,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.query_engine import BaseQueryEngine
from llama_index.indices.base import BaseIndex
from llama_index.llms import OpenAI


def build_sentence_window_index(
    documents: List[Document],
    llm: OpenAI,
    embed_model: str = "local:BAAI/bge-small-en-v1.5",
    save_dir: PathLike[str] = cast(PathLike[str], "sentence_index"),
    window_size: int = 3,
) -> VectorStoreIndex | BaseIndex:
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = cast(
            VectorStoreIndex,
            load_index_from_storage(
                StorageContext.from_defaults(persist_dir=cast(str, save_dir)),
                service_context=sentence_context,
            ),
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index: VectorStoreIndex | BaseIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
) -> BaseQueryEngine:
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
        streaming=True,
    )
    return sentence_window_engine
