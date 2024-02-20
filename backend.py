from typing import List, Literal

import openai
from llama_index.core import Document
from llama_index.llms.openai import OpenAI

from scripts.query_engine_builder import build_index, get_query_engine
from scripts.utils import get_openai_api_key
from llama_index.core.embeddings.utils import EmbedType

openai.api_key = get_openai_api_key()

model_name = "gpt-3.5-turbo-0125"
llm = OpenAI(model=model_name, temperature=0.1)

#! Redundant code
def build_index_and_query_engine(
    documents: List[Document],
    embed_model: EmbedType,
    window_size: int = 3,
    chunk_sizes: List[int] | None = None,
    rag_type: Literal["basic", "sentence_window", "auto_merging"] = "basic",
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
):
    index = build_index(
        documents=documents,
        embed_model=embed_model,
        window_size=window_size,
        chunk_sizes=chunk_sizes,
        rag_type=rag_type,
    )

    query_engine = get_query_engine(
        index=index,
        similarity_top_k=similarity_top_k,
        rerank_top_n=rerank_top_n,
        rag_type=rag_type,
    )

    return query_engine
