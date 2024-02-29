from scripts.utils import memory

from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.core.llms.utils import LLMType


def build_sentence_window_chat_engine(
    llm: LLMType,
    index: BaseIndex,
    similarity_top_k=6,
    rerank_top_n=2,
) -> BaseChatEngine:
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )

    sentence_window_engine = index.as_chat_engine(
        llm=llm,
        memory=memory,
        similarity_top_k=similarity_top_k,
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        node_postprocessors=[postproc, rerank],
        verbose=True,
    )

    return sentence_window_engine
