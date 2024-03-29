from scripts.utils import memory

from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine


def build_sentence_window_chat_engine(
    llm: LLMType,
    index: BaseIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2,
) -> BaseChatEngine:
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-TinyBERT-L-2-v2"
    )

    sentence_window_engine = index.as_chat_engine(
        llm=llm,
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        similiarity_top_k=similarity_top_k,
        memory=memory,
        verbose=True,
        node_postprocessors=[postproc, rerank],
    )

    return sentence_window_engine
