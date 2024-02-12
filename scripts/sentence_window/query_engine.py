from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index import (
    VectorStoreIndex,
)
from llama_index.query_engine import BaseQueryEngine
from llama_index.indices.base import BaseIndex


def get_sentence_window_query_engine(
    index: VectorStoreIndex | BaseIndex,
    similarity_top_k=6,
    rerank_top_n=2,
) -> BaseQueryEngine:
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
        streaming=True,
    )
    return sentence_window_engine
