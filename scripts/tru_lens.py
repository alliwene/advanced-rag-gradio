from scripts.utils import get_openai_api_key

import numpy as np
import openai
from trulens_eval import Feedback, TruLlama, OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.query_engine import BaseQueryEngine


openai.api_key = get_openai_api_key()
provider = fOpenAI()

qa_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

qs_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

# grounded = Groundedness(groundedness_provider=provider, summarize_provider=provider)
grounded = Groundedness(groundedness_provider=provider)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


def get_prebuilt_trulens_recorder(
    query_engine: BaseChatEngine | BaseQueryEngine, app_id: str
):
    tru_recorder = TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)
    return tru_recorder
