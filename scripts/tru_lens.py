from trulens_eval import Feedback, TruLlama, OpenAI
from trulens_eval.feedback import Groundedness

import numpy as np

openai = OpenAI()

qa_relevance = Feedback(
    openai.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

# grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(query_engine, app_id=app_id, feedbacks=feedbacks)
    return tru_recorder
