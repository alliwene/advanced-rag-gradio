import warnings
from scripts import utils

warnings.filterwarnings("ignore")

import openai
import nest_asyncio
from llama_index.llms import OpenAI

from scripts.basic_rag import build_basic_rag_index, get_basic_rag_query_engine
from scripts.sentence_window import build_sentence_window_index, get_sentence_window_query_engine
from scripts.auto_merging import build_automerging_index, get_automerging_query_engine

nest_asyncio.apply() 

openai.api_key = utils.get_openai_api_key()

model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name, temperature=0.1)


