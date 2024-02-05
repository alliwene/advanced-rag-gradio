import warnings
from scripts import utils

warnings.filterwarnings("ignore")

import openai
from llama_index.llms import OpenAI


openai.api_key = utils.get_openai_api_key()

model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name, temperature=0.1)
