import warnings

warnings.filterwarnings("ignore")

import openai
from llama_index import SimpleDirectoryReader

from scripts import utils
from backend import build_index


openai.api_key = utils.get_openai_api_key()

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

index = build_index(documents, rag_type="veronica")

print(index)
