from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

from llama_index import SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.chat_engine.types import ChatMode

import openai

from scripts import utils
from scripts.basic_rag import build_basic_rag_index, get_basic_rag_query_engine
from scripts.sentence_window import (
    build_sentence_window_index,
    get_sentence_window_query_engine,
)
from scripts.auto_merging import build_automerging_index, get_automerging_query_engine


openai.api_key = utils.get_openai_api_key()

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()


llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)


# Necessary to use the latest OpenAI models that support function calling API
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode=ChatMode.OPENAI, verbose=True)

for _ in range(3):
    question = input("Ask me anything: ")
    response = chat_engine.stream_chat(question, tool_choice="query_engine_tool")
