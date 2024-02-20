from pprint import pprint
import warnings
from typing import List

warnings.filterwarnings("ignore")

import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.utils import LLMType

from scripts import utils

openai.api_key = utils.get_openai_api_key()

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()
llm = OpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)


# service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(documents, embed_model=OpenAIEmbedding())

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = index.as_chat_engine(
    llm=llm,
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
    similarity_top_k=3,
    memory=memory,
    verbose=True,
)

# messages: List[ChatMessage] = []

for _ in range(2):
    print("\n")
    question = input("Ask me anything: ")
    # messages.append(ChatMessage(role=MessageRole.USER, content=question))
    response = chat_engine.chat(
        question,
        # chat_history=messages,
    )
    # messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response.response))
    # for token in response.response_gen:
    #     print(token, end="")
    pprint(response.response)

# print("\n")
# pprint(messages)
