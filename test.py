import warnings

from llama_index.core import SimpleDirectoryReader
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer

from backend import build_index_and_query_engine
from scripts.query_engine_builder import build_index

warnings.filterwarnings("ignore")

documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

index = build_index(documents, rag_type="basic")

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
    verbose=True,
    similiarity_top_k=3,
    memory=memory,
)

for _ in range(2):
    print("\n")
    question = input("Ask me anything: ")
    response = chat_engine.chat(question)

    print(response.response)

# query_engine: BaseQueryEngine | RetrieverQueryEngine = build_index_and_query_engine(
#     documents, rag_type="auto_merging"
# )

# response = query_engine.query("What is the best way to build a career in AI?")

# response.print_response_stream()
