import warnings

warnings.filterwarnings("ignore")

from llama_index import SimpleDirectoryReader
from llama_index.query_engine import BaseQueryEngine, RetrieverQueryEngine

from backend import build_index_and_query_engine


documents = SimpleDirectoryReader(
    input_files=["pdfs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

query_engine: BaseQueryEngine | RetrieverQueryEngine = build_index_and_query_engine(
    documents, rag_type="auto_merging"
)

response = query_engine.query("What is the best way to build a career in AI?")

response.print_response_stream()
