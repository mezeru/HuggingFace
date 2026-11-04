import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent import AgentWorkflow
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent
)
from llama_index.core.tools import QueryEngineTool



db = chromadb.PersistentClient(path='./chroma_db')

chroma_collections = db.get_or_create_collection('chroma')
vector_store = ChromaVectorStore(chroma_collection=chroma_collections)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        embed_model,
    ],
    vector_store=vector_store
)

if chroma_collections.count() == 0:

    documents = SimpleDirectoryReader(
        input_files=['./llamaindex_overview.txt']
    ).load_data()
    nodes = pipeline.run(documents=documents)

    print(f"Generated {len(nodes)} nodes")

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode='compact'
)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


calculate_agent = ReActAgent(
    name="Calculator Agent",
    description="Performs Basic Arithmetic Calculatins",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    llm=llm
    tools=[add,subtract]
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

query_engine_agent = AgentWorkflow(
    agents=[calculate_agent,query_agent],
    root_agent="calculator"
)


