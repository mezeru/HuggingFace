import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader




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

response = query_engine.query(
    "Why use LlamaIndex instead of BM25?"
)

print("\n=== LLM Response ===")
print(response.response)
