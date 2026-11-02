from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, Tool
from langfuse import get_client
from dotenv import load_dotenv, dotenv_values 
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# from langchain_community.retrievers import BM25Retrieverv
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()

# langfuse = get_client()

search_tool = DuckDuckGoSearchTool()

class PartyPlanningRetrieverTool(Tool):
    name="party_planning_retrival"
    description='Uses Semantic Search to retrive relevant party planning ideas as for Alfreds superhero-themed party at Wayne Manor.'
    inputs={
        "query":{
            "type":"string",
            "description": "The Query to perform, This should be a query related to party planning or superhero themes.",
        }
    }
    output_type="string"

    def __init__(self,docs,**kwargs):
        super().__init__(**kwargs)
        self.embedding_model = HuggingFaceEmbeddings(
             model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding_model
        )

        self.retiever = self.vector_store.as_retriever(
            search_kwargs={"k": 2}
        )

    def forward(self, query:str) -> str:
        assert isinstance(query,str), "Your Search Query must be a string"

        docs = self.retiever.invoke(
            query
        )
        return "Retrived Ideas "+ " ".join([
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ])

party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

party_ideas.extend([
    {
        "text": "Guests can explore immersive simulated environments using head-mounted displays.",
        "source": "VR Experiences"
    },
    {
        "text": "The event features a sophisticated masked gala with formal evening attire.",
        "source": "Luxury Events"
    },
    {
        "text": "Professional chefs prepare a five-course tasting experience for attendees.",
        "source": "Catering"
    },
    {
        "text": "The venue offers an augmented reality experience that overlays digital characters onto the physical world.",
        "source": "AR Entertainment"
    },
    {
        "text": "The gaming company develops virtual reality training simulations for industrial workers.",
        "source": "Industrial Technology"
    }
])

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]        

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = text_splitter.split_documents(source_docs)

party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)



model = InferenceClientModel(
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    max_tokens=2048,
    temperature=0.1
)

print(party_planning_retriever(query='Artificial Reality'))

# SmolagentsInstrumentor().instrument()

# agent = CodeAgent(tools=[party_planning_retriever], model=InferenceClientModel())

# response = agent.run(
#     "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
# )

# langfuse.flush()