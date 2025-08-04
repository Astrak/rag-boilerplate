from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import boto3
import os
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

prompt = PromptTemplate.from_template("You are an assistant for question-answering tasks. " +
"Use the following pieces of retrieved context to answer the question." + 
"If you don't know the answer, just say that you don't know. " +
"Use three sentences maximum and keep the answer concise. " +
"""Question: {question} 
Context: {context} 
Answer:""")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = openai_api_key

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
    raise EnvironmentError("LANGSMITH_API_KEY not found")

os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found")

os.environ["GOOGLE_API_KEY"] = google_api_key

os.makedirs("./vectorstore", exist_ok=True)
s3 = boto3.client('s3', region_name="eu-north-1")
s3.download_file("rag-faiss-index-bucket", "vectorstores/index.faiss", "./vectorstore/index.faiss")
s3.download_file("rag-faiss-index-bucket", "vectorstores/index.pkl", "./vectorstore/index.pkl")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve_from_store1(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve_from_store1, generate])
graph_builder.add_edge(START, "retrieve_from_store1")
graph = graph_builder.compile()

app = FastAPI()

class SearchRequest(BaseModel):
    question: str

@app.post("/search")
def search(request: SearchRequest):
    print('search request received: ' + request.question)
    result = graph.invoke({"question": request.question})  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer']}
