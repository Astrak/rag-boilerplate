from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import boto3
import os
from pydantic import BaseModel

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

os.environ["OPENAI_API_KEY"] = openai_api_key

app = FastAPI()

class SearchRequest(BaseModel):
    question: str

os.makedirs("./vectorstore", exist_ok=True)
s3 = boto3.client('s3', region_name="eu-north-1")
s3.download_file("rag-faiss-index-bucket", "vectorstores/index.faiss", "./vectorstore/index.faiss")
s3.download_file("rag-faiss-index-bucket", "vectorstores/index.pkl", "./vectorstore/index.pkl")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)

@app.post("/search")
def search(request: SearchRequest):
    query = request.question
    docs = vector_store.similarity_search(query)
    return {"results": [doc.page_content for doc in docs]}
