from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import numpy as np

def get_store():
    # os.makedirs("./vectorstore", exist_ok=True)
    # s3 = boto3.client('s3', region_name="eu-north-1")
    # s3.download_file("rag-faiss-index-bucket", "vectorstores/index.faiss", "./vectorstore/index.faiss")
    # s3.download_file("rag-faiss-index-bucket", "vectorstores/index.pkl", "./vectorstore/index.pkl")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)