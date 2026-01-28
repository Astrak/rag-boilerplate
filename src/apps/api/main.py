from apps.api.search_prompt import get_search_prompt
from apps.api.analyze_prompt import get_analyze_prompt
from apps.api.env import fill_env
from graph.main import Graph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from typing_extensions import TypedDict
from apps.api.ip_throttler_middleware import IPThrottleMiddleware
from datetime import datetime
import os
import boto3

sources = os.getenv("SOURCES")
if not sources:
    print("No SOURCES variable specified, using all sources in the knowledge-sources folder:")
    sources = os.listdir("./knowledge-sources/")
else:
    sources = sources.split(',')
print("Running RAG from sources: ", sources)

fill_env()

try:
    sync = input("Sync knowledge-sources from bucket? (y/n): ").lower().startswith('y')
    if sync:
        s3_client = boto3.client("s3")
        files = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket="rag-faiss-index-bucket"):
            for cp in page.get("Contents", []):
                file = cp["Key"]
                files.append(file)
        for file in files:
            print(file)
except Exception as e:
    raise ValueError('Didnt understand what to do with knowledge sources')

search_prompt = get_search_prompt()
analyze_prompt = get_analyze_prompt()

search_graph = Graph(search_prompt, sources)
analysis_graph = Graph(analyze_prompt, sources)

allowed_origins = [
    "https://polemia.surge.sh",
    "https://ia.polemia.com",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=[
        "*",
        "X-Twitter-Active-User",
        "X-Twitter-Client",
        "X-Twitter-Client-Version",
        "X-Twitter-API-Version",
        "X-Twitter-Auth-Type",
        "X-Twitter-Client-DeviceID",
        "X-Twitter-Client-Language",
        "X-Twitter-Response-Format",
    ],
)
app.add_middleware(IPThrottleMiddleware)

@app.get("/")
def home():
    return "hello world"

class SearchRequest(BaseModel):
    question: str
    sources: Optional[List[str]] = sources
    llm: str

class Resource(TypedDict):
    url: str
    title: str

@app.post("/retrieve")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('\033[93mRETRIEVE: Request received at: ' + str(time))
    print('RETRIEVE: Question is: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('RETRIEVE: Sources are: ' + ", ".join(request.sources))
    # analysis_graph.folders = sources.split(',')
    result = analysis_graph.retrieve({'question': request.question, 'discussion': ''}) 
    resources: list[Resource] = []
    for doc in result['context']:
        resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
    print('\033[93mRETRIEVE: Answered in ' + str(datetime.utcnow() - time))
    return {"resources": resources}

@app.post("/sumup")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('\033[93mSUMUP: Request received at: ' + str(time))
    print('SUMUP: Question is: ' + request.question)
    print('SUMUP: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    # search_graph.folders = sources.split(',')
    result = search_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('\033[93mSUMUP: Answered in ' + str(datetime.utcnow() - time))
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

@app.post("/search")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('\033[93mSUMUP: Request received at: ' + str(time))
    print('SUMUP: Question is: ' + request.question)
    print('SUMUP: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    # search_graph.folders = sources.split(',')
    result = search_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('\033[93mSUMUP: Answered in ' + str(datetime.utcnow() - time))
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

@app.post("/analyze")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('\033[93mANALYZE: Request received at: ' + str(time))
    print('ANALYZE: Question is: ' + request.question)
    print('ANALYZE: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    # analysis_graph.folders = sources.split(',')
    result = analysis_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('\033[93mANALYZE: Answered in ' + str(datetime.utcnow() - time))
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost'] }
