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

folders = os.getenv("FOLDERS")
if not folders:
    raise EnvironmentError("FOLDERS not found. Run with FOLDERS='./folder1/,./folder2/,./folder3/'")
folders_list = [folder_name.strip() for folder_name in folders.split(",")]

DEFAULT_SOURCES = [folder_name.removeprefix('./').removesuffix('/') for folder_name in folders.split(',')]

print('Using following knowledge folders for RAG: ' + ','.join(folders_list))

fill_env()

search_prompt = get_search_prompt()
analyze_prompt = get_analyze_prompt()

search_graph = Graph(search_prompt, folders_list)
analysis_graph = Graph(analyze_prompt, folders_list)

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
    sources: Optional[List[str]] = DEFAULT_SOURCES
    llm: str

class Resource(TypedDict):
    url: str
    title: str

@app.post("/retrieve")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('Request received at: ' + str(time))
    print('Request type is RETRIEVE: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('Request for sources: ' + ", ".join(request.sources))
    analysis_graph.folders = sources.split(',')
    result = analysis_graph.retrieve({'question': request.question, 'discussion': ''}) 
    resources: list[Resource] = []
    for doc in result['context']:
        resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
    print('Sources RETRIEVE found in ' + str(datetime.utcnow() - time))
    return {"resources": resources}

@app.post("/sumup")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('Request received at: ' + str(time))
    print('Request type is SUMUP: ' + request.question)
    print('Request for sources: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    search_graph.folders = sources.split(',')
    result = search_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('Request SUMUP answered in ' + str(datetime.utcnow() - time))
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

@app.post("/search")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('Request received at: ' + str(time))
    print('Request type is SUMUP: ' + request.question)
    print('Request for sources: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    search_graph.folders = sources.split(',')
    result = search_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('Request ANALYZE answered in ' + str(datetime.utcnow() - time))
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

@app.post("/analyze")
def search(request: SearchRequest):
    time = datetime.utcnow()
    print('Request received at: ' + str(time))
    print('Request type is ANALYZE: ' + request.question)
    print('Request for sources: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    analysis_graph.folders = sources.split(',')
    result = analysis_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('Request answered in ' + str(datetime.utcnow() - time))
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost'] }
