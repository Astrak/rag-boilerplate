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
from enum import Enum, auto
from datetime import datetime
import time
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
        s3 = boto3.resource("s3", region_name="eu-north-1")
        bucket = s3.Bucket("rag-faiss-index-bucket")
        files = []
        for obj in bucket.objects.all():
            file = obj.key
            files.append(file)
        for file in files:
            if file.endswith('/') or ".pdf" or any(sub in file for sub in [".pdf", "scraped_articles.pkl.gz"]):
                continue
            print(f"Downloading {file} into {os.getcwd()}/knowledge-sources/{file}...")
            os.makedirs(os.path.dirname(f"{os.getcwd()}/knowledge-sources/{file}"), exist_ok=True)
            s3.download_file(
                Bucket="rag-faiss-index-bucket", 
                Filename=f"knowledge-sources/{file}", 
                Key=file
            )
except Exception as e:
    raise ValueError('Didnt understand what to do with knowledge sources', e)

analyze_prompt = get_analyze_prompt()

graph = Graph(analyze_prompt, sources)
graph.preload_indices()

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

class AnswerSize(str, Enum):
    SMALL = "170" 
    MEDIUM = "400"
    BIG = "800"

class SearchRequest(BaseModel):
    question: str
    sources: Optional[List[str]] = sources
    answerSize: Optional[AnswerSize] = AnswerSize.SMALL

class Resource(TypedDict):
    url: str
    title: str

@app.post("/retrieve")
def search(request: SearchRequest):
    start_time = time.perf_counter()
    print('\033[93mRETRIEVE: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('RETRIEVE: Question is: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('RETRIEVE: Sources are: ' + ", ".join(request.sources))
    # analysis_graph.folders = sources.split(',')
    result = graph.retrieve({'question': request.question, 'discussion': ''}) 
    resources: list[Resource] = []
    for doc in result['context']:
        resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
    print(f'\033[93mRETRIEVE: Answered in {((time.perf_counter() - start_time) * 1_000):.0f}ms')
    return {"resources": resources}

@app.post("/sumup")
def search(request: SearchRequest):
    start_time = time.perf_counter()
    print('\033[93mSUMUP: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('SUMUP: Question is: ' + request.question)
    print('SUMUP: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    # search_graph.folders = sources.split(',')
    result = graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print(f'\033[93mSUMUP: Answered in {((time.perf_counter() - start_time) * 1_000):.0f}ms')
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

@app.post("/search")
def search(request: SearchRequest):
    start_time = time.perf_counter()
    print('\033[93mSUMUP: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('SUMUP: Question is: ' + request.question)
    print('SUMUP: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    # search_graph.folders = sources.split(',')
    result = graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print(f'\033[93mSUMUP: Answered in {((time.perf_counter() - start_time) * 1_000):.0f}ms')
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

@app.post("/analyze")
def search(request: SearchRequest):
    start_time = time.perf_counter()
    print('\033[93mANALYZE: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('ANALYZE: Question is: ' + request.question)
    print('ANALYZE: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    # analysis_graph.folders = sources.split(',')
    result = graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print(f'\033[93mANALYZE: Answered in {((time.perf_counter() - start_time) * 1_000):.0f}ms')
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost'] }
