import json
from apps.api.analyze_prompt import get_analyze_prompt
from apps.api.env import fill_env
from graph.main import Graph
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional, List
from typing_extensions import TypedDict
from apps.api.ip_throttler_middleware import IPThrottleMiddleware
from enum import Enum
from datetime import datetime
import time
import os
import boto3

fill_env()

bucket_name = os.getenv("BUCKET")
if not bucket_name:
    raise EnvironmentError('No BUCKET variable specified')
os.environ['BUCKET'] = bucket_name
print("Bucket: ", bucket_name)

bucket_region = os.getenv("BUCKET_REGION")
if not bucket_region:
    raise EnvironmentError('No BUCKET_REGION variable specified')
os.environ['BUCKET_REGION'] = bucket_region
print("Bucket region: ", bucket_region)

try:
    sync = input("Sync knowledge-sources from bucket? (y/n): ").lower().startswith('y')
    if sync:
        s3 = boto3.resource("s3", region_name=bucket_region)
        bucket = s3.Bucket(bucket_name)
        files = []
        for obj in bucket.objects.all():
            file = obj.key
            files.append(file)
        print(f"Files to download from bucket: {files}")
        for file in files:
            if file.endswith('/') or any(sub in file for sub in [".pdf", "scraped_articles.pkl.gz"]):
                print(f"Skipping {file}")
                continue
            print(f"Downloading {file} into {os.getcwd()}/knowledge-sources/{file}...")
            os.makedirs(os.path.dirname(f"{os.getcwd()}/knowledge-sources/{file}"), exist_ok=True)
            bucket.download_file(
                Filename=f"knowledge-sources/{file}", 
                Key=file
            )
except Exception as e:
   raise ValueError('Didnt understand what to do with knowledge sources', e)

sources = os.getenv("SOURCES")
if not sources:
    print("No SOURCES variable specified, using all sources in the knowledge-sources folder:")
    sources = os.listdir("./knowledge-sources/")
else:
    sources = sources.split(',')
print("Running RAG from sources: ", sources)

analyze_prompt = get_analyze_prompt()

graph = Graph(analyze_prompt, sources)
graph.preload_indices()

allowed_origins = [
    "https://polemia.surge.sh",
    "https://ia.polemia.com",
    "https://lovable.app",
    "https://lovable.dev",
    "https://lovableproject.com",
    "https://iapolemia.lovable.app",
    "https://iapolemia2.lovable.app"
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
    SMALL = "1" 
    MEDIUM = "2"
    BIG = "3"

class SearchRequest(BaseModel):
    question: str
    sources: Optional[List[str]] = sources
    answerSize: Optional[AnswerSize] = AnswerSize.SMALL

class Resource(TypedDict):
    url: str
    title: str

BASIC_CACHE = {}

@app.post("/retrieve")
def retrieve(request: SearchRequest):
    start_time = time.time()
    print('\033[93mRETRIEVE: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('RETRIEVE: Question is: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('RETRIEVE: Sources are: ' + ", ".join(request.sources))
    graph.folders = sources.split(',')
    result = graph.retrieve({'question': request.question, 'discussion': ''}) 
    resources: list[Resource] = []
    for doc in result['context']:
        resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
    print(f'\033[93mRETRIEVE: Answered in {((time.time() - start_time) * 1_000):.0f}ms')
    return {"resources": resources}

@app.post("/analyze")
async def analyze(request: SearchRequest):
    start_time = time.time()
    print('\033[93mANALYZE: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('ANALYZE: Question is: ' + request.question)
    print('ANALYZE: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    answer_size = 160 + (int(request.answerSize) - 1) * 150
    print(f'ANALYZE: Answer size requested is: {AnswerSize(request.answerSize)}')
    graph.folders = sources.split(',')
    cache_string = f"{request.question}&{AnswerSize(request.answerSize)}"
    if cache_string in BASIC_CACHE:
        print(f'\033[93mANALYZE: Answered from cache')
        return BASIC_CACHE[cache_string]
    graph.prompt = get_analyze_prompt(answer_size)
    result = await graph.ainvoke(request.question) 
    print(f'\033[93mANALYZE: Answered in {((time.time() - start_time) * 1_000):.0f}ms')
    answer = {"results": result['answer'], "resources": result['resources'], "cost": result['cost'] }
    BASIC_CACHE[cache_string] = answer
    return answer

@app.post("/stream-analyze")
async def stream_analyze(request: SearchRequest):
    start_time = time.time()
    print('\033[93mSTREAM-ANALYZE: Request received at: ' + str(datetime.fromtimestamp(start_time)))
    print('STREAM-ANALYZE: Question is: ' + request.question)
    print('STREAM-ANALYZE: Sources are: ' + ", ".join(request.sources))
    sources = ",".join([f"./{src}/" for src in request.sources])
    answer_size = 160 + (int(request.answerSize) - 1) * 150
    print(f'STREAM-ANALYZE: Answer size requested is: {AnswerSize(request.answerSize)}')
    graph.folders = sources.split(',')
    cache_string = f"{request.question}&{AnswerSize(request.answerSize)}"
    graph.prompt = get_analyze_prompt(answer_size)
    async def event_generator():

        if cache_string in BASIC_CACHE:
            yield f"data: {json.dumps({ "type": "resources", "resources": BASIC_CACHE[cache_string]['resources']  })}\n\n"
            yield f"data: {json.dumps({ "type": "token", "data": BASIC_CACHE[cache_string]['results'] })}\n\n"
            yield f"data: {json.dumps({ "type": "cost", "cost": BASIC_CACHE[cache_string]['cost'] })}\n\n"
            yield f"data: {json.dumps({ "type": "done" })}\n\n"
            print(f'\033[93mSTREAM-ANALYZE: Answered from cache')
            return  
          
        text = ""
        resources: list[Resource] = []
        cost = ""
        try:            
            async for event in graph.astream_events(request.question):
                kind = event["event"]
                if kind == "on_chain_start":
                    if event["data"].get('input') and event['data']['input'].get('context'):
                        print('SENDING RESOURCES')
                        for doc in event["data"]["input"]['context']:
                            resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
                        yield f"data: {json.dumps({ "type": "resources", "resources": resources })}\n\n"
                    else:
                        print("ERROR input not found")
                        yield ""
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    text += content
                    if content:
                        yield f"data: {json.dumps({ "type": "token", "data": content })}\n\n"
                if kind == "on_chain_end":
                    if event['data'].get('output') and event['data']['output'].get('cost'):
                        cost = event['data']['output']['cost']
                        yield f"data: {json.dumps({ "type": "cost", "cost": event['data']['output']['cost'] })}\n\n"
            yield f"data: {json.dumps({"type": "done"})}\n\n"
        finally:
            answer = {"results": text, "resources": resources, "cost": cost }
            BASIC_CACHE[cache_string] = answer
            print(f'\033[93mSTREAM-ANALYZE: Answered in {((time.time() - start_time) * 1_000):.0f}ms')
            pass
    return StreamingResponse(event_generator(), media_type="text/event-stream")

    #     yield "data: [DONE]\n\n"
    # async def event_generator():
    #     async for ev in graph.astream_with_tokens(request.question):
    #         if ev.get("done"):
    #             yield f"data: {json.dumps({'full': ev['full_answer'], 'otherData': ev['other_data']})}\n\n"
    #             yield "data: [DONE]\n\n"
    #         elif ev.get("delta"):
    #             yield f"data: {ev['delta']}\n\n"

    # return StreamingResponse(
    #     event_generator(),
    #     media_type="text/event-stream",
    #     headers={
    #         "Cache-Control": "no-cache",
    #         "X-Accel-Buffering": "no"   # important for nginx reverse proxy
    #     }
    # )
