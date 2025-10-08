# Multi-app RAG server

This repo allows to build RAG solutions for several applications.
It has two parts:

- one to regularly scrape a set of defined websites (crawls+indexes pages)
- one to serve as server and endpoint for the RAG chatbot.

It is designed to run on a lightweight server (EC2 t3micro), so the indices are split and read incrementally.
The indices are sychronized on an S3, which is updated when a new web scraping is done.
The server regularly updates the local indices downstream.

## Server Start

```py
nohup python3 app/main.py &
```
