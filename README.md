# RAG API + Telegram Chatbot

In this app I am building my own RAG application. It was initially a playground but eventually made it into a full blown app with Telegram bot and API endpoint for a customer's implementation on a website.

One specific feature is that I put that on a free EC2 instance with 2GB RAM and I didn't want to resort to Pinecone or Chroma for managing the data just because of it.
The customer's data was not much but having the full vector store at runtime in RAM wasn't cutting it, as LangChain was designed to do. So I went for my own implementation of the RAG that splits the vectors in chunks and does the similarity search step by step on the chunks. It takes around 100ms instead of 10. This is not a problem since the answer's generation time rather meets its bottleneck in waiting for LLMs to generate it.

## Capabilities

1. Discover all URLs for a given website when a sitemap is not available
1. Scrape the website
1. Vectorize the DB
1. Handle chunked vector batches and chunked indices.
1. chunked RAG
1. Node-based code with LangChain
