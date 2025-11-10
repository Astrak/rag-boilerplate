# Multi-app RAG server

This repo allows to build RAG solutions for several applications.
It has two parts:

- one to scrape a set of defined websites (crawls+indexes pages)
- one to serve and endpoint for the RAG chatbot.

It is designed to run on a lightweight server (EC2 t3micro), so the indices are split and read incrementally.

## Install

Run with python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -m requirements.txt
pip install -e .
```

## Server Start

The environment requires the following keys:

- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`
- `GOOGLE_API_KEY`
- `TELEGRAM_BOT_TOKEN`

To start the telegram bot:

```bash
nohup python3 app.telegram.main &
```

It should run at `@GweltazBot`.

To start the web api:

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

Verify launch with `curl http://16.171.231.107:8000`: it should answer with `"hello world"`

To query the API:

```bash
curl -X POST \
-H "Content-Type: application/json" \
-d '{"question": "Is macron gay?"}' \
http://16.171.231.107:8000/search
```

## Debug

Clear the following:
`pip install -m requirements.txt`

## Scraper

The scraper is not entirely automatized so one can vet if the data is correctly picked from the targeted website.

It is built to have an associated S3 bucket storing the url lists.

The user must separately:

- discover valid article urls of the target,
- download articles texts,
- then vectorize.

### Discover URLs for a given website

Use the `UrlDiscoverer` and wait for it to complete. On a blog of 5000 articles it takes 2 to 3 hours.

```bash
BASE_URL='https://mywebsite.com' python3 -m scraper.discover_urls.main > scraped_urls.csv
```

Then double check scraped_urls.csv, and make sure to review the following:

- every line is a valid URL of an article to be stored and vectorized.
- no empty lines remain
- remove .jpg, .pdf, .xlsx and other links that are not web pages
- remove pages that are not articles, like contact form etc (put it in EXCLUDED_PATHS beforehand)

Store this file preciously. When updating the data, the easiest will be to append the last articles of the target to this file if feasible.

### Record all articles

Use the `ArticleDiscoverer` and wait for it to complete.

The result is a file `scraped-articles.pkl.gz` in the relative path assigned in `FOLDER`.

Before starting, make sure that the title selector is correct and the metadata selectors too.

Turn on and off `LOG_ARTICLES` to log what the saved articles actually look like.

```bash
FOLDER='./my-folder/' ARTICLE_SELECTOR='#article-content' LOG_ARTICLES='true' python3 -m scraper.download_articles.main
```

### Vectorize

Use the `Vectorizer` and wait for it to complete. It takes a few minutes. Make sure the `FOLDER` in the `main.py` file is what you expect.

```bash
FOLDER='./my-folder/' python3 -m scraper.vectorizer.main
```
