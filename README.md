# Multi-app RAG server

This repo allows to build RAG solutions for several applications.
It has two parts:

- one to scrape a set of defined websites (crawls+indexes pages)
- one with the RAG chatbots for X, telegram, and one for web-based single-answers interfaces.

It is designed to run on a lightweight server (EC2 t3micro), so the indices are split and read incrementally.

## Install

Run with python `3.12`:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Server Start

The environment requires the following keys:

- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`

### Launching apps

#### Telegram:

```bash
FOLDERS='./my-folder1/,./my-folder-2/' nohup python app.telegram.main &
```

It should run at `@GweltazBot`.

#### Endpoint

```bash
FOLDERS='./my-folder1/,./my-folder-2/' uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

Verify launch with `curl http://localhost:8000`: it should answer with `"hello world"`

To query the API there are two endpoints:

- `/retrieve` offers a list of matching sources as in a Google Search:

```bash
curl -X POST \
-H "Content-Type: application/json" \
-d '{"question": "Soros"}' \
http://localhost:8000/search
```

- `/analyze` offers an in-depth analysis on a given keyword or question:

```bash
curl -X POST \
-H "Content-Type: application/json" \
-d '{"question": "Je veux un résumé sur la situation des sans-papiers en 2021"}' \
http://localhost:8000/analyze
```

## Building the knowledge base

The scraper is not entirely automatized so one can vet if the data is correctly picked from the targeted website.

It is built to have an associated S3 bucket storing the url lists.

The user must separately:

- discover valid article urls of the target,
- download articles texts or extract PDFs,
- then vectorize.

### Discover data

The knowledge is stored in an untracked folder `./knowledge-sources/` at the source of the repo. Every online source knowledge must be identified as folder named by its public domain: `my-website.com`.

#### Scrape website URLs for a given website

To update all `url-list.csv` of all knowledge sources, just do:

```bash
python -m scraper.discover_urls.main
```

To scrawl for specific sources: target specific sources, `SOURCES="my-website.com,website2.com"` for instance (comma separated). Indexes are created or updated for each source in a file at `<source>/url-list.csv`. A blacklist can be added of specific URLs that may be visited but not to record, at `<source>/blacklist.csv`, and URLs or patterns that must not be explored are to be listed in `<source>/ignore.csv` (like `/tag/`). It may be long to build the initial URL list but updating is optimized.

```bash
SOURCES='my-website.com' python -m scraper.discover_urls.main
```

Then double check `my-website.com/url-list.csv`, and make sure to review the following:

- every line is a valid URL of an article to be stored and vectorized.
- no empty lines remain except the last
- remove pages that are not articles, like contact forms, articles lists etc., put them in `blacklist.csv` or `ignore.csv`

Store this file preciously. When updating the data, the easiest will be to append the last articles of the target to this file if feasible.

#### Accumulate data locally (PDFs for now)

Files can be read locally too, only PDFs for now. If a PDF is online, starting with `https://`, it can be kept in the `url-list.csv` above, it will be handled as such by the `ArticleDiscoverer` in the next step. But one can also create a local knowledge base of offline data, which the final LLM's output won't be able to give a link to, but only a reference of edition. For this, create a folder `local-knowledge` within your domain folder, and just put the PDF there with the name you wish, like `my-pdf.pdf`.

### Record documents

Before starting, make sure `selectors` files are created in the sources' folders. It should look like this:

```
title=h1#my-title
author=div.author-field
date=
article=div#content > article
comment=whatever you need to remind yourself there
```

You'll also need dev credentials from Adobe to handle scraping PDFs.

```bash
ADOBE_CLIENT_ID='XXX' ADOBE_CLIET_SECRET='XXX' SOURCES='source-1,source2.com' LOG_ARTICLES='true' python -m scraper.download_articles.main
```

The result will be a file `<source-folder>/scraped-articles.pkl.gz`.

If you don't provide a `SOURCES` argument, it will loop on all folders in `./knowledge-sources/`.

Turn on and off `LOG_ARTICLES` to check if your selectors are working as expected.

### Vectorize

Make sure you have some budget and launch vectorization:

```bash
SOURCES='website1.com,source2' python -m scraper.vectorizer.main
```

If you don't provide a `SOURCES` argument, it will loop on all folders in `./knowledge-sources/`.
