# Multi-app RAG server

This repo allows to build RAG solutions for several applications.
It has two parts:

- one to scrape a set of defined websites (crawls+indexes pages)
- one with the RAG chatbots for X, telegram, and one for web-based single-answers interfaces.

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
- `XAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`

### Launching apps

#### Telegram:

```bash
FOLDERS='./my-folder1/,./my-folder-2/' nohup python3 app.telegram.main &
```

It should run at `@GweltazBot`.

#### Endpoint

```bash
FOLDERS='./my-folder1/,./my-folder-2/' uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

Verify launch with `curl http://16.171.231.107:8000`: it should answer with `"hello world"`

To query the API there are two endpoints:

- `/search` offers an overview for keyword searches as in Google Overview, limited to 120 words:

```bash
curl -X POST \
-H "Content-Type: application/json" \
-d '{"question": "Soros"}' \
http://16.171.231.107:8000/search
```

- `/analyze` offers an in-depth analysis on a given keyword or question:

```bash
curl -X POST \
-H "Content-Type: application/json" \
-d '{"question": "Je veux un résumé sur la situation des sans-papiers en 2021"}' \
http://16.171.231.107:8000/analyze
```

## Debug

Clear the following:
`pip install -m requirements.txt`

## Building the knowledge base

The scraper is not entirely automatized so one can vet if the data is correctly picked from the targeted website.

It is built to have an associated S3 bucket storing the url lists.

The user must separately:

- discover valid article urls of the target,
- download articles texts or extract PDFs,
- then vectorize.

### Discover data

The knowledge is stored in an untracked folder `./knowledge-sources/` at the source of the repo. Every online source knowledge mut be identified by its public domain: `my-website.com`.

#### Scrape website URLs for a given website

To update all `url-list.csv` of all knowledge sources, just do:

```bash
python3 -m scraper.discover_urls.main
```

To scrawl for specific sources: target specific sources, `SOURCES="my-website.com,website2.com"` for instance (comma seprated). Indexes are created or updated for each source in a file at `<source>/url-list.csv`. A blacklist can be added of specific undesired URLs at `<source>/blacklist.csv`. Wait for it to complete. On a blog of 5000 articles it takes 2 to 3 hours.

```bash
SOURCES='my-website.com' python3 -m scraper.discover_urls.main
```

Then double check `my-website.com/url-list.csv`, and make sure to review the following:

- every line is a valid URL of an article to be stored and vectorized.
- no empty lines remain
- remove .jpg, .xlsx and other links that are not web pages or PDFs. These paths are handled in `main.py` in an `excluded_paths` variable.
- remove pages that are not articles, like contact forms, articles lists etc.

Store this file preciously. When updating the data, the easiest will be to append the last articles of the target to this file if feasible.

#### Accumulate data locally (PDFs for now)

Files can be read locally too, only PDFs for now. If a PDF is online, starting with `https://`, it can be kept in the `url-list.csv` above, it will be handled as such by the `ArticleDiscoverer` in the next step. But one can also create a local knowledge base of offline data, which the final LLM's output won't be able to give a link to, but only a reference of edition. For this, create a folder `local-knowledge` within your domain folder, and just put the PDF there with the name you wish, like `my-pdf.pdf`.

### Record documents

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
