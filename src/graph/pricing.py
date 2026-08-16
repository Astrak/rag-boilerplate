"""Centralized pricing for the models the RAG graph calls, and the
token-counting/cost-estimation helpers built on top of it.

Rates are USD per token, derived from each provider's published
USD-per-million-token price. They're a point-in-time snapshot for
*estimating* spend (e.g. surfacing per-answer cost in the API/UI) — not a
billing-grade source of truth. Re-check against the provider pricing pages
below before relying on them for anything stricter than an estimate:
  - xAI (chat):        https://docs.x.ai/developers/pricing
  - OpenAI (embeddings): https://platform.openai.com/docs/pricing
"""

from typing import TypedDict


class ModelRate(TypedDict):
    input_per_million: float
    output_per_million: float


# grok-3 is what graph/main.py's ChatXAI instance currently runs; grok-4 is
# included because apps/telegram and apps/x use it directly via
# init_chat_model. Both are priced the same as of writing ($3/$15 per
# million input/output tokens).
CHAT_MODEL_RATES: dict[str, ModelRate] = {
    "grok-3": {"input_per_million": 3.00, "output_per_million": 15.00},
    "grok-4": {"input_per_million": 3.00, "output_per_million": 15.00},
}
DEFAULT_CHAT_MODEL = "grok-3"

# Embeddings only have an input side (no generated tokens to price).
EMBEDDING_MODEL_RATES: dict[str, float] = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def chars_to_tokens_approx(text: str) -> int:
    """Rough ~4-chars-per-token fallback, used only when a provider response
    doesn't carry actual usage counts."""
    return len(text) // 4 + 1


class CostDetail(TypedDict):
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    estimated: bool  # True when token counts are approximated rather than provider-reported


def _chat_rate(model: str) -> ModelRate:
    rate = CHAT_MODEL_RATES.get(model)
    if rate is not None:
        return rate
    # Unknown/new model: fall back to the priciest known rate so an
    # unrecognized model undercounts cost rather than silently reads as free.
    return max(CHAT_MODEL_RATES.values(), key=lambda r: r["input_per_million"] + r["output_per_million"])


def _embedding_rate(model: str) -> float:
    rate = EMBEDDING_MODEL_RATES.get(model)
    if rate is not None:
        return rate
    return max(EMBEDDING_MODEL_RATES.values())


def estimate_chat_cost(input_tokens: int, output_tokens: int, model: str, *, estimated: bool = False) -> CostDetail:
    rate = _chat_rate(model)
    input_cost = input_tokens * rate["input_per_million"] / 1_000_000
    output_cost = output_tokens * rate["output_per_million"] / 1_000_000
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "estimated": estimated,
    }


def estimate_embedding_cost(tokens: int, model: str, *, estimated: bool = False) -> CostDetail:
    rate_per_million = _embedding_rate(model)
    input_cost = tokens * rate_per_million / 1_000_000
    return {
        "model": model,
        "input_tokens": tokens,
        "output_tokens": 0,
        "input_cost": input_cost,
        "output_cost": 0.0,
        "total_cost": input_cost,
        "estimated": estimated,
    }


class CostBreakdown(TypedDict):
    embedding: CostDetail
    generation: CostDetail
    total_cost: float
    currency: str


def combine_costs(embedding: CostDetail, generation: CostDetail) -> CostBreakdown:
    return {
        "embedding": embedding,
        "generation": generation,
        "total_cost": embedding["total_cost"] + generation["total_cost"],
        "currency": "USD",
    }


def embedding_only_breakdown(embedding: CostDetail) -> CostBreakdown:
    """Wraps a standalone embedding cost (e.g. the /retrieve endpoint, which
    never calls the chat model) in the same CostBreakdown shape /analyze and
    /stream-analyze return, so API consumers can rely on one response shape."""
    no_generation: CostDetail = {
        "model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "estimated": False,
    }
    return combine_costs(embedding, no_generation)
