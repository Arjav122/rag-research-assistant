from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    qdrant_url: str
    database_url: str
    secret_key: str
    qdrant_collection: str = "ai_research_chunks"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4.1-mini"

    # Ingestion / retrieval stability (tune via .env as you scale)
    arxiv_page_size: int = 100
    arxiv_inter_request_delay_seconds: float = 3.0
    ingest_min_document_chars: int = 200
    ingest_min_chunk_chars: int = 40
    chunk_size: int = 900
    chunk_overlap: int = 150
    embedding_batch_size: int = 64
    qdrant_upsert_batch_size: int = 128

    # Retrieval tuning (hybrid + RRF + diversity; does not replace architecture)
    retrieval_prefetch_multiplier: int = 8
    retrieval_rrf_k: int = 60
    # Cross-encoder reranking every candidate is CPU-heavy (minutes on large pools without GPU).
    retrieval_rerank_pool_cap: int = 32
    retrieval_max_chunks_per_paper_prefetch: int = 3
    retrieval_final_max_per_paper: int = 2
    retrieval_metadata_boost_cap: float = 0.08

    # Tier 1 retrieval intelligence (each gated by a flag so it's safely disablable)
    # Corpus-wide BM25 scrolls all chunks (~40k+) into RAM on build; on CPU Docker this can
    # add tens of seconds per cold build. Enable in .env when you need maximum lexical recall.
    retrieval_use_corpus_bm25: bool = True
    retrieval_corpus_bm25_top_k: int = 50
    retrieval_corpus_bm25_refresh_seconds: int = 3600
    retrieval_corpus_bm25_max_chunks: int = 80000  # safety cap on in-memory index size

    retrieval_use_query_rewrite: bool = True
    retrieval_query_rewrite_cache_size: int = 256

    retrieval_use_hyde: bool = True
    retrieval_hyde_max_tokens: int = 6  # only HyDE when query is short
    retrieval_hyde_max_chars: int = 48
    # Optional lightweight expansion for acronym/method/benchmark heavy queries.
    # Keep disabled by default; enable only if eval shows measurable precision gain.
    retrieval_use_technical_expansion: bool = False

    retrieval_use_section_intent: bool = True
    retrieval_section_intent_boost: float = 0.05

    # bge-reranker-large emits sigmoid-like scores in [~0, ~1]. Empirical probe across
    # this corpus (407 papers): healthy in-domain queries top 0.75-1.0; truly off-domain
    # ("Tokyo population", "chocolate cake") top <0.25. Tangentially-relevant off-domain
    # queries (e.g. CRISPR vs a corpus with retinal/biology papers) can still score high —
    # the threshold can't detect those, but the LLM correctly disclaims and a separate
    # disclaimer-aware citation suppressor handles that case downstream.
    retrieval_low_confidence_threshold: float = 0.5
    retrieval_use_confidence_guardrail: bool = True

    # Recommendations: nudge papers whose title/snippet lexically matches query tokens
    # (reduces "tangentially related" items outranking narrow-topic papers).
    recommendation_lexical_boost_weight: float = 0.18


@lru_cache
def get_settings() -> Settings:
    return Settings()
