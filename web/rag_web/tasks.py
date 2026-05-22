import json
import time
import traceback
from datetime import datetime, timezone

from celery import shared_task
from django.conf import settings
import pandas as pd

from agentic.retriever.create_index import create_unified_corpus
from agentic.retriever.hybrid_retriever import HybridRetriever
from agentic.retriever.reranker import Reranker
from agentic.models.expansion_query import QueryExpansion
from pathlib import Path


# _unified_corpus_cache = None
# _reranker_cache       = None
# _law_df_cache         = None
# _court_df_cache       = None


# def _get_data():
#     global _law_df_cache, _court_df_cache
#     if _law_df_cache is None:
#         base_dir = settings.BASE_DIR
#         PRJECT_ROOT = base_dir.parent
#         #artifacts  = settings.ARTIFACTS_DIR
#         root = PRJECT_ROOT
#         law_path   = root/"artifacts" /"chunks" / "laws_chunks.parquet"
#         court_path = root/"artifacts" / "chunks" / "court_chunks.parquet"
#         _law_df_cache   = pd.read_parquet(law_path)
#         _court_df_cache = pd.read_parquet(court_path)
#         if getattr(settings, "RAG_DEV_MODE", False):
#             n = getattr(settings, "RAG_DEV_ROWS", 500)
#             _law_df_cache   = _law_df_cache.head(n)
#             _court_df_cache = _court_df_cache.head(n // 2)
#     return _law_df_cache, _court_df_cache


# def _get_unified_corpus(law_df, court_df):
#     global _unified_corpus_cache
#     if _unified_corpus_cache is None:
#         _unified_corpus_cache = create_unified_corpus(law=law_df, court=court_df)
#     return _unified_corpus_cache


# def _get_reranker():
#     global _reranker_cache
#     if _reranker_cache is None:
#         _reranker_cache = Reranker(
#             use_stage2=getattr(settings, "RAG_USE_STAGE2_RERANKER", False)
#         )
#     return _reranker_cache

#_pipeline_cache = None
def _get_pipeline():
    """
    Get or build the RAGPipeline instance.
    Cached at module level — survives across multiple Celery task calls
    within the same worker process.
    """
    #_pipeline_cache = None
    #global _pipeline_cache
    _pipeline_cache = None
    if _pipeline_cache is None:
        from agentic.pipeline.rag_pipeline import RAGPipeline

        artifacts_dir = settings.ARTIFACTS_DIR

        _pipeline_cache = RAGPipeline(
            hybrid_top_k        = settings.RAG_HYBRID_TOP_K,
            reranker_top_k      = settings.RAG_RERANKER_TOP_K,
            use_stage2_reranker = settings.RAG_USE_STAGE2_RERANKER,
            save_dir            = str(artifacts_dir),
            dev_mode            = settings.RAG_DEV_MODE,
            dev_rows            = settings.RAG_DEV_ROWS,
        )
        # startup() loads data + builds FAISS + loads reranker models
        _pipeline_cache.startup()

    return _pipeline_cache




@shared_task(bind=True, name="rag_web.tasks.run_rag_query")
def run_rag_query(self, query_run_id: int, query_text: str):
    """
    Main Celery task — runs the full RAG pipeline for one query.
    Fetches QueryRun by task_id (self.request.id) to avoid race conditions.
    """
    from rag_web.models import QueryRun

    # Fetch by task_id — avoids the race condition where query_run_id
    # is not yet saved when the task starts
    task_id   = self.request.id
    query_run = None

    for _ in range(10):
        try:
            query_run = QueryRun.objects.get(task_id=task_id)
            break
        except QueryRun.DoesNotExist:
            time.sleep(0.3)

    if query_run is None:
        raise RuntimeError(f"QueryRun not found for task_id={task_id}")

    start = time.time()

    try:
        # ── STARTED ───────────────────────────────────────────────────────
        query_run.status = QueryRun.Status.STARTED
        query_run.save(update_fields=["status"])
        self.update_state(state="STARTED", meta={"step": "Loading pipeline...", "progress": 10})

        # ── Get cached pipeline ───────────────────────────────────────────
        pipeline = _get_pipeline()
        self.update_state(state="STARTED", meta={"step": "Expanding query...", "progress": 35})

        # ── Build single-row val_df for this query ────────────────────────
        # RAGPipeline._process_query() expects a pd.Series with 'query' field
        import pandas as pd
        query_series = pd.Series({
            "id":    query_run.pk,
            "query": query_text,
        })

        self.update_state(state="STARTED", meta={"step": "Retrieving citations...", "progress": 55})

        # ── Run single query through pipeline ─────────────────────────────
        result = pipeline._process_query(query_series)

        self.update_state(state="STARTED", meta={"step": "Reranking...", "progress": 80})

        citations    = result["citations"]
        expanded     = result.get("expanded", "")
        top_results  = result.get("top_results")

        # Serialize results DataFrame to JSON
        results_records = []
        if top_results is not None and not top_results.empty:
            records = top_results.to_dict("records")
            for rec in records:
                for k, v in rec.items():
                    if hasattr(v, "item"):
                        rec[k] = v.item()
            results_records = records

        self.update_state(state="STARTED", meta={"step": "Saving results...", "progress": 92})

        # ── Save to DB ────────────────────────────────────────────────────
        elapsed = time.time() - start
        query_run.status         = QueryRun.Status.SUCCESS
        query_run.expanded_query = expanded
        query_run.citations_json = json.dumps(citations)
        query_run.results_json   = json.dumps(results_records)
        query_run.completed_at   = datetime.now(timezone.utc)
        query_run.elapsed_secs   = elapsed
        query_run.save()

        return {"status": "SUCCESS", "citations": citations, "elapsed": elapsed}

    except Exception as exc:
        elapsed = time.time() - start
        query_run.status        = QueryRun.Status.FAILURE
        query_run.error_message = f"{str(exc)}\n\n{traceback.format_exc()}"
        query_run.elapsed_secs  = elapsed
        query_run.completed_at  = datetime.now(timezone.utc)
        query_run.save()
        raise exc

# @shared_task(bind=True, name="rag_web.tasks.run_rag_query")   # fixed: was dashboard.tasks
# def run_rag_query(self, query_run_id: int, query_text: str):
#     from rag_web.models import QueryRun   # fixed: was dashboard.models

#     query_run = QueryRun.objects.get(pk=query_run_id)
#     start = time.time()

#     try:
#         query_run.status = QueryRun.Status.STARTED
#         query_run.save(update_fields=["status"])
#         self.update_state(state="STARTED", meta={"step": "Loading data...", "progress": 10})

#         law_df, court_df = _get_data()
#         self.update_state(state="STARTED", meta={"step": "Building index...", "progress": 25})

#         unified = _get_unified_corpus(law_df, court_df)
#         self.update_state(state="STARTED", meta={"step": "Expanding query...", "progress": 40})

#         val_df = pd.DataFrame([{"id": query_run_id, "query": query_text}])
#         expander = QueryExpansion()
#         expander.val_df = val_df
#         expanded_query  = expander.Eng_plus_Germ()

#         query_run.expanded_query = expanded_query
#         query_run.save(update_fields=["expanded_query"])
#         self.update_state(state="STARTED", meta={"step": "Retrieving candidates...", "progress": 55})

#         retriever = HybridRetriever(
#             law_df         = law_df,
#             val_df         = val_df,
#             unified_corpus = unified,
#             expanded_query = expanded_query,
#             query_text     = query_text,
#         )
#         candidates = retriever.retrieve(top_k=getattr(settings, "RAG_HYBRID_TOP_K", 100))
#         self.update_state(state="STARTED", meta={"step": "Reranking results...", "progress": 75})

#         reranker = _get_reranker()
#         reranked = reranker.rerank(
#             query      = expanded_query,
#             candidates = candidates,
#             top_k      = getattr(settings, "RAG_RERANKER_TOP_K", 10),
#         )
#         citations = reranker.get_citations(reranked)
#         self.update_state(state="STARTED", meta={"step": "Saving results...", "progress": 90})

#         results_records = reranked.to_dict("records")
#         for rec in results_records:
#             for k, v in rec.items():
#                 if hasattr(v, "item"):
#                     rec[k] = v.item()

#         elapsed = time.time() - start
#         query_run.status         = QueryRun.Status.SUCCESS
#         query_run.citations_json = json.dumps(citations)
#         query_run.results_json   = json.dumps(results_records)
#         query_run.completed_at   = datetime.now(timezone.utc)
#         query_run.elapsed_secs   = elapsed
#         query_run.save()

#         return {"status": "SUCCESS", "citations": citations, "elapsed": elapsed}

#     except Exception as exc:
#         elapsed = time.time() - start
#         query_run.status        = QueryRun.Status.FAILURE
#         query_run.error_message = f"{str(exc)}\n\n{traceback.format_exc()}"
#         query_run.elapsed_secs  = elapsed
#         query_run.completed_at  = datetime.now(timezone.utc)
#         query_run.save()
#         raise exc
