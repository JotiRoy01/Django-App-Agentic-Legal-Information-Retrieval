import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from agentic.data_loader.data_loader import load
from agentic.models.expansion_query import QueryExpansion
from agentic.retriever.create_index import create_unified_corpus
from agentic.retriever.hybrid_retriever import HybridRetriever
from agentic.retriever.reranker import Reranker
from agentic.exception import Agentic_Exception
from agentic.logger.logging import setup_logging, get_logger

setup_logging()
logger = get_logger()


# setup_logging()
# logger = get_logger()

class RAGPipeline:
    """
    Full end-to-end retrieval pipeline.
 
    Parameters
    ----------
    hybrid_top_k : int
        Number of candidates HybridRetriever returns per query. Default 100.
    reranker_top_k : int
        Number of final results Reranker returns per query. Default 10.
    use_stage2_reranker : bool
        Whether to run the Stage 2 bge-reranker-v2-m3.
        Set False for CPU-only / fast development runs. Default True.
    save_dir : str
        Directory to save submission and logs. Default 'artifacts'.
    """
 
    def __init__(
        self,
        hybrid_top_k: int          = 100,
        reranker_top_k: int        = 10,
        use_stage2_reranker: bool  = True,
        save_dir: str              = "artifacts",
        dev_mode: bool = False,
        dev_rows: int = 500
    ):
        try:
            self.hybrid_top_k         = hybrid_top_k
            self.reranker_top_k       = reranker_top_k
            self.use_stage2_reranker  = use_stage2_reranker
            self.save_dir             = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.dev_mode = dev_mode
            self.dev_rows = dev_rows
 
            # Populated during startup()
            self.law_df       = None
            self.court_df     = None
            self.val_df       = None
            self.unified      = None   # Unified_Corpus — FAISS index for law + court
            self.reranker     = None   # Two-stage Reranker
            self.expander     = None   # QueryExpansion (Qwen2.5)
            
 
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
 
    # ── STARTUP ───────────────────────────────────────────────────────────────
 
    def startup(self):
        """
        Load all data and build all indexes.
        Run ONCE before the query loop.
        Heavy operations: FAISS encoding + reranker model loading.
        """
        try:
            logger.info("=" * 60)
            logger.info("RAG Pipeline Startup")
            logger.info("=" * 60)
            t0 = time.time()
 
            # ── Step 1: Load datasets ─────────────────────────────────────────
            logger.info("[1/4] Loading datasets...")
            # self.law_df   = load(filename="laws_de.csv")
            # self.court_df = load(filename="court_considerations.csv")
            self.law_df   = pd.read_parquet("artifacts/chunks/laws_chunks.parquet")
            self.court_df = pd.read_parquet("artifacts/chunks/court_chunks.parquet")
            self.val_df   = load(filename="val.csv")

            if self.dev_mode:
                self.law_df   = self.law_df.head(self.dev_rows)
                self.court_df = self.court_df.head(self.dev_rows // 2)
                self.val_df   = self.val_df.head(5)   # only 5 queries in dev mode
                logger.info(f"DEV MODE: law={len(self.law_df)} rows, "
                f"court={len(self.court_df)} rows, "
                f"queries={len(self.val_df)}")
 
            logger.info(f"  law_de.csv       : {len(self.law_df):,} rows")
            logger.info(f"  court_considerations.csv : {len(self.court_df):,} rows")
            logger.info(f"  val.csv          : {len(self.val_df):,} queries")
 
            # ── Step 2: Build Unified_Corpus (FAISS for law + court) ──────────
            # This is the slow step — encodes the full corpus ONCE.
            # All per-query FAISS searches reuse this index.
            logger.info("[2/4] Building Unified_Corpus (law + court FAISS index)...")
            logger.info("  This encodes the full corpus — takes several minutes.")
            self.unified = create_unified_corpus(
                law=self.law_df,
                court=self.court_df,
            )
            logger.info(f"  Unified FAISS index size: {self.unified.unified_faiss_index.ntotal:,} vectors")
 
            # ── Step 3: Load Reranker ─────────────────────────────────────────
            logger.info("[3/4] Loading reranker models...")
            self.reranker = Reranker(use_stage2=self.use_stage2_reranker)
 
            # ── Step 4: Initialize QueryExpansion ────────────────────────────
            # QueryExpansion loads val.csv internally.
            # We will override val_df per query using _expand_query().
            logger.info("[4/4] Initializing QueryExpansion (Qwen2.5-1.5B)...")
            self.expander = QueryExpansion()
 
            elapsed = time.time() - t0
            logger.info(f"\nStartup complete in {elapsed:.1f}s")
            logger.info("=" * 60)
 
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
 
    # ── QUERY EXPANSION ───────────────────────────────────────────────────────
 
    def _expand_query(self, query_row: pd.Series) -> str:
        """
        Expand a single query using QueryExpansion.
 
        QueryExpansion.Eng_plus_Germ() is hardcoded to val_df.iloc[0].
        We patch the instance's val_df to a single-row DataFrame containing
        only the current query — so iloc[0] returns the correct query.
        No changes to expansion_query.py needed.
 
        Parameters
        ----------
        query_row : pd.Series — one row from val_df
 
        Returns
        -------
        str — expanded German+English keywords
        """
        # Patch: override the expander's val_df to the current query row only
        self.expander.val_df = pd.DataFrame([query_row])
        expanded = self.expander.Eng_plus_Germ()
        return expanded
 
    # ── SINGLE QUERY ─────────────────────────────────────────────────────────
 
    def _process_query(self, query_row: pd.Series) -> dict:
        """
        Run the full pipeline for one query row from val_df.
 
        Returns
        -------
        dict with keys:
            query_id  : str or int
            query     : str   — original English query
            expanded  : str   — German+English expanded query
            citations : list[str]  — final predicted citation strings
            top_results : pd.DataFrame  — full reranked results for debugging
        """
        try:
            query_id   = query_row.get("id", query_row.name)
            query_text = str(query_row["query"])
            
            # Sanitize query_text for logging (remove Unicode dashes that Windows console can't handle)
            sanitized_query = query_text[:100].replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
 
            logger.info(f"\nQuery {query_id}: '{sanitized_query}'")
 
            # ── Step 5: Query expansion ───────────────────────────────────────
            logger.info("  [5] Expanding query...")
            expanded_query = self._expand_query(query_row)
            sanitized_expanded = expanded_query[:100].replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
            logger.info(f"  Expanded: '{sanitized_expanded}'")
 
            # ── Step 6: Hybrid retrieval ──────────────────────────────────────
            # 4 retrievers: BM25 (law) + FAISS (law) + FAISS (unified) + Regex
            logger.info(f"  [6] Hybrid retrieval (top_k={self.hybrid_top_k})...")
            retriever = HybridRetriever(
                law_df         = self.law_df,
                val_df         = pd.DataFrame([query_row]),  # single-row val for BM25
                unified_corpus = self.unified,
                expanded_query = expanded_query,
                query_text     = query_text,
            )
            candidates = retriever.retrieve(top_k=self.hybrid_top_k)
            logger.info(f"  Retrieved {len(candidates)} candidates.")
 
            # ── Step 7: Reranker ──────────────────────────────────────────────
            # Two-stage: mmarco-MiniLM (fast, all 100) → bge-reranker-v2-m3 (top 25)
            logger.info(f"  [7] Reranking (top_k={self.reranker_top_k})...")
            reranked = self.reranker.rerank(
                query      = expanded_query,
                candidates = candidates,
                top_k      = self.reranker_top_k,
            )
 
            # ── Step 8: Extract citations ─────────────────────────────────────
            # Citation strings are already in the reranked DataFrame.
            # This is the competition submission output.
            citations = self.reranker.get_citations(reranked)
            logger.info(f"  Final citations: {citations}")
 
            return {
                "query_id":    query_id,
                "query":       query_text,
                "expanded":    expanded_query,
                "citations":   citations,
                "top_results": reranked,
            }
 
        except Exception as e:
            query_id = query_row.get("id", "?")
            # Sanitize error message for Windows console
            error_str = str(e).replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
            logger.error(f"  Query {query_id} failed: {error_str}")
            raise Agentic_Exception(e, sys) from e
 
    # ── MAIN RUN ──────────────────────────────────────────────────────────────
 
    def run(self) -> pd.DataFrame:
        """
        Run the full pipeline over all queries in val.csv.
 
        Returns
        -------
        pd.DataFrame — submission DataFrame with columns:
            query_id  : str or int
            citations : str  — semicolon-separated citation strings
                               e.g. "Art. 641 ZGB; BGE 148 III 1; Art. 184 OR"
        """
        try:
            if self.val_df is None:
                raise RuntimeError("Pipeline not started. Call startup() first.")
 
            logger.info("\n" + "=" * 60)
            logger.info(f"Processing {len(self.val_df)} queries")
            logger.info("=" * 60)
 
            submission_rows = []
            detailed_logs   = []
            pipeline_start  = time.time()
 
            for i, (_, query_row) in enumerate(self.val_df.iterrows(), start=1):
                q_start = time.time()
                logger.info(f"\n[Query {i}/{len(self.val_df)}]")
 
                result = self._process_query(query_row)
 
                # Format citations as semicolon-separated string for submission
                citations_str = "; ".join(result["citations"])
 
                submission_rows.append({
                    "query_id":  result["query_id"],
                    "citations": citations_str,
                })
 
                # Detailed log for analysis
                detailed_logs.append({
                    "query_id":  result["query_id"],
                    "query":     result["query"],
                    "expanded":  result["expanded"],
                    "citations": citations_str,
                    "n_results": len(result["citations"]),
                })
 
                elapsed = time.time() - q_start
                logger.info(f"  Query {i} done in {elapsed:.1f}s | "
                             f"Citations: {citations_str[:80]}")
 
            # ── Save submission ───────────────────────────────────────────────
            submission_df = pd.DataFrame(submission_rows)
            submission_path = self.save_dir / "submission.csv"
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"\nSubmission saved → {submission_path}")
 
            # ── Save detailed logs ────────────────────────────────────────────
            logs_df = pd.DataFrame(detailed_logs)
            logs_path = self.save_dir / "pipeline_logs.csv"
            logs_df.to_csv(logs_path, index=False)
            logger.info(f"Detailed logs saved → {logs_path}")
 
            # ── Save run metadata ─────────────────────────────────────────────
            total_elapsed = time.time() - pipeline_start
            metadata = {
                "timestamp":         datetime.now().isoformat(),
                "total_queries":     len(self.val_df),
                "total_elapsed_s":   round(total_elapsed, 2),
                "avg_per_query_s":   round(total_elapsed / len(self.val_df), 2),
                "hybrid_top_k":      self.hybrid_top_k,
                "reranker_top_k":    self.reranker_top_k,
                "use_stage2":        self.use_stage2_reranker,
            }
            meta_path = self.save_dir / "run_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
 
            logger.info("\n" + "=" * 60)
            logger.info("Pipeline complete!")
            logger.info(f"  Total queries   : {len(self.val_df)}")
            logger.info(f"  Total time      : {total_elapsed:.1f}s")
            logger.info(f"  Avg per query   : {total_elapsed/len(self.val_df):.1f}s")
            logger.info(f"  Submission file : {submission_path}")
            logger.info("=" * 60)
 
            return submission_df
 
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def run_pipeline(
    hybrid_top_k: int         = 100,
    reranker_top_k: int       = 10,
    use_stage2_reranker: bool = True,
    save_dir: str             = "artifacts",
) -> pd.DataFrame:
    """
    Convenience function — builds and runs the full pipeline in one call.
 
    Parameters
    ----------
    hybrid_top_k        : candidates per query from HybridRetriever (default 100)
    reranker_top_k      : final citations per query from Reranker    (default 10)
    use_stage2_reranker : use bge-reranker-v2-m3 as Stage 2         (default True)
    save_dir            : directory to save submission.csv           (default 'artifacts')
 
    Returns
    -------
    pd.DataFrame — submission with columns: query_id, citations
    """
    pipeline = RAGPipeline(
        hybrid_top_k        = hybrid_top_k,
        reranker_top_k      = reranker_top_k,
        use_stage2_reranker = use_stage2_reranker,
        save_dir            = save_dir,
    )
    pipeline.startup()
    return pipeline.run()
 
 
if __name__ == "__main__":
    # Default run: full pipeline, top-100 hybrid, top-10 reranked
    # Set use_stage2_reranker=False for fast CPU-only dev runs
    submission = run_pipeline(
        hybrid_top_k        = 100,
        reranker_top_k      = 10,
        use_stage2_reranker = True,
        save_dir            = "artifacts",
    )
    print(submission.head())