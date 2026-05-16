import re
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
 
from agentic.exception import Agentic_Exception

@dataclass
class QueryResult :
    """
    This class store the evaluation result for a single query.
    """
    query_id: str
    query_text: str
    predicted: list[str]
    gold: list[str]
    precision: float = 0.0
    recall: float=0.0
    f1: float = 0.0
    average_precision: float = 0.0
    reciprocal_rank: float = 0.0
    hit_at_k: bool = False # did top_k contain correct citation

    # correctly predicted
    true_positives: list[str] = field(default_factory=list)
    # predicted but wrong
    false_positives: list[str] = field(default_factory=list)
    # gold but not predicted
    false_negatives: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport :
    """
    Full evaluation report across all queries.
    I will show it in my Django dashboard.
    """
    query_results:    list[QueryResult]
    macro_f1:         float = 0.0      # average F1 across queries (competition metric)
    macro_precision:  float = 0.0
    macro_recall:     float = 0.0
    map_at_k:         float = 0.0      # Mean Average Precision@K
    mrr:              float = 0.0      # Mean Reciprocal Rank
    hit_rate_at_k:    float = 0.0      # fraction of queries with at least 1 hit
    total_queries:    int   = 0
    perfect_queries:  int   = 0        # query with f1 is 1.0
    zero_f1_queries:  int   = 0        # query with f1 is 0.0


    def summary(self) -> str:
        """Print a clean summary — used in logs and Django views.
        """
        lines = [
            "=" * 55,
            "EVALUATION REPORT",
            "=" * 55,
            f"  Total queries      : {self.total_queries}",
            f"  Macro F1 (main)    : {self.macro_f1:.4f}",
            f"  Macro Precision    : {self.macro_precision:.4f}",
            f"  Macro Recall       : {self.macro_recall:.4f}",
            f"  MAP@K              : {self.map_at_k:.4f}",
            f"  MRR                : {self.mrr:.4f}",
            f"  Hit Rate@K         : {self.hit_rate_at_k:.4f}",
            f"  Perfect (F1=1.0)   : {self.perfect_queries}/{self.total_queries}",
            f"  Zero F1            : {self.zero_f1_queries}/{self.total_queries}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable dict for Django views / JSON API."""
        return {
            "macro_f1":        round(self.macro_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall":    round(self.macro_recall, 4),
            "map_at_k":        round(self.map_at_k, 4),
            "mrr":             round(self.mrr, 4),
            "hit_rate_at_k":   round(self.hit_rate_at_k, 4),
            "total_queries":   self.total_queries,
            "perfect_queries": self.perfect_queries,
            "zero_f1_queries": self.zero_f1_queries,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Per-query results as DataFrame — for charts in Django."""
        rows = []
        for r in self.query_results:
            rows.append({
                "query_id":         r.query_id,
                "query_text":       r.query_text,
                "precision":        round(r.precision, 4),
                "recall":           round(r.recall, 4),
                "f1":               round(r.f1, 4),
                "average_precision": round(r.average_precision, 4),
                "reciprocal_rank":  round(r.reciprocal_rank, 4),
                "hit_at_k":         r.hit_at_k,
                "n_predicted":      len(r.predicted),
                "n_gold":           len(r.gold),
                "n_true_positives": len(r.true_positives),
                "true_positives":   "; ".join(r.true_positives),
                "false_positives":  "; ".join(r.false_positives),
                "false_negatives":  "; ".join(r.false_negatives),
            })
        return pd.DataFrame(rows)
    
# citation normalization
def normalize_citation(citation: str) -> str:
    """
    Normalize a citation string for comparison.
    Handles: case, extra whitespace, period spacing.
    e.g. "Art. 641 ZGB" == "art.641 zgb" == "ART. 641  ZGB"
    """
    c = citation.strip().lower()
    c = re.sub(r'\s+', ' ', c)       # collapse multiple spaces
    c = re.sub(r'\s*\.\s*', '.', c)  # normalize spaces around periods
    return c
 
 
def parse_citations(citations_str: str) -> list[str]:
    """
    Parse a semicolon-separated citation string into a list.
    Handles both '; ' and ';' separators.
    Skips empty strings.
    e.g. "Art. 641 ZGB; BGE 148 III 1" → ["Art. 641 ZGB", "BGE 148 III 1"]
    """
    if pd.isna(citations_str) or str(citations_str).strip() == "":
        return []
    parts = str(citations_str).split(";")
    return [p.strip() for p in parts if p.strip()]

class CitationEvaluator :
    """
    Evaluates citation retrieval quality at both query and corpus level.
 
    Instantiate once, call evaluate_query() or evaluate_submission().
    """
 
    def __init__(self, k: int = 10):
        """
        Parameters
        ----------
        k : int
            The K in MAP@K and Hit@K. Should match reranker_top_k. Default 10.
        """
        self.k = k

    # ── Single query ─────

    def evaluate_query(
        self,
        predicted:  list[str],
        gold:       list[str],
        query_id:   str = "",
        query_text: str = "",
    ) -> QueryResult:
        """
        Compute all metrics for one query.
 
        Parameters
        ----------
        predicted  : list of predicted citation strings (ordered, best first)
        gold       : list of gold citation strings (unordered)
        query_id   : identifier for this query
        query_text : raw query text (for reporting)
 
        Returns
        -------
        QueryResult with all metrics filled in.
        """
        # Normalize for comparison
        pred_normalized = [normalize_citation(c) for c in predicted[:self.k]]
        gold_normalized = set(normalize_citation(c) for c in gold)
 
        # Keep original strings for reporting
        pred_original = predicted[:self.k]
        gold_original = list(gold)


        if not pred_normalized and not gold_normalized:
            precision = recall = f1 = 1.0
        elif not pred_normalized:
            precision = recall = f1 = 0.0
        elif not gold_normalized:
            precision = 1.0
            recall    = 0.0
            f1        = 0.0
        else:
            tp_normalized = [c for c in pred_normalized if c in gold_normalized]
            precision = len(tp_normalized) / len(pred_normalized)
            recall    = len(tp_normalized) / len(gold_normalized)
            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        

        gold_norm_to_orig = {
            normalize_citation(c): c for c in gold_original
        }
        pred_norm_to_orig = {
            normalize_citation(c): c for c in pred_original
        }
 
        true_positives  = [pred_norm_to_orig[c] for c in pred_normalized
                           if c in gold_normalized]
        false_positives = [pred_norm_to_orig[c] for c in pred_normalized
                           if c not in gold_normalized]
        false_negatives = [gold_norm_to_orig[c] for c in gold_normalized
                           if c not in set(pred_normalized)]


        # Average Precision (for MAP@K)
        ap = 0.0
        hits = 0
        for rank, norm_cit in enumerate(pred_normalized, start=1):
            if norm_cit in gold_normalized:
                hits += 1
                ap += hits / rank
        average_precision = ap / len(gold_normalized) if gold_normalized else 0.0


        # ── Reciprocal Rank (for MRR) ─────────────────────────────────────────
        reciprocal_rank = 0.0
        for rank, norm_cit in enumerate(pred_normalized, start=1):
            if norm_cit in gold_normalized:
                reciprocal_rank = 1.0 / rank
                break
        

        # ── Hit@K ─────────────────────────────────────────────────────────────
        hit_at_k = any(c in gold_normalized for c in pred_normalized)
 
        return QueryResult(
            query_id          = str(query_id),
            query_text        = str(query_text),
            predicted         = pred_original,
            gold              = gold_original,
            precision         = precision,
            recall            = recall,
            f1                = f1,
            average_precision = average_precision,
            reciprocal_rank   = reciprocal_rank,
            hit_at_k          = hit_at_k,
            true_positives    = true_positives,
            false_positives   = false_positives,
            false_negatives   = false_negatives,
        )

    # ── Full submission ───────────────────────────────────────────────────────
 
    def evaluate_submission(
        self,
        submission_df: pd.DataFrame,
        gold_df:       pd.DataFrame,
        query_id_col:  str = "query_id",
        citations_col: str = "citations",
        query_col:     str = "query",
    ) -> EvaluationReport:
        """
        Evaluate a full submission DataFrame against gold labels.
 
        Parameters
        ----------
        submission_df : output of rag_pipeline.run()
                        columns: query_id, citations (semicolon-separated)
        gold_df       : val.csv loaded as DataFrame
                        columns: query_id (or id), citations, query
        query_id_col  : name of the query ID column. Default 'query_id'.
        citations_col : name of the citations column. Default 'citations'.
        query_col     : name of the query text column. Default 'query'.
 
        Returns
        -------
        EvaluationReport with per-query results and aggregate metrics.
        """
        try:
            # Normalize gold_df column names
            # val.csv may use 'id' instead of 'query_id'
            if 'id' in gold_df.columns and query_id_col not in gold_df.columns:
                gold_df = gold_df.rename(columns={'id': query_id_col})
 
            # Build gold lookup: query_id → (citations list, query text)
            gold_lookup = {}
            for _, row in gold_df.iterrows():
                qid        = str(row[query_id_col])
                gold_cits  = parse_citations(row.get(citations_col, ""))
                query_text = str(row.get(query_col, ""))
                gold_lookup[qid] = (gold_cits, query_text)
 
            # Evaluate each row in submission
            query_results = []
            for _, row in submission_df.iterrows():
                qid       = str(row[query_id_col])
                predicted = parse_citations(row.get(citations_col, ""))
                gold_cits, query_text = gold_lookup.get(qid, ([], ""))
 
                result = self.evaluate_query(
                    predicted  = predicted,
                    gold       = gold_cits,
                    query_id   = qid,
                    query_text = query_text,
                )
                query_results.append(result)
 
            # ── Aggregate metrics ─────────────────────────────────────────────
            n = len(query_results)
 
            macro_f1        = np.mean([r.f1 for r in query_results])
            macro_precision = np.mean([r.precision for r in query_results])
            macro_recall    = np.mean([r.recall for r in query_results])
            map_at_k        = np.mean([r.average_precision for r in query_results])
            mrr             = np.mean([r.reciprocal_rank for r in query_results])
            hit_rate_at_k   = np.mean([r.hit_at_k for r in query_results])
            perfect         = sum(1 for r in query_results if r.f1 == 1.0)
            zero_f1         = sum(1 for r in query_results if r.f1 == 0.0)
 
            report = EvaluationReport(
                query_results    = query_results,
                macro_f1         = float(macro_f1),
                macro_precision  = float(macro_precision),
                macro_recall     = float(macro_recall),
                map_at_k         = float(map_at_k),
                mrr              = float(mrr),
                hit_rate_at_k    = float(hit_rate_at_k),
                total_queries    = n,
                perfect_queries  = perfect,
                zero_f1_queries  = zero_f1,
            )
 
            print(report.summary())
            return report
 
        except Exception as e:
            raise Agentic_Exception(e, sys) from e