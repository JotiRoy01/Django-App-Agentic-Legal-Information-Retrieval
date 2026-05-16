import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
 
import pandas as pd
 
from agentic.data_loader.data_loader import load
from agentic.evaluation.evaluator import CitationEvaluator, EvaluationReport
from agentic.exception import Agentic_Exception
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
 
def _setup_logger() -> logging.Logger:
    # logger = logging.getLogger("Evaluation")
    # if not logger.handlers:
    #     handler = logging.StreamHandler()
    #     handler.setFormatter(logging.Formatter(
    #         "%(asctime)s | %(levelname)s | %(message)s",
    #         datefmt="%H:%M:%S"
    #     ))
    #     logger.addHandler(handler)
    # logger.setLevel(logging.INFO)

    from agentic.logger.logging import setup_logging, get_logger

    setup_logging()
    logger = get_logger()
    return logger
 
logger = _setup_logger()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EvaluationRunner
# ─────────────────────────────────────────────────────────────────────────────
 
class EvaluationRunner:
    """
    Loads data, runs CitationEvaluator, saves all results to disk.
 
    Parameters
    ----------
    k        : int — K for MAP@K and Hit@K. Must match reranker_top_k. Default 10.
    save_dir : str — directory to save evaluation outputs. Default 'artifacts/evaluation'.
    """
 
    def __init__(
        self,
        k: int        = 10,
        save_dir: str = "artifacts/evaluation",
    ):
        self.k        = k
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = CitationEvaluator(k=k)
 
    def run(
        self,
        submission_df: pd.DataFrame,
        gold_df:       pd.DataFrame,
    ) -> EvaluationReport:
        """
        Run full evaluation and save all outputs to disk.
 
        Parameters
        ----------
        submission_df : output of rag_pipeline.run()
                        columns: query_id, citations
        gold_df       : val.csv as DataFrame
                        columns: id/query_id, query, citations
 
        Returns
        -------
        EvaluationReport — use .summary() to print, .to_dict() for Django API.
        """
        try:
            logger.info("=" * 55)
            logger.info("Running Evaluation")
            logger.info(f"  Submission rows : {len(submission_df)}")
            logger.info(f"  Gold rows       : {len(gold_df)}")
            logger.info(f"  K               : {self.k}")
            logger.info("=" * 55)
 
            # ── Run evaluation ────────────────────────────────────────────────
            report = self.evaluator.evaluate_submission(
                submission_df = submission_df,
                gold_df       = gold_df,
            )
 
            # ── Save outputs ──────────────────────────────────────────────────
            self._save_summary(report)
            self._save_per_query(report)
            self._save_worst_queries(report)
            self._save_best_queries(report)
            self._save_error_analysis(report)
 
            logger.info(f"\nAll evaluation outputs saved to: {self.save_dir}/")
            return report
 
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
 
    # ── Save methods ──────────────────────────────────────────────────────────
 
    def _save_summary(self, report: EvaluationReport):
        """Save aggregate metrics as JSON — Django API endpoint."""
        summary = {
            "timestamp":    datetime.now().isoformat(),
            "k":            self.k,
            **report.to_dict(),
        }
        path = self.save_dir / "evaluation_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Saved: {path}")
 
    def _save_per_query(self, report: EvaluationReport):
        """Save per-query results CSV — Django bar charts."""
        df = report.to_dataframe()
        path = self.save_dir / "per_query_results.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Saved: {path}")
 
    def _save_worst_queries(self, report: EvaluationReport, n: int = 10):
        """Save bottom-N queries by F1 — for debugging failures."""
        df = report.to_dataframe()
        worst = df.nsmallest(n, "f1")
        path  = self.save_dir / "worst_queries.csv"
        worst.to_csv(path, index=False)
        logger.info(f"  Saved: {path} (bottom {n} queries by F1)")
 
    def _save_best_queries(self, report: EvaluationReport, n: int = 10):
        """Save top-N queries by F1."""
        df   = report.to_dataframe()
        best = df.nlargest(n, "f1")
        path = self.save_dir / "best_queries.csv"
        best.to_csv(path, index=False)
        logger.info(f"  Saved: {path} (top {n} queries by F1)")
 
    def _save_error_analysis(self, report: EvaluationReport):
        """
        Save detailed error analysis — what citations are being missed most often.
        Useful for improving the pipeline: if the same citation keeps appearing
        in false_negatives, it means your retriever is not finding that article.
        """
        # Count how often each citation appears in false negatives
        fn_counts: dict[str, int] = {}
        fp_counts: dict[str, int] = {}
 
        for r in report.query_results:
            for cit in r.false_negatives:
                fn_counts[cit] = fn_counts.get(cit, 0) + 1
            for cit in r.false_positives:
                fp_counts[cit] = fp_counts.get(cit, 0) + 1
 
        # Most missed citations (false negatives)
        fn_df = pd.DataFrame([
            {"citation": cit, "missed_count": count}
            for cit, count in sorted(fn_counts.items(), key=lambda x: -x[1])
        ])
        fn_path = self.save_dir / "most_missed_citations.csv"
        fn_df.to_csv(fn_path, index=False)
 
        # Most wrongly predicted citations (false positives)
        fp_df = pd.DataFrame([
            {"citation": cit, "wrong_count": count}
            for cit, count in sorted(fp_counts.items(), key=lambda x: -x[1])
        ])
        fp_path = self.save_dir / "most_wrong_citations.csv"
        fp_df.to_csv(fp_path, index=False)
 
        logger.info(f"  Saved: {fn_path}")
        logger.info(f"  Saved: {fp_path}")
 
        # Print top 5 missed citations as a quick diagnostic
        if not fn_df.empty:
            logger.info("\n  Top 5 most missed citations (check your retriever):")
            for _, row in fn_df.head(5).iterrows():
                logger.info(f"    '{row['citation']}' missed {row['missed_count']} times")
 
        if not fp_df.empty:
            logger.info("\n  Top 5 most wrongly predicted (check your reranker):")
            for _, row in fp_df.head(5).iterrows():
                logger.info(f"    '{row['citation']}' wrong {row['wrong_count']} times")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Convenience function — standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def run_evaluation(
    submission_path: str = "artifacts/submission.csv",
    gold_path:       str = "data/val.csv",
    k:               int = 10,
    save_dir:        str = "artifacts/evaluation",
) -> EvaluationReport:
    """
    Load submission + gold from disk and run full evaluation.
 
    Parameters
    ----------
    submission_path : path to submission.csv from rag_pipeline.run()
    gold_path       : path to val.csv with gold citations
    k               : K for MAP@K. Must match reranker_top_k in pipeline.
    save_dir        : where to save evaluation outputs
 
    Returns
    -------
    EvaluationReport
 
    Usage:
        from agentic.evaluation.evaluation_report import run_evaluation
        report = run_evaluation()
        print(report.summary())
    """
    try:
        logger.info(f"Loading submission: {submission_path}")
        submission_df = pd.read_csv(submission_path)
 
        logger.info(f"Loading gold labels: {gold_path}")
        gold_df = pd.read_csv(gold_path)
 
        runner = EvaluationRunner(k=k, save_dir=save_dir)
        return runner.run(submission_df=submission_df, gold_df=gold_df)
 
    except Exception as e:
        raise Agentic_Exception(e, sys) from e
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    report = run_evaluation(
        submission_path = "artifacts/submission.csv",
        gold_path       = "data/val.csv",
        k               = 10,
    )
    print(report.summary())