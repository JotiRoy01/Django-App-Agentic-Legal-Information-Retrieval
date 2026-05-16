#!/usr/bin/env python3
"""
Run the full RAG pipeline for the Agentic Retrieval Competition.

This script loads the data, builds indexes, and processes all queries in val.csv
to generate citation predictions.

Usage:
    python scripts/run_pipeline.py

The results will be saved to artifacts/submission.csv
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from agentic.pipeline.rag_pipeline import RAGPipeline
from agentic.logger.logging import setup_logging, get_logger

def main():
    # Setup logging
    setup_logging()
    logger = get_logger()

    logger.info("Starting RAG Pipeline execution...")

    try:
        # Initialize pipeline with default settings
        pipeline = RAGPipeline(
            hybrid_top_k=100,
            reranker_top_k=10,
            use_stage2_reranker=True,
            save_dir="artifacts",
            dev_mode = True
        )

        # Startup: load data and build indexes (this takes time)
        pipeline.startup()

        # Run the full pipeline on all queries
        submission_df = pipeline.run()

        # Save results
        output_path = Path("artifacts") / "submission.csv"
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        logger.info("Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()