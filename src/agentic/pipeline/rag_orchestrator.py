"""
RAG Pipeline Orchestration Module.

Manages the complete chunking, storage, and retrieval workflow for production deployment.

Features:
- Load datasets from config
- Apply production chunking pipeline
- Store chunks efficiently
- Provide retrieval interfaces
- Generate pipeline statistics

Author: Agentic RAG Pipeline
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Union
import pandas as pd
import json
from datetime import datetime
import logging

from agentic.data_loader.data_loader import DataLoader
from agentic.chunkings.production_chunker import (
    ProductionChunkingPipeline,
    LawsSemanticChunker,
    CourtDecisionHierarchicalChunker
)
from agentic.exception import Agentic_Exception

from agentic.logger.logging import setup_logging, get_logger

setup_logging()
logger = get_logger()


class RAGPipelineOrchestrator:
    """
    Orchestrates the complete RAG pipeline from data loading to chunk storage.
    
    Workflow:
    1. Load raw datasets (laws_de.csv, court_considerations.csv)
    2. Apply production chunking
    3. Store chunks (CSV/Parquet)
    4. Generate statistics and metadata
    5. Provide retrieval interfaces
    """
    
    def __init__(self, 
                 project_root: Optional[Path] = None,
                 artifacts_dir: str = "artifacts/chunks",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the orchestrator.
        
        Args:
            project_root: Root directory of the project. Auto-detected if None.
            artifacts_dir: Directory to store chunk artifacts
            logger: Logger instance. If None, creates a new one.
        """
        self.project_root = project_root or self._detect_project_root()
        self.artifacts_dir = Path(artifacts_dir) if isinstance(artifacts_dir, str) else artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        self.data_loader = DataLoader()
        self.pipeline = ProductionChunkingPipeline()
    
    @staticmethod
    def _detect_project_root() -> Path:
        """Auto-detect project root."""
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / 'pyproject.toml').exists() or (current / 'README.md').exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup basic logger."""
        # logger = logging.getLogger("RAG_Pipeline")
        # if not logger.handlers:
        #     handler = logging.StreamHandler()
        #     handler.setFormatter(logging.Formatter(
        #         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        #     ))
        #     logger.addHandler(handler)
        #     logger.setLevel(logging.INFO)
        from agentic.logger.logging import setup_logging, get_logger

        setup_logging()
        logger = get_logger()
        #logger = logging.getLogger("RAG_Pipeline")
        return logger
    
    def load_laws_dataset(self, 
                         filename: str = "laws_de.csv",
                         nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load laws dataset.
        
        Args:
            filename: CSV filename
            nrows: Maximum rows to load (useful for testing)
        
        Returns:
            DataFrame with columns: citation, text, title
        """
        self.logger.info(f"Loading laws dataset: {filename}")
        
        try:
            df = self.data_loader.load_data(filename, nrows=nrows)
            self.logger.info(f"Loaded {len(df)} laws records")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load laws dataset: {e}")
            raise
    
    def load_court_dataset(self, 
                          filename: str = "court_considerations.csv",
                          nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load court decisions dataset.
        
        Args:
            filename: CSV filename
            nrows: Maximum rows to load (useful for testing)
        
        Returns:
            DataFrame with columns: citation, text
        """
        self.logger.info(f"Loading court dataset: {filename}")
        
        try:
            df = self.data_loader.load_data(filename, nrows=nrows)
            self.logger.info(f"Loaded {len(df)} court decision records")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load court dataset: {e}")
            raise
    
    def chunk_laws(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply semantic chunking to laws dataset.
        
        Returns:
            DataFrame with columns: 
            - citation, title, chunk_id, text, tokens, article, chunk_type, position
        """
        self.logger.info(f"Chunking {len(df)} laws records (semantic strategy)...")
        
        try:
            chunks_df = self.pipeline.chunk_laws_dataset(df)
            self.logger.info(f"Generated {len(chunks_df)} law chunks")
            
            stats = self.pipeline.get_chunking_stats(chunks_df)
            self.logger.info(f"Laws chunking stats: {stats}")
            
            return chunks_df
        except Exception as e:
            self.logger.error(f"Failed to chunk laws: {e}")
            raise
    
    def chunk_court_decisions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hierarchical chunking to court decisions dataset.
        
        Returns:
            DataFrame with columns:
            - citation, chunk_id, text, tokens, section, chunk_type, position
        """
        self.logger.info(f"Chunking {len(df)} court decision records (hierarchical strategy)...")
        
        try:
            chunks_df = self.pipeline.chunk_court_decisions_dataset(df)
            self.logger.info(f"Generated {len(chunks_df)} court decision chunks")
            
            stats = self.pipeline.get_chunking_stats(chunks_df)
            self.logger.info(f"Court chunking stats: {stats}")
            
            return chunks_df
        except Exception as e:
            self.logger.error(f"Failed to chunk court decisions: {e}")
            raise
    
    def save_chunks(self, 
                   chunks_df: pd.DataFrame, 
                   name: str,
                   format: str = "parquet") -> Path:
        """
        Save chunks to disk.
        
        Args:
            chunks_df: DataFrame with chunks
            name: Name for the saved file (e.g., "laws_chunks")
            format: 'parquet' (recommended) or 'csv'
        
        Returns:
            Path to saved file
        """
        if format == "parquet":
            filepath = self.artifacts_dir / f"{name}.parquet"
            chunks_df.to_parquet(filepath, index=False)
            self.logger.info(f"Saved chunks to {filepath}")
        elif format == "csv":
            filepath = self.artifacts_dir / f"{name}.csv"
            chunks_df.to_csv(filepath, index=False)
            self.logger.info(f"Saved chunks to {filepath}")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath
    
    def load_chunks(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load chunks from disk."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def run_full_pipeline(self, 
                         laws_file: str = "laws_de.csv",
                         court_file: str = "court_considerations.csv",
                         laws_nrows: Optional[int] = None,
                         court_nrows: Optional[int] = None,
                         save_format: str = "parquet") -> Dict:
        """
        Run the complete RAG pipeline end-to-end.
        
        Returns:
            Dictionary with paths and statistics
        """
        start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info("Starting RAG Pipeline Orchestration")
        self.logger.info("="*60)
        
        try:
            # 1. Load datasets
            self.logger.info("\n[1/5] Loading datasets...")
            laws_df = self.load_laws_dataset(laws_file, nrows=laws_nrows)
            court_df = self.load_court_dataset(court_file, nrows=court_nrows)
            
            # 2. Chunk laws
            self.logger.info("\n[2/5] Chunking laws dataset...")
            laws_chunks = self.chunk_laws(laws_df)
            
            # 3. Chunk court decisions
            self.logger.info("\n[3/5] Chunking court decisions...")
            court_chunks = self.chunk_court_decisions(court_df)
            
            # 4. Save chunks
            self.logger.info("\n[4/5] Saving chunks to disk...")
            laws_path = self.save_chunks(laws_chunks, "laws_chunks", format=save_format)
            court_path = self.save_chunks(court_chunks, "court_chunks", format=save_format)
            
            # 5. Generate report
            self.logger.info("\n[5/5] Generating pipeline report...")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'input_datasets': {
                    'laws': {
                        'file': laws_file,
                        'records': len(laws_df),
                        'nrows_limit': laws_nrows
                    },
                    'court_decisions': {
                        'file': court_file,
                        'records': len(court_df),
                        'nrows_limit': court_nrows
                    }
                },
                'output_chunks': {
                    'laws': {
                        'file': str(laws_path),
                        'num_chunks': len(laws_chunks),
                        'stats': self.pipeline.get_chunking_stats(laws_chunks)
                    },
                    'court_decisions': {
                        'file': str(court_path),
                        'num_chunks': len(court_chunks),
                        'stats': self.pipeline.get_chunking_stats(court_chunks)
                    }
                },
                'total_chunks': len(laws_chunks) + len(court_chunks),
                'total_tokens': laws_chunks['tokens'].sum() + court_chunks['tokens'].sum()
            }
            
            # Save report
            report_path = self.artifacts_dir / "pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info("\n" + "="*60)
            self.logger.info("RAG Pipeline Orchestration Complete!")
            self.logger.info("="*60)
            self.logger.info(f"Total Chunks: {report['total_chunks']}")
            self.logger.info(f"Total Tokens: {report['total_tokens']}")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            self.logger.info(f"Report saved to: {report_path}")
            self.logger.info("="*60 + "\n")
            
            return report
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


class RetrievalInterface:
    """Simple interface for retrieving chunks by citation or text search."""
    
    def __init__(self, chunks_df: pd.DataFrame):
        """Initialize with chunks DataFrame."""
        self.chunks_df = chunks_df
    
    def get_by_citation(self, citation: str) -> pd.DataFrame:
        """Get all chunks for a specific citation."""
        return self.chunks_df[self.chunks_df['citation'] == citation]
    
    def get_by_article(self, article: str) -> pd.DataFrame:
        """Get all chunks for a specific article."""
        if 'article' in self.chunks_df.columns:
            return self.chunks_df[self.chunks_df['article'] == article]
        return pd.DataFrame()
    
    def get_by_section(self, section: str) -> pd.DataFrame:
        """Get all chunks for a specific section."""
        if 'section' in self.chunks_df.columns:
            return self.chunks_df[self.chunks_df['section'] == section]
        return pd.DataFrame()
    
    def search_text(self, query: str) -> pd.DataFrame:
        """Simple text search in chunks."""
        return self.chunks_df[
            self.chunks_df['text'].str.contains(query, case=False, na=False)
        ]
    
    def get_by_token_range(self, min_tokens: int, max_tokens: int) -> pd.DataFrame:
        """Get chunks within token range."""
        return self.chunks_df[
            (self.chunks_df['tokens'] >= min_tokens) & 
            (self.chunks_df['tokens'] <= max_tokens)
        ]
    
    def get_chunks_for_source(self, source_type: str) -> pd.DataFrame:
        """Get all chunks for a source type ('law' or 'court_decision')."""
        return self.chunks_df[self.chunks_df['source_type'] == source_type]


# Convenience functions
def run_production_chunking_pipeline(
    laws_file: str = "laws_de.csv",
    court_file: str = "court_considerations.csv",
    laws_nrows: Optional[int] = None,
    court_nrows: Optional[int] = None,
    save_format: str = "parquet"
) -> Dict:
    """
    Quick entry point to run the full RAG chunking pipeline.
    
    Example:
        >>> report = run_production_chunking_pipeline(laws_nrows=1000, court_nrows=5000)
        >>> print(report['total_chunks'])
    """
    orchestrator = RAGPipelineOrchestrator()
    return orchestrator.run_full_pipeline(
        laws_file=laws_file,
        court_file=court_file,
        laws_nrows=laws_nrows,
        court_nrows=court_nrows,
        save_format=save_format
    )
