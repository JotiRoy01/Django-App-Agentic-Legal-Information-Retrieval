"""
Production-level RAG chunking pipeline for legal document retrieval.

Specialized chunkers for:
- laws_de.csv: Semantic chunking by article subsections
- court_considerations.csv: Hierarchical chunking by decision sections

Author: Agentic RAG Pipeline
"""

import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from agentic.exception import Agentic_Exception
import sys


@dataclass
class ChunkMetadata:
    """Metadata for a chunk."""
    source_type: str  # 'law' or 'court_decision'
    citation: str  # Art. 1, BGE 139 I 2, etc.
    section_id: str  # Section identifier
    parent_section: Optional[str] = None  # Parent hierarchy
    subsection: Optional[str] = None  # Subsection if applicable
    position: int = 0  # Position in parent document


class TokenCounter:
    """Simple token counter using character-based approximation for legal texts."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Estimate token count. Uses ~4 chars per token (GPT-3 approximation).
        For production, consider using tiktoken library.
        """
        # More accurate for legal texts (German has longer words)
        return len(text) // 3.5
    
    @staticmethod
    def estimate_tokens_from_words(word_count: int) -> int:
        """Estimate tokens from word count (~1.3 tokens per word on average)."""
        return int(word_count * 1.3)


class LawsSemanticChunker:
    """
    Semantic chunker for Swiss/German legal texts (laws_de.csv).
    
    Strategy:
    - Preserves article structure (Art. 1, Art. 1 Abs. 1, etc.)
    - Groups by numbered subsections (Abs., sentences)
    - Target: 100-300 tokens per chunk
    - Minimal overlap (preserves clarity)
    
    Example:
        Art. 1 Abs. 1 - text about principle
        Art. 1 Abs. 2 - text about exceptions
        → Two separate chunks, each self-contained
    """
    
    def __init__(self, target_tokens: int = 200, max_tokens: int = 300, min_tokens: int = 50):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
    
    def extract_article_structure(self, text: str) -> List[Dict]:
        """
        Extract article structure from legal text.
        Returns: [{"article": "Art. 1", "abs": ["Abs. 1 content", "Abs. 2 content", ...]}]
        """
        # Pattern for Swiss legal articles: Art. 1, Art. 1 Abs. 1, Art. 1a, § 1, etc.
        article_pattern = r"(Art\.?\s+\d+[a-zA-Z]*)"
        abs_pattern = r"(Abs\.?\s+\d+)"
        
        articles = []
        current_article = None
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for article header
            if re.search(article_pattern, line):
                article_match = re.search(article_pattern, line)
                if article_match:
                    # Save previous article if exists
                    if current_article and current_article.get('content'):
                        articles.append(current_article)
                    
                    current_article = {
                        'article': article_match.group(1),
                        'content': line,
                        'sections': []
                    }
            elif current_article and line:
                current_article['content'] += '\n' + line
            
            i += 1
        
        # Append last article
        if current_article and current_article.get('content'):
            articles.append(current_article)
        
        return articles
    
    def split_by_subsections(self, text: str, article_header: str) -> List[Dict]:
        """
        Split article content by Abs. (subsection), sentences, or logical breaks.
        """
        chunks = []
        
        # Try to split by "Abs. X"
        abs_pattern = r"(?:^|\n)\s*Abs\.?\s*\d+\.?\s+"
        
        if re.search(abs_pattern, text):
            # Split by Abs.
            sections = re.split(abs_pattern, text)
            
            for section in sections:
                section = section.strip()
                if len(section) < self.min_tokens:
                    continue
                
                tokens = TokenCounter.count_tokens(section)
                
                if tokens <= self.max_tokens:
                    chunks.append({
                        'text': section,
                        'tokens': tokens,
                        'article': article_header,
                        'type': 'subsection'
                    })
                else:
                    # If still too large, split by sentences
                    sub_chunks = self._split_by_sentences(section, article_header)
                    chunks.extend(sub_chunks)
        else:
            # No clear Abs. structure, split by sentences
            chunks = self._split_by_sentences(text, article_header)
        
        return chunks
    
    def _split_by_sentences(self, text: str, article_header: str) -> List[Dict]:
        """Split text by sentences (German legal text patterns)."""
        chunks = []
        
        # German sentence enders: . ! ? followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            tokens = TokenCounter.count_tokens(test_chunk)
            
            if tokens <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it meets minimum
                if current_chunk and TokenCounter.count_tokens(current_chunk) >= self.min_tokens:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': TokenCounter.count_tokens(current_chunk),
                        'article': article_header,
                        'type': 'sentence_group'
                    })
                
                # Start new chunk
                current_chunk = sentence
        
        # Don't forget last chunk
        if current_chunk and TokenCounter.count_tokens(current_chunk) >= self.min_tokens:
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': TokenCounter.count_tokens(current_chunk),
                'article': article_header,
                'type': 'sentence_group'
            })
        
        return chunks
    
    def chunk(self, text: str, citation: str = "") -> List[Dict]:
        """
        Main entry point: chunk a law document semantically.
        
        Returns: List of chunks with metadata
        """
        if not text or not isinstance(text, str):
            return []
        
        chunks = []
        
        try:
            # First, extract article structure
            articles = self.extract_article_structure(text)
            
            if not articles:
                # Fallback: treat entire text as one article
                articles = [{'article': citation or 'Full Text', 'content': text}]
            
            for article_info in articles:
                article_header = article_info.get('article', citation)
                article_content = article_info.get('content', '')
                
                # Split article by subsections
                article_chunks = self.split_by_subsections(article_content, article_header)
                chunks.extend(article_chunks)
        
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
        
        return chunks if chunks else [{'text': text, 'tokens': TokenCounter.count_tokens(text), 'article': citation}]


class CourtDecisionHierarchicalChunker:
    """
    Hierarchical chunker for Swiss court decisions (court_considerations.csv).
    
    Structure of BGE (Swiss Federal Court decisions):
        BGE 139 I 2 E. 1      - Decision header
        BGE 139 I 2 E. 1.1    - First reasoning point
        BGE 139 I 2 E. 1.2    - Second reasoning point
        BGE 139 I 2 E. 2      - Second main section
        ...
    
    Strategy:
    - Keep BGE decision number + section identifier together
    - Group related subsections (E. 1.1, E. 1.2 together)
    - Target: 200-500 tokens per chunk
    - Preserve hierarchy for context
    
    Example output:
        Chunk 1: "BGE 139 I 2 E. 1\n[E. 1.1 content]\n[E. 1.2 content]"
        Chunk 2: "BGE 139 I 2 E. 2\n[E. 2 content]"
    """
    
    def __init__(self, target_tokens: int = 350, max_tokens: int = 500, min_tokens: int = 100):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
    
    def extract_decision_sections(self, text: str) -> List[Dict]:
        """
        Extract decision sections from court text.
        Pattern: "E. 1", "E. 1.1", "E. 2", etc.
        """
        # Pattern for BGE sections: E. 1, E. 1.1, E. 1.2, E. 2, etc.
        section_pattern = r"E\.?\s+\d+(?:\.\d+)?"
        
        sections = []
        current_section = None
        current_content = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line contains a section header
            section_match = re.search(section_pattern, line_stripped)
            
            if section_match and re.match(r"^E\.?\s+\d+(?:\.\d+)?[\s,]", line_stripped):
                # This is a new section header
                if current_section:
                    sections.append({
                        'section_id': current_section,
                        'content': current_content.strip()
                    })
                
                current_section = section_match.group(0)
                current_content = line_stripped
            else:
                if current_section:
                    current_content += '\n' + line_stripped
        
        # Don't forget last section
        if current_section and current_content:
            sections.append({
                'section_id': current_section,
                'content': current_content.strip()
            })
        
        return sections
    
    def group_subsections(self, sections: List[Dict]) -> List[Dict]:
        """
        Group subsections hierarchically.
        E.g., group E. 1.1, E. 1.2 with parent E. 1
        """
        if not sections:
            return []
        
        grouped = []
        current_group = None
        current_content = ""
        
        for section in sections:
            section_id = section.get('section_id', '')
            content = section.get('content', '')
            
            # Extract main section number (e.g., "1" from "E. 1.1")
            main_section = re.search(r'E\.?\s+(\d+)', section_id)
            if not main_section:
                continue
            
            main_num = main_section.group(1)
            is_subsection = '.' in section_id
            
            if not is_subsection:
                # This is a main section (E. 1, E. 2, etc.)
                if current_group:
                    grouped.append(current_group)
                
                current_group = {
                    'section_id': section_id,
                    'subsections': [(section_id, content)],
                    'content': content
                }
                current_content = content
            else:
                # This is a subsection (E. 1.1, E. 1.2, etc.)
                if current_group:
                    # Add to current group if same main section
                    current_content += '\n' + content
                    current_group['subsections'].append((section_id, content))
                    current_group['content'] = current_content
        
        if current_group:
            grouped.append(current_group)
        
        return grouped
    
    def chunk_group(self, group: Dict, decision_header: str) -> List[Dict]:
        """
        Chunk a grouped section, respecting the size limits.
        """
        chunks = []
        section_id = group.get('section_id', '')
        content = group.get('content', '').strip()
        
        if not content:
            return []
        
        tokens = TokenCounter.count_tokens(content)
        
        # If fits in one chunk, return as is
        if tokens <= self.max_tokens:
            if tokens >= self.min_tokens:
                chunks.append({
                    'text': f"{section_id}\n{content}",
                    'tokens': tokens,
                    'section': section_id,
                    'decision': decision_header,
                    'type': 'decision_section'
                })
        else:
            # Need to split further
            subsections = group.get('subsections', [(section_id, content)])
            
            current_chunk_text = section_id
            current_chunk_tokens = TokenCounter.count_tokens(current_chunk_text)
            
            for sub_id, sub_content in subsections:
                sub_tokens = TokenCounter.count_tokens(sub_content)
                test_text = current_chunk_text + '\n' + sub_content
                test_tokens = TokenCounter.count_tokens(test_text)
                
                if test_tokens <= self.max_tokens:
                    current_chunk_text = test_text
                    current_chunk_tokens = test_tokens
                else:
                    # Save current chunk and start new one
                    if current_chunk_tokens >= self.min_tokens:
                        chunks.append({
                            'text': current_chunk_text,
                            'tokens': current_chunk_tokens,
                            'section': section_id,
                            'decision': decision_header,
                            'type': 'decision_section'
                        })
                    
                    current_chunk_text = f"{section_id}\n{sub_content}"
                    current_chunk_tokens = TokenCounter.count_tokens(current_chunk_text)
            
            # Don't forget last chunk
            if current_chunk_tokens >= self.min_tokens:
                chunks.append({
                    'text': current_chunk_text,
                    'tokens': current_chunk_tokens,
                    'section': section_id,
                    'decision': decision_header,
                    'type': 'decision_section'
                })
        
        return chunks
    
    def chunk(self, text: str, citation: str = "") -> List[Dict]:
        """
        Main entry point: chunk a court decision hierarchically.
        
        Returns: List of chunks with metadata
        """
        if not text or not isinstance(text, str):
            return []
        
        chunks = []
        
        try:
            # Extract sections
            sections = self.extract_decision_sections(text)
            
            if not sections:
                # Fallback: treat entire text as one
                tokens = TokenCounter.count_tokens(text)
                if tokens >= self.min_tokens:
                    return [{
                        'text': text,
                        'tokens': tokens,
                        'section': 'Full Text',
                        'decision': citation,
                        'type': 'full_text'
                    }]
                return []
            
            # Group subsections hierarchically
            grouped = self.group_subsections(sections)
            
            # Chunk each group
            for group in grouped:
                group_chunks = self.chunk_group(group, citation)
                chunks.extend(group_chunks)
        
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
        
        return chunks


class ProductionChunkingPipeline:
    """
    Unified pipeline for chunking both laws and court decisions.
    
    Usage:
        pipeline = ProductionChunkingPipeline()
        
        # Chunk laws
        laws_chunks_df = pipeline.chunk_laws_dataset(laws_df)
        
        # Chunk court decisions
        court_chunks_df = pipeline.chunk_court_decisions_dataset(court_df)
    """
    
    def __init__(self):
        self.laws_chunker = LawsSemanticChunker(target_tokens=200, max_tokens=300, min_tokens=50)
        self.court_chunker = CourtDecisionHierarchicalChunker(target_tokens=350, max_tokens=500, min_tokens=100)
    
    def chunk_laws_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chunk entire laws dataset.
        
        Input DataFrame columns: citation, text, title
        Output DataFrame columns: citation, title, chunk_id, text, tokens, article, chunk_type
        """
        all_chunks = []
        
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
        
        try:
            for idx, row in df.iterrows():
                citation = row.get("citation", f"law_{idx}")
                title = row.get("title", "")
                text = row.get("text", "")
                
                if not text or not isinstance(text, str):
                    continue
                
                # Chunk using semantic chunker
                chunks = self.laws_chunker.chunk(text, citation=citation)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'source_type': 'law',
                        'citation': citation,
                        'title': title,
                        'chunk_id': f"{citation}_chunk_{i}",
                        'text': chunk.get('text', ''),
                        'tokens': chunk.get('tokens', 0),
                        'article': chunk.get('article', ''),
                        'chunk_type': chunk.get('type', 'unknown'),
                        'position': i
                    })
        
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
        
        return pd.DataFrame(all_chunks)
    
    def chunk_court_decisions_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chunk entire court decisions dataset.
        
        Input DataFrame columns: citation, text
        Output DataFrame columns: citation, chunk_id, text, tokens, section, chunk_type
        """
        all_chunks = []
        
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty")
        
        try:
            for idx, row in df.iterrows():
                citation = row.get("citation", f"decision_{idx}")
                text = row.get("text", "")
                
                if not text or not isinstance(text, str):
                    continue
                
                # Chunk using hierarchical chunker
                chunks = self.court_chunker.chunk(text, citation=citation)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'source_type': 'court_decision',
                        'citation': citation,
                        'chunk_id': f"{citation}_chunk_{i}",
                        'text': chunk.get('text', ''),
                        'tokens': chunk.get('tokens', 0),
                        'section': chunk.get('section', ''),
                        'chunk_type': chunk.get('type', 'unknown'),
                        'position': i
                    })
        
        except Exception as e:
            raise Agentic_Exception(e, sys) from e
        
        return pd.DataFrame(all_chunks)
    
    def get_chunking_stats(self, chunks_df: pd.DataFrame) -> Dict:
        """Get statistics about chunks."""
        if chunks_df.empty:
            return {}
        
        return {
            'total_chunks': len(chunks_df),
            'total_tokens': chunks_df['tokens'].sum(),
            'avg_tokens_per_chunk': chunks_df['tokens'].mean(),
            'min_tokens': chunks_df['tokens'].min(),
            'max_tokens': chunks_df['tokens'].max(),
            'chunks_below_threshold': len(chunks_df[chunks_df['tokens'] < 50]),
            'chunks_above_threshold': len(chunks_df[chunks_df['tokens'] > 500])
        }
