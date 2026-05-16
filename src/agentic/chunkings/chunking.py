import re
from typing import List, Dict, Tuple
import pandas as pd
from agentic.exception import Agentic_Exception
import sys
# from agentic.data_loader import DataLoader
# from agentic.data_loader import load, load_from_config

# laws_df = load(filename="laws_de.csv", nrows=1000)


# Step 1: split by legal markers (Swiss/German legal articles)
def split_by_articles(text: str) -> list[Dict[str, str]]:
    """
    Split legal text by articles. Handles formats like:
    - Art. 1, Art. 1a, Art. 1 Abs. 1, Abs. 2, § 1, etc.
    Returns list of dicts with article info and content
    """
    # Enhanced pattern for German/Swiss legal format
    article_pattern = r"((?:Art\.?|§)\s*\d+[a-zA-Z]*(?:\s+Abs\.?\s*\d+)?)"
    parts = re.split(article_pattern, text)

    chunks = []
    for i in range(1, len(parts), 2):
        article_header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        
        if content:  # Only include if there's content
            chunks.append({
                "article": article_header,
                "content": content,
                "full_text": article_header + " " + content
            })

    return chunks if chunks else [{"article": "Full Text", "content": text, "full_text": text}]

# Step 2: fallback paragraph split (intelligent splitting for legal text)
def split_by_paragraph(text: str, min_length: int = 50) -> list[str]:
    """
    Split by paragraphs, handling legal text structure.
    Removes empty lines and filters out very short fragments.
    """
    # Split by multiple newlines (paragraph breaks) or numbered points
    paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.)', text)
    
    result = []
    for p in paragraphs:
        p_clean = p.strip()
        # Only include paragraphs with reasonable length
        if len(p_clean) > min_length:
            result.append(p_clean)
    
    return result if result else [text.strip()]

# Step 3: token chunking with sliding window (improved)
def token_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Chunk text by tokens with sliding window overlap.
    chunk_size: target characters per chunk
    overlap: overlapping characters between chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)

        # Calculate overlap in words
        if end >= len(words):
            break
            
        # Move start position considering overlap
        overlap_words = max(1, overlap // 5)  # Rough conversion: assume avg 5 chars per word
        start = end - overlap_words

    return chunks

# Step 4: Hybrid pipeline with metadata
def hybrid_chunk(text: str, article_info: Dict = None) -> list[Dict[str, str]]:
    """
    Intelligent chunking strategy for legal text:
    1. First split by articles (if found)
    2. Then by paragraphs (if chunk too large)
    3. Finally by tokens (if still too large)
    
    Returns list of chunks with metadata.
    """
    max_chunk_words = 300
    final_chunks = []
    
    try:
        # Try article split first
        article_chunks = split_by_articles(text)
        
        for article in article_chunks:
            article_header = article.get("article", "Full Text")
            content = article.get("full_text", "")
            
            if len(content.split()) <= max_chunk_words:
                # Chunk is small enough
                final_chunks.append({
                    "text": content,
                    "article": article_header,
                    "parent_section": article_info if article_info else article_header
                })
            else:
                # Try paragraph split
                paragraphs = split_by_paragraph(content)
                
                for para in paragraphs:
                    if len(para.split()) <= max_chunk_words:
                        final_chunks.append({
                            "text": para,
                            "article": article_header,
                            "parent_section": article_info if article_info else article_header
                        })
                    else:
                        # Final fallback: token chunking
                        token_chunks = token_chunk(para, chunk_size=max_chunk_words, overlap=50)
                        for token_chunk_text in token_chunks:
                            if token_chunk_text.strip():  # Filter empty chunks
                                final_chunks.append({
                                    "text": token_chunk_text,
                                    "article": article_header,
                                    "parent_section": article_info if article_info else article_header
                                })
                                
    except Exception as e:
        raise Agentic_Exception(e, sys) from e

    return final_chunks if final_chunks else [{
        "text": text,
        "article": "Full Text",
        "parent_section": article_info if article_info else "Full Text"
    }]

# Step 5: Apply to DataFrame with enriched metadata
def build_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Chunking on the DataFrame. Creates a structured dataset with chunks.
    
    Returns DataFrame with columns:
    - citation: Reference to the law
    - title: Title of the law document
    - chunk_id: Unique chunk identifier
    - text: The chunk content
    - article: Article/section reference
    - parent_section: Parent section for context
    """
    all_chunks = []
    
    if df is None:
        raise FileNotFoundError("DataFrame is None")
    
    try:
        for idx, row in df.iterrows():
            citation = row.get("citation", "")
            title = row.get("title", "")
            text = row.get("text", "")
            
            if not text or not isinstance(text, str):
                continue
            
            chunks = hybrid_chunk(text, article_info=citation)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "citation": citation,
                    "title": title,
                    "chunk_id": f"{citation}_{i}",
                    "text": chunk.get("text", ""),
                    "article": chunk.get("article", ""),
                    "parent_section": chunk.get("parent_section", "")
                })
    
    except Exception as e:
        raise Agentic_Exception(e, sys) from e

    return pd.DataFrame(all_chunks)