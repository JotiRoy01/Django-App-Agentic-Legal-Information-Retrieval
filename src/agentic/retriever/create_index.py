import torch
import pandas as pd
import numpy as np
import time
import faiss
from agentic.data_loader.data_loader import load
from sentence_transformers import SentenceTransformer, util


class Unified_Corpus() :
    def __init__(self, law:pd.DataFrame, court: pd.DataFrame) :
        # Re-initialize the embedding model 
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

        # We will create a NEW, unified FAISS index for BOTH laws and court cases
        embedding_dim = 384 # MiniLM dimension
        self.unified_faiss_index = faiss.IndexFlatIP(embedding_dim)

        # load the DataFrame
        self.laws_df = law
        self.court_df = court
        # We need a list to keep track the unified citation.
        self.unified_citations = []
    
    def add_law(self) :
        start_time = time.time()
        laws_texts = self.laws_df['text'].fillna("").tolist()
        laws_embeddings = self.embedding_model.encode(laws_texts, show_progress_bar=True)
        laws_embeddings = np.array(laws_embeddings).astype('float32')
        faiss.normalize_L2(laws_embeddings)
        self.unified_faiss_index.add(laws_embeddings)

        # Adding the laws_df citation to unified_citaiton
        self.unified_citations.extend(self.laws_df['citation'].tolist())
    
    def add_court(self) :
        chunk_size = 50
        chunk_count = 0

        # Extract text and fill NaNs
        for i in range(0, len(self.court_df), chunk_size) :
            chunk = self.court_df[i:i+chunk_size]
            texts = chunk['text'].fillna("").tolist()
            citations = chunk['citation'].tolist()

            # Encode the chunk
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False) # Turn off progress bar for cleaner logs
            embeddings = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings)    

            # Add to FAISS index
            self.unified_faiss_index.add(embeddings)

            # Add to unified citations list
            self.unified_citations.extend(citations)

        print(f"Final FAISS Index Size: {self.unified_faiss_index.ntotal} vectors.")
        print(f"Total Unified Citations Length: {len(self.unified_citations)}")

def create_unified_corpus(law:pd.DataFrame, court: pd.DataFrame, **kwargs) :

    unified = Unified_Corpus(law=law, court=court)
    unified.add_law()
    unified.add_court()
    return unified