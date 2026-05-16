from rank_bm25 import BM25Okapi
import time
import pandas as pd
import numpy as np
from agentic.data_loader.data_loader import load


class BM25 :
    """
    Best Match technique followed the TF-IDF.
    BM match two sentence based on their keyword frequency.
    It return the Best matching 25 index value.
    """
    def __init__(self, laws_de:pd.DataFrame, val:pd.DataFrame) :
        self.corpus = laws_de
        self.val = val

    def Best_Match(self) -> list:   
        print("Preparing BM25 on the FULL laws corpus...")
        start_time = time.time()

        # Simple tokenization: lowercasing and splitting by space 
        # We fill NaN values with empty string to avoid errors
        tokenized_corpus = [str(doc).lower().split() for doc in self.corpus['text'].fillna("")]

        # Initialize BM25 model
        bm25 = BM25Okapi(tokenized_corpus)

        print(f"BM25 indexing finished in {time.time() - start_time:.2f} seconds.")

        # Prepare the same test query 
        test_query = self.val['query'].iloc[0]
        tokenized_query = test_query.lower().split()

        print("\nSearching with BM25...")
        # Get top 3 scores and their indices 
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k = 100
        # np.argsort returns indices in ascending order, we take the last 'top_k' and reverse it
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        return top_indices
