import faiss
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

class Faiss :
    def __init__(self, law: pd.DataFrame, expanded_query: str) :
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.laws_df = law
        #self.val = val
        self.expanded_query = expanded_query
    
    def Faiss_retriever(self) :
        """
        Fiass_retirever return tow value matrix:
        dense_scores_full = the dimension of the dense_scores_full matrix (number_of_query, top_k). it contain the score of query and corpus dot product.
        dense_indices_full = the dimension of the dense_indices_full matrix (number_of_query, top_k). it contain row index of the corpus dataframe. it's values sorted by dense_scores_full values.
        """
        full_corpus_embeddings = self._Encode_corpus()
        embedding_dim = full_corpus_embeddings.shape[1]
        # Normalize vector for inner product as consine similarity
        faiss.normalize_L2(full_corpus_embeddings)
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(full_corpus_embeddings)

        # Normalize the query vector
        query_emb_full = self._Encode_query()
        faiss.normalize_L2(query_emb_full)
        top_k_search = 100
        
        # Vector search by the faiss
        dense_scores_full, dense_indices_full = index.search(query_emb_full, top_k_search)
        return dense_scores_full, dense_indices_full

    def _Encode_corpus(self) :
        full_corpus_embeddings = self.embedding_model.encode(self.laws_df["text"].fillna("").tolist(), show_progress_bar = True)
        full_corpus_embeddings = np.array(full_corpus_embeddings).astype("float32")
        return full_corpus_embeddings
        
    def _Encode_query(self) :
        query_emb_full = self.embedding_model.encode([self.expanded_query])
        query_emb_full = np.array(query_emb_full).astype("float32")
        return query_emb_full