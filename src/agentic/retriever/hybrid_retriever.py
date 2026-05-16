import numpy as np
import pandas as pd
import sys
import faiss
from typing import List, Tuple
 
from agentic.retriever.BM25_retriever import BM25
from agentic.retriever.faiss_retriever import Faiss
from agentic.retriever.create_index import Unified_Corpus
from agentic.retriever.retriever_with_regex import extract_and_clean_citations
from agentic.exception import Agentic_Exception



def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60
) -> dict[int, float]:
    """
    Fuse multiple ranked lists of corpus row-indices into a single RRF score.
 
    Args:
        ranked_lists : Each inner list is corpus row-indices, ordered best-first.
                       Lists can be different lengths — that is fine.
        k            : RRF smoothing constant. Default 60.
 
    Returns:
        dict mapping corpus_index -> rrf_score (higher = more relevant).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores

# Hybrid Retriever
class HybridRetriever:
    def __init__(
        self,
        law_df: pd.DataFrame,
        val_df: pd.DataFrame,
        unified_corpus: Unified_Corpus,
        expanded_query: str,
        query_text: str,
        top_k: int = 100,
        rrf_k: int = 60,
    ):
        try:
            self.law_df         = law_df
            self.val_df         = val_df
            self.unified_corpus = unified_corpus
            self.expanded_query = expanded_query
            self.query_text     = query_text
            self.rrf_k          = rrf_k
            self.top_k          = top_k
            self.law_size       = len(self.law_df)
            self.court_df       = unified_corpus.court_df

            self.bm25_retriever = BM25(self.law_df, self.val_df)
            self.faiss_retriever = Faiss(law=self.law_df, expanded_query=self.expanded_query)
        except Exception as e:
            raise Agentic_Exception(e, sys) from e


    def _regex_retrieve(self) -> list[int]:
        """
        Use extract_and_clean_citations() on the raw query to get citation strings.
        Then look those strings up in corpus['citation'] to get row indices.
        Returns a list of matched row indices (treated as highest-confidence hits).
        If no citations are found in the query, returns an empty list.
        """
        # Call your existing function directly
        citation_strings = extract_and_clean_citations(self.query_text)
 
        if not citation_strings:
            return [], []

        law_indices = []
        for cit in citation_strings:
            mask = self.law_df["citation"].str.contains(
                cit, case=False, na=False, regex=False
            )
            law_indices.extend(self.law_df.index[mask].tolist())

        # Deduplicate while preserving order
        law_indices = list(dict.fromkeys(law_indices))
        court_indices = []

        return law_indices, court_indices

    def _bm25_retrieve(self) -> list[int] :
        """
        Extract the row of dataFrame based on frequency of keyword.
        return the indices of the law_de dataframe
        """
        indices = self.bm25_retriever.Best_Match()
        
        return indices.tolist()
    
    def _law_faiss_retrieve(self) -> list[int] :
        """
        Extract the row based on fiass similarity of law_de and expandend_query
        return the indices which has mostly like to the query
        """
        _, dense_indices_full = self.faiss_retriever.Faiss_retriever()
        return dense_indices_full[0].tolist()

    def _unified_faiss_retrieve(self, top_k: int = 100) -> Tuple[List[int], List[int]] :
        """
        Search the pre-build unified Faiss Index(law+court_considertion)
        Return two seperate ranked list
        law_indices - unified result that map to law_de
        court_indices - unified result that map to court_considerations
        """
        query_emb = self.unified_corpus.embedding_model.encode([self.expanded_query])
        query_emb = np.array(query_emb).astype('float32')
        faiss.normalize_L2(query_emb)

        k = min(top_k, self.unified_corpus.unified_faiss_index.ntotal)

        _, unified_indices = self.unified_corpus.unified_faiss_index.search(query_emb, k)

        # unified_indices shape (1, k) - take row 0
        indices = unified_indices[0].tolist()

        # Split law and court based on position on unified_citation
        # Position 0..law_size - 1 = law rows and law_size..
        law_indices = [i for i in indices if i < self.law_size]
        court_indices = [i for i in indices if i >= self.law_size]

        return law_indices, court_indices

    def retrieve(self, top_k: int = None) -> pd.DataFrame:
        """
        Wrapper for public API. Allows caller to override top_k if needed.
        """
        if top_k is not None:
            self.top_k = top_k
        return self.retriever()

    def retriever(self) ->list[dict] :
        """
        Use regex, BM25 and Fiass to retrieve the citation from the corpus
        Load BM25 from the BM25_retriever module.
        Load Fiass from the faiss_retriever module

        Return 
        -------
        List of dicts with keys:
        citation: str
        text: str
        source: str
        rrf_score: float
        rank: int
        BM25: int or None
        faiss_rank: int or None
        unified_fiss_rank: int or None
        regex_match: bool
        """

        try :
            print(f"\n{'='*60}")
            print(f"Hybrid Retriever | 4 retrievers | top_k= {self.top_k}")
            # Sanitize Unicode for Windows console
            safe_query = self.query_text[:80].replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
            safe_expanded = self.expanded_query[:80].replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
            print(f"Query     : '{safe_query}'\n")
            print(f"Expanded:   '{safe_expanded}'\n")
            # step 1: regex retriever
            regex_law_ranked, regex_court_ranked = self._regex_retrieve()

            # -- step 2: BM25 index searching
            bm25_law_ranked = self._bm25_retrieve()
            
            # -- step 3: Faiss index searching
            law_faiss_ranked = self._law_faiss_retrieve()

            # -- step 4: combine laws_de and court_considerations for searching
            unified_law_ranked, unified_court_ranked = self._unified_faiss_retrieve(top_k=self.top_k)


            print(f"BM25 (law only)       : {len(bm25_law_ranked)} candidates")
            print(f"FAISS (law only)      : {len(law_faiss_ranked)} candidates")
            print(f"FAISS unified (law)   : {len(unified_law_ranked)} candidates")
            print(f"FAISS unified (court) : {len(unified_court_ranked)} candidates")

            # -- step 4: Fuse all law candidates with RRF
            law_ranked_list = []
            if regex_law_ranked:
                law_ranked_list.append(regex_law_ranked)
            law_ranked_list.extend([bm25_law_ranked, law_faiss_ranked, unified_law_ranked])

            law_rrf_score = reciprocal_rank_fusion(ranked_lists=law_ranked_list, k=self.rrf_k)

            # Step 5: RRF over court index space 
            court_ranked_lists = []
            if regex_court_ranked:
                court_ranked_lists.append(regex_court_ranked)
            if unified_court_ranked:
                court_ranked_lists.append(unified_court_ranked)

            court_rrf_scores = (
                reciprocal_rank_fusion(court_ranked_lists, k=self.rrf_k)
                if court_ranked_lists else {}
            )

            # step 6: rank lookup for diagnostics
            bm25_rank_lookup = {idx: r+1 for r, idx in enumerate(bm25_law_ranked)}
            law_faiss_rank_lookup = {idx: r+1 for r, idx in enumerate(law_faiss_ranked)}
            unified_indices = unified_law_ranked + unified_court_ranked
            faiss_rank_unified_lookup = {idx: r+1 for r, idx in enumerate(unified_indices)}

            regex_law_set   = set(regex_law_ranked)
            regex_court_set = set(regex_court_ranked)

            # ── Step 5: Combine law + court into one pool, sort by rrf_score ──
            all_candidates = [
                ('law',   idx, score) for idx, score in law_rrf_score.items()
            ]
            all_candidates += [
                ('court', idx, score) for idx, score in court_rrf_scores.items()
            ]
            all_candidates.sort(key=lambda x: x[2], reverse=True)

            # step 7: build final result list
            results = []

            for final_rank, (source, idx, score) in enumerate(all_candidates[:self.top_k], start=1):
                if source == "law":
                    row = self.law_df.iloc[idx]
                    citation = row.get('citation', f'law_idx_{idx}')
                    text = str(row.get('text', ''))
                    is_regex = idx in regex_law_set
                else:
                    citation = self.unified_corpus.unified_citations[idx]
                    court_offset = idx - self.law_size
                    court_row = self.court_df.iloc[court_offset]
                    text = str(court_row.get('text', ''))
                    is_regex = idx in regex_court_set

                results.append({
                    'citation':           citation,
                    'text':               text,
                    'source':             source,
                    'rrf_score':          round(score, 6),
                    'rank':               final_rank,
                    'bm25_rank':          bm25_rank_lookup.get(idx) if source == 'law' else None,
                    'law_faiss_rank':     law_faiss_rank_lookup.get(idx) if source == 'law' else None,
                    'unified_faiss_rank': faiss_rank_unified_lookup.get(idx),
                    'regex_match':        is_regex,
                })
            # ── Summary
            law_hits   = sum(1 for r in results if r['source'] == 'law')
            court_hits = sum(1 for r in results if r['source'] == 'court')
            regex_hits = sum(1 for r in results if r['regex_match'])
            print(f"\nRetrieved {len(results)} | "
                  f"law={law_hits} | court={court_hits} | regex_confirmed={regex_hits}")

            return pd.DataFrame(results)

        except Exception as e :
            raise Agentic_Exception(e, sys) from e

