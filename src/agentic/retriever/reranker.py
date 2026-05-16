import torch
import numpy as np
import pandas as pd
import sys
from sentence_transformers import CrossEncoder
from agentic.exception import Agentic_Exception

# Stage 1
STAGE1_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
STAGE2_MODEL = 'BAAI/bge-reranker-v2-m3'

MAX_LENGTH = 512
STAGE1_KEEP = 25
REGEX_BOOST = 1.25

class Reranker :
    """
    Use two reranker for research level searching.
    STAGE1:
    model: 
    input: 
    output: 

    STAGE2:
    model:
    input:
    output: 
    """
    def __init__(self, stage1_model: str = STAGE1_MODEL, stage2_model: str = STAGE2_MODEL, stage1_keep: int = STAGE1_KEEP, batch_size: int = 32, use_stage2: bool = True,) :
        try :
            self.stage1_keep = stage1_keep
            self.batch_size  = batch_size
            self.use_stage2  = use_stage2
            self.device      = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Reranker: device='{self.device}'")

            print(f"Loading Stage 1 model: {stage1_model}")
            self.stage1 = CrossEncoder(
                stage1_model,
                max_length=MAX_LENGTH,
                device=self.device,
            )

            if self.use_stage2:
                print(f"Loading Stage 2 model: {stage2_model}")
                self.stage2 = CrossEncoder(
                    stage2_model,
                    max_length=MAX_LENGTH,
                    device=self.device,
            )
            else:
    
                print("Stage 2 disabled — running Stage 1 only.")
        except Exception as e:
            raise Agentic_Exception(e, sys) from e      

     
    def _build_pairs(self, query: str, candidates: pd.DataFrame) -> list[tuple[str, str]]:
        """
        Build (query, text) pairs for the cross-encoder.
        Prepends citation to document text so the model sees the legal source.
        Truncates to ~600 words to stay within 512 token budget.
        Research best practice: ~64 tokens query + ~448 tokens document
        """
        pairs = []
        for _, row in candidates.iterrows():
            doc = f"[{row.get('citation', '')}] {str(row.get('text', ''))}"
            doc_words = doc.split()
            if len(doc_words) > 600:
                doc = " ".join(doc_words[:600])
            pairs.append((query, doc))
        return pairs


    def _score_in_batches(
        self, model: CrossEncoder, pairs: list[tuple[str, str]]
    ) -> np.ndarray:
        """
        Score all pairs in batches — never score one by one.
        Batching is mandatory for performance (research best practice).
        Returns raw logit scores as numpy array.
        """
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch        = pairs[i : i + self.batch_size]
            batch_scores = model.predict(batch, show_progress_bar=False)
            all_scores.extend(batch_scores.tolist())
        return np.array(all_scores)


    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        mn, mx = scores.min(), scores.max()
        if mx == mn:
            return np.ones_like(scores) * 0.5
        return (scores - mn) / (mx - mn)

    # Main Rank
    def rerank(self,query: str,candidates: pd.DataFrame,top_k: int = 10,) -> pd.DataFrame:
        """
        two-stage rerank of hybrid retrieval candiadates.
        Parameters
        query: str - expanded query from QuerayExpansion
        candidate: pd.DataFrame - output from HybridRetriever
        text, source - rrf_score, regex_match
        top_k : int - final number of result to return

        Outputs

        """
        try:
            if candidates.empty:
                print("Reranker: no candidates received.")
                return candidates
 
            print(f"\n{'='*60}")
            print(f"Reranker | {len(candidates)} candidates | top_k={top_k}")
            # Sanitize query for Windows console
            safe_query = query[:80].replace('\u2192', '->').replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
            print(f"Query: '{safe_query}'")
            print(f"{'='*60}")
 
            df = candidates.copy().reset_index(drop=True)


            # ── Stage 1: score ALL candidates ────────────────────────────────
            print(f"Stage 1 ({STAGE1_MODEL}):")
            print(f"  Scoring {len(df)} candidates in batches of {self.batch_size}...")
            pairs1       = self._build_pairs(query, df)
            raw_scores1  = self._score_in_batches(self.stage1, pairs1)
            norm_scores1 = self._normalize(raw_scores1)
            df['stage1_score'] = norm_scores1
 
            # Keep top STAGE1_KEEP for Stage 2
            keep_n      = min(self.stage1_keep, len(df))
            df_filtered = df.nlargest(keep_n, 'stage1_score').reset_index(drop=True)
            print(f"  Stage 1 done. Top score: {norm_scores1.max():.4f} | "
                  f"Passing top {len(df_filtered)} to Stage 2.")

            

            # ── Stage 2: score filtered candidates ───────────────────────────
            if self.use_stage2 and self.stage2 is not None:
                print(f"Stage 2 ({STAGE2_MODEL}):")
                print(f"  Scoring {len(df_filtered)} candidates...")
                pairs2       = self._build_pairs(query, df_filtered)
                raw_scores2  = self._score_in_batches(self.stage2, pairs2)
                norm_scores2 = self._normalize(raw_scores2)
                df_filtered['stage2_score'] = norm_scores2
                print(f"  Stage 2 done. Top score: {norm_scores2.max():.4f}")
 
                # Final score: Stage 2 = 70%, Stage 1 = 30%
                # Stage 2 is more accurate; Stage 1 catches edge cases
                df_filtered['final_score'] = (
                    0.70 * df_filtered['stage2_score'] +
                    0.30 * df_filtered['stage1_score']
                )
            else:
                df_filtered['stage2_score'] = 0.0
                df_filtered['final_score']  = df_filtered['stage1_score']
            

            # ── Regex boost ───────────────────────────────────────────────────
            # Citations explicitly found in the query text get a score boost.
            # Prevents a strong regex match from being buried by a
            # semantically similar but legally incorrect chunk.
            if 'regex_match' in df_filtered.columns:
                regex_mask = df_filtered['regex_match'] == True
                df_filtered.loc[regex_mask, 'final_score'] += REGEX_BOOST
                n_boosted = int(regex_mask.sum())
                if n_boosted > 0:
                    print(f"Regex boost (+{REGEX_BOOST}) applied to "
                          f"{n_boosted} citation-matched result(s).")
 
            # ── Final sort and top_k selection ────────────────────────────────
            df_final = df_filtered.nlargest(
                min(top_k, len(df_filtered)), 'final_score'
            ).reset_index(drop=True)
 
            df_final['final_rank'] = df_final.index + 1
 
            # ── Summary ───────────────────────────────────────────────────────
            law_hits   = int((df_final['source'] == 'law').sum()) \
                         if 'source' in df_final.columns else '?'
            court_hits = int((df_final['source'] == 'court').sum()) \
                         if 'source' in df_final.columns else '?'
 
            print(f"\nReranker complete:")
            print(f"  Final top-{len(df_final)} | law={law_hits} | court={court_hits}")
            # Sanitize citation for Windows console
            top_citation = str(df_final.iloc[0]['citation'])[:60].replace('\u2192', '->').replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
            print(f"  Top result: '{top_citation}' "
                  f"(final_score={df_final.iloc[0]['final_score']:.4f})")
 
            return df_final
 
        except Exception as e:
            raise Agentic_Exception(e, sys) from e

    def get_citations(self, reranked_df: pd.DataFrame) -> list[str]:
        """
        Extract final citation strings from reranked results.
        This is the output fed directly to your competition submission.
 
        Returns
        -------
        List of citation strings ordered by final_rank (best first).
        e.g. ["Art. 641 ZGB", "BGE 148 III 1", "Art. 184 OR"]
        """
        if reranked_df.empty:
            return []
        return reranked_df['citation'].tolist()
 