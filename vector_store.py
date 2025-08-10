import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi  # type: ignore
from rapidfuzz import fuzz  # type: ignore

_RETRIEVER_CACHE: Dict[str, "LocalRetriever"] = {}


class LocalRetriever:
    """
    BM25-based lexical retriever with strict bank/fee/category filters,
    keyword/bank bonuses, and fuzzy de-duplication. Optional dense
    hybrid can be added later without changing the API.
    """
    def __init__(self, df: Optional[pd.DataFrame], k: int = 10):
        self.k = k
        self.df = df.copy().fillna("") if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
        self._bm25 = None
        self._corpus: List[List[str]] = []
        if not self.df.empty:
            texts = [self._row_text(r) for _, r in self.df.iterrows()]
            self._corpus = [self._tokenize(t) for t in texts]
            self._bm25 = BM25Okapi(self._corpus)

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in str(text).lower().replace("/", " ").replace("-", " ").split() if t]

    def _row_text(self, row: pd.Series) -> str:
        return " ".join([
            str(row.get("Card Name", row.get("card_name",""))),
            str(row.get("Description", row.get("description",""))),
            str(row.get("Key Benefits", row.get("key_benefits",""))),
            str(row.get("Tags", row.get("tags",""))),
            str(row.get("Eligibility", row.get("eligibility",""))),
            str(row.get("Bank Name", row.get("bank_name",""))),
            str(row.get("Card Type", row.get("card_type",""))),
        ])

    def _apply_filters(self, df: pd.DataFrame, bank: Optional[str], max_fee: Optional[int],
                       categories: Optional[List[str]]) -> pd.DataFrame:
        out = df.copy()
        if bank:
            strict = out[out.apply(lambda r: bank.lower() in str(r.get("Bank Name", r.get("bank_name",""))).lower(), axis=1)]
            if not strict.empty:
                out = strict
        if max_fee:
            def fee_val(x):
                try: return int(str(x).replace(",",""))
                except: return 10**9
            col = "Annual Fee" if "Annual Fee" in out.columns else "annual_fee"
            if col in out.columns:
                out = out[out[col].apply(fee_val) <= int(max_fee)]
        if categories:
            cats = [c.lower() for c in categories]
            def in_cats(row):
                blob = " ".join([str(row.get("Tags","")), str(row.get("Key Benefits","")), str(row.get("Description",""))]).lower()
                return any(c in blob for c in cats)
            out = out[out.apply(in_cats, axis=1)]
        return out

    def _keyword_bonus(self, query: str, row: pd.Series) -> float:
        q = query.lower()
        txt = " ".join([
            str(row.get("Card Name","")),
            str(row.get("Description","")),
            str(row.get("Key Benefits","")),
            str(row.get("Tags",""))
        ]).lower()
        bonus = 0.0
        keyword_synonyms = [
            "cashback","travel","lounge","fuel","shopping","dining","online","groceries","rewards","airport",
            "airport lounge","priority pass","milestone","annual fee waiver","forex","foreign","international",
            "movie","movies","cinema","railway lounge","travel insurance","dining offers","fuel surcharge"
        ]
        for w in keyword_synonyms:
            if w in q and w in txt: bonus += 0.05
        return min(bonus, 0.20)

    def _bank_bonus(self, bank: Optional[str], row: pd.Series) -> float:
        if not bank: return 0.0
        b = str(row.get("Bank Name", row.get("bank_name",""))).lower()
        return 0.25 if bank.lower() in b else 0.0

    def search(self, query: str, bank: Optional[str]=None, max_fee: Optional[int]=None,
               categories: Optional[List[str]]=None, k: Optional[int]=None) -> pd.DataFrame:
        if self.df.empty or self._bm25 is None:
            return pd.DataFrame()
        k = k or self.k

        # 1) hard filters
        base = self._apply_filters(self.df, bank, max_fee, categories)
        if base.empty: base = self.df

        # 2) BM25 lexical scoring with soft bonuses
        idx_map = base.index.to_list()
        queries = self._tokenize(query)
        # compute BM25 on original order mapping
        # map filtered rows into corpus indices by content equality fallback
        # For simplicity, recompute tokens for filtered base
        base_tokens = [self._tokenize(self._row_text(r)) for _, r in base.iterrows()]
        bm25 = BM25Okapi(base_tokens)
        sims = bm25.get_scores(queries)

        rows: List[tuple[int, float]] = []
        for (i, row), sim in zip(base.iterrows(), sims):
            score = float(sim) + self._keyword_bonus(query, row) + self._bank_bonus(bank, row)
            rows.append((i, score))
        rows.sort(key=lambda x: x[1], reverse=True)

        # 3) diversity (distinct card names) with fuzzy grouping
        seen, picked = set(), []
        for i, _ in rows:
            name = str(base.loc[i].get("Card Name", base.loc[i].get("card_name",""))).strip().lower()
            if any((fuzz.partial_ratio(name, s) >= 92) for s in seen):
                continue
            seen.add(name); picked.append(i)
            if len(picked) >= max(k*2, 12):
                break
        candidates = base.loc[picked].copy()

        # 4) return top-k in normalized schema
        out = candidates.head(k).copy()
        return out.rename(columns={
            "Card Name":"card_name","Bank Name":"bank_name","Card Type":"card_type",
            "Tags":"tags","Website":"website","Description":"description",
            "Eligibility":"eligibility","Annual Fee":"annual_fee","Key Benefits":"key_benefits","FAQ":"faq"
        })


class CreditCardVectorStore:
    def __init__(self, force_reindex: bool=False, data_path: Optional[str] = None):
        self.data_path = data_path or os.getenv("CREDIT_CARD_DATA_PATH", "credit_cards_dataset.csv")
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path).fillna("")
        else:
            self.df = pd.DataFrame()
        cache_key = f"{self.data_path}::bm25"
        if (not force_reindex) and cache_key in _RETRIEVER_CACHE:
            self.retriever = _RETRIEVER_CACHE[cache_key]
        else:
            self.retriever = LocalRetriever(self.df)
            _RETRIEVER_CACHE[cache_key] = self.retriever

    def as_retriever(self, k: int = 10) -> LocalRetriever:
        self.retriever.k = k
        return self.retriever
