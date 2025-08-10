import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("credit_cards_dataset.csv").fillna("")
model = SentenceTransformer('all-MiniLM-L6-v2')

def textify(row):
    return " ".join([str(row.get("Card Name","")), str(row.get("Description","")), str(row.get("Key Benefits","")), str(row.get("Tags","")), str(row.get("Eligibility",""))])

emb = model.encode([textify(r) for _, r in df.iterrows()], normalize_embeddings=True)
def recommend(query, k=5):
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = emb @ q
    idx = sims.argsort()[::-1][:k]
    return df.iloc[idx]

if __name__ == "__main__":
    print(recommend("cashback with lounge access under 1000 fee", k=3))
