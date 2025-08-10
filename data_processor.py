import os
import pandas as pd

class CreditCardDataProcessor:
    def __init__(self, csv_file: str = None):
        self.csv = csv_file or os.getenv("CREDIT_CARD_DATA_PATH", "credit_cards_dataset.csv")
        self.df = pd.read_csv(self.csv).fillna("") if os.path.exists(self.csv) else pd.DataFrame()

    def get_df(self) -> pd.DataFrame:
        return self.df.copy()
