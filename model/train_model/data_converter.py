import torch
from torch.utils.data import Dataset

# === Preprocessing Data ===
class PreprocessingData(Dataset):
    def __init__(self, df):
        self.df = df
        self.df_column = df["user_idx"]

    def __len__(self):
        return len(self.df_column)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            torch.tensor(row["user_idx"], dtype= torch.long),
            torch.tensor(row["book_idx"], dtype= torch.long),
            torch.tensor(row["age_idx"], dtype= torch.long),
            torch.tensor(row["location_idx"], dtype= torch.long),
            torch.tensor(row["publisher_idx"], dtype= torch.long),
            torch.tensor(row["year_production_idx"], dtype= torch.long),
            torch.tensor(row["author_idx"], dtype= torch.long),
        )