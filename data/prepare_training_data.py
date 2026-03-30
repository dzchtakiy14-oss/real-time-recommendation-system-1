import pandas as pd

from data.processor_data import processor_data

# =====================
# Prepare Training Data
# =====================
# === Load Data ===
df_user = pd.read_csv("")
df_book = pd.read_csv("")
df_rating = pd.read_csv("")

processor_data(df_user, df_book, df_rating)