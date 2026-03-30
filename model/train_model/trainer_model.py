import torch
import faiss
import joblib
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model.train.data_converter import PreprocessingData
from model_structure import TwoTowerModel
from softmax import SoftmaxLossWithCorrection
from eval_metrics import recall_at_k
from eval_metrics import ndcg_at_k
from eval_metrics import average_precision_at_k

# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============
# Training Model
# ==============
def trainer_model(df):
    # =========
    # Load Data
    # =========
    df["user_idx"] = df['user_idx'].astype(int)
    df["book_idx"] = df['book_idx'].astype(int)
    print(df.describe())

    # ====================
    # Prepare Loading Data
    # ====================
    # === Load Data as Batchs ===
    train_data, test_data = train_test_split(df, test_size=0.2)
    train_data_loader = DataLoader(PreprocessingData(train_data), batch_size=1500, shuffle=True)
    test_data_loader = DataLoader(PreprocessingData(test_data), batch_size=150)


    # =============
    # Prepare Model
    # =============
    # === Prepare Config ===
    num_users = df["user_idx"].max() + 1
    num_categ_ages = df["age_idx"].max() + 1
    num_locations = df["location_idx"].max() + 1
    num_publishers = df["publisher_idx"].max() + 1
    num_periods = df["year_production_idx"].max() + 1
    num_authors = df["author_idx"].max() + 1

    config = {
    "num_users": num_users,
    "num_categories_ages": num_categ_ages,
    "num_locations": num_locations,
    "num_publishers": num_publishers,
    "num_periods": num_periods,
    "num_authors": num_authors,
    }

    # === Prepare Model ===
    model = TwoTowerModel(num_users, num_categ_ages, num_locations, num_publishers, num_periods, num_authors)
    model.to(device)
    model.train()


    # ==============
    # Training Model
    # ==============
    # === Prepare Tools ===
    criterion = SoftmaxLossWithCorrection()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # === Training Model ===
    epochs = 70

    for epoch in range(epochs):
      loss_values = []
      for batch in train_data_loader:
        user_idx, book_idx, age, location, publisher, period, author = [x.to(device) for x in batch]

        # === Create User vec ===
        user_vec = model.user_tower(user_idx, age, location)

        # === Create Item Vec ===
        item_vec = model.item_tower(publisher, period, author)

        # === Compute Weights ===
        loss = criterion(user_vec, item_vec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Save Loss Value ===
        loss_values.append(loss.item())

      # === Compute Mean Loss ===
      print(f"Epoch_{epoch + 1}/{epochs}: {np.mean(loss_values)}")


    # ==============================
    # Storing Items Vectors in Faiss
    # ==============================
    # === Prepare Data ===
    items = df["book_idx"].unique().tolist()

    df_items = df.drop_duplicates("book_idx").set_index("book_idx")
    publishers = df_items.loc[items, "publisher_idx"].tolist()
    periods = df_items.loc[items, "year_production_idx"].tolist()
    authors = df_items.loc[items, "author_idx"].tolist()

    # === Preprocessing to Tensors ===
    publishers_tens = torch.tensor(publishers, dtype=torch.long, device=device)
    periods_tens = torch.tensor(periods, dtype=torch.long, device=device)
    authors_tens = torch.tensor(authors, dtype=torch.long, device=device)

    # === Compute Items Vectors ===
    model.eval()
    with torch.no_grad():
      item_vecs_np = model.item_tower(publishers_tens, periods_tens, authors_tens).cpu().numpy().astype("float32")

    items_np = np.array(items, dtype=np.int64)

    # === Stroring Vectors in Faiss ===
    dim = item_vecs_np.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(item_vecs_np, items_np)

    # === Storing Items Vectors ===
    mapping_item_idx_to_vec = {idx: vec for idx, vec in zip(items, item_vecs_np.tolist())}

    # === Storing Users Vectors ===
    users = df["user_idx"].unique().tolist()
    df_user = df.drop_duplicates("user_idx").set_index("user_idx")
    ages = df_user.loc[users, "age_idx"].tolist()
    locations = df_user.loc[users, "location_idx"].tolist()

    users_tens = torch.tensor(users, dtype=torch.long, device=device)
    ages_tens = torch.tensor(ages, dtype=torch.long, device=device)
    locations_tens = torch.tensor(locations, dtype=torch.long, device=device)

    # === Compute Base Users Vectors ===
    with torch.no_grad():
      users_vecs = model.user_tower(users_tens, ages_tens, locations_tens).cpu().numpy().tolist()

    mapping_user_idx_to_vec = {idx: vec for idx, vec in zip(users, users_vecs)}


    # ==========
    # Test Model
    # ==========
    # === Interacted Items ===
    interacted_items_dict = test_data.groupby("user_idx")["book_idx"].apply(list).to_dict()

    # === Measurements ===
    k = 10
    ndcg_k = []
    recall_k = []
    average_precision_k = []

    for batch in test_data_loader:
      user_idx, _, age, location, _, _, _ = [x.to(device) for x in batch]
      with torch.no_grad():
          user_vec = model.user_tower(user_idx, age, location).cpu().numpy().astype("float32")
      s, total_indices = index.search(user_vec, k)

      # === Evaluate Model ===
      for indices, user in zip(total_indices.tolist(), user_idx):
        interacted_items = interacted_items_dict[user.item()]

        # === Compute Evaluations ===
        ndcg = ndcg_at_k(indices, interacted_items, k)
        recall = recall_at_k(indices, interacted_items, k)
        average_precision = average_precision_at_k(indices, interacted_items, k)

        ndcg_k.append(ndcg)
        recall_k.append(recall)
        average_precision_k.append(average_precision)

    mean_ndcg = np.mean(ndcg_k)
    mean_recall = np.mean(recall_k)
    mean_average_precision = np.mean(average_precision_k)

    print(f"ndcg@{k}: {mean_ndcg:.2f}")
    print(f"recall@{k}: {mean_recall:.2f}")
    print(f"average_precision@{k}: {mean_average_precision:.2f}")


    # =========
    # Load Data
    # =========
    torch.save(model.state_dict(), "model/model_weights/model_weights.pt")
    joblib.dump(config, "model/model_weights/model_config.pt")
    joblib.dump(mapping_item_idx_to_vec, "storage/store/item_idx_to_vec.pkl")
    joblib.dump(mapping_user_idx_to_vec, "storage/store/user_idx_to_vec.pkl")
    faiss.write_index(index, "storage/store/faiss_index.bin")

    print("Done.")