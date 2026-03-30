from joblib import load 
import torch

# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============
# Load Mapping
# ============
mapping_user_idx_to_vec = load("storage/store/user_idx_to_vec.pkl")

def retrieve_base_user_vec(user_idx:int):
    base_user_vec = mapping_user_idx_to_vec.get(user_idx, None)
    if base_user_vec is None:
        return None 
    base_user_vec_tens = torch.tensor(base_user_vec, dtype=torch.float32, device=device).unsqueeze(0)
    return base_user_vec_tens