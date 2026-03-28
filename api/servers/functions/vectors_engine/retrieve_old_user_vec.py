import redis 
import torch
import numpy as np


# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="localhost", port=6379, db=0, decode_responses=False)
r = redis.Redis(connection_pool=pool)

# =====================
# Retrieve Old User Vec
# =====================
def retrieve_old_user_vec(user_idx):
    # === Config ===
    key = f"old_user_vec:{user_idx}"
    
    # === Retrieve Old Vector ===
    old_vec_byte = r.get(key)
    if not old_vec_byte:
        return None
    
    # === Convert to "array" ===
    old_vec_np = np.frombuffer(old_vec_byte, dtype=np.float32)

    # === Convert to "tensor" ===
    old_vec_tens = torch.tensor(old_vec_np, dtype=torch.float32, device=device).unsqueeze(0)

    return old_vec_tens
