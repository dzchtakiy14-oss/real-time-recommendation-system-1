from joblib import load
import time
import torch
import redis

from api.servers.encoding_user_item import encoding_user_id
from api.servers.functions.vectors_engine.retrieve_old_user_vec import retrieve_old_user_vec
from api.servers.functions.vectors_engine.last_interacted_items_vec import compute_interacted_items_vec
from api.servers.functions.vectors_engine.context_vec import compute_context_vec
from api.servers.functions.vectors_engine.retrieve_base_user_vec import retrieve_base_user_vec
from api.servers.functions.retrieve_recommendations.retrieve_common_items import retrieve_common_items
from api.servers.functions.retrieve_recommendations.provide_recommendation import providing_recommendation 
from model.model_structure import TwoTowerModel


# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="localhost", port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)


# =====================
# Recommendation Engine
# =====================
def recommendation_engine(user_id, k: int = 10):
    # === Config ===
    user_idx = encoding_user_id(user_id)
    curr_time = time.time()
    vectors = []
    weights = []

    # ====== Update User Vector =======
    # === Retrieve Old User Vector ===
    old_user_vec_tens = retrieve_old_user_vec(user_idx)
    if old_user_vec_tens is not None:
        vectors.append(old_user_vec_tens)
        weights.append(0.1)

    # === Retrieve Base Vector === 
    base_user_vec = retrieve_base_user_vec(user_idx)
    if base_user_vec is not None:
        vectors.append(base_user_vec)
        weights.append(0.1)

    # === Compute Context Vec ===
    context_vec_tens = compute_context_vec(curr_time)
    if context_vec_tens is not None:
        vectors.append(context_vec_tens)
        weights.append(0.3)

    # === Compute Interacted Items Vec ===
    interacted_items_vec_tens = compute_interacted_items_vec(user_idx)
    if interacted_items_vec_tens is not None:
        vectors.append(interacted_items_vec_tens)
        weights.append(0.5)
    
    # === Integrate Vectors with Weights ===
    weights_tens = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
    vectors_tens = torch.stack(vectors).to(device)

    weighted_vectors = vectors_tens * weights_tens

    # === Create New User Vec ===
    with torch.no_grad():
        new_user_vec = torch.sum(weighted_vectors, dim=0).cpu().numpy()

    # === Storing New User Vec === 
    key = f"old_user_vec:{user_idx}"
    r.set(key, new_user_vec[0].tobytes())

    # === Providing Recommendations ===
    recommendations = providing_recommendation(user_idx, new_user_vec, k)
    
    return recommendations