import random
import redis 
from joblib import load

from api.servers.functions.retrieve_recommendations.maximal_marginal_relevance import mmr_ranker_fast

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="localhost", port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)

# ============
# Load Mapping
# ============
mapping_item_idx_to_title = load("tools/encoders/item_idx_to_title.pkl")
mapping_item_idx_to_image = load("tools/encoders/book_idx_to_images_links.pkl")
mapping_items_idx_to_id = load("tools/encoders/items_idx_to_id.pkl")

# =======================
# Retrieve Common Items
# =======================
def retrieve_common_items(user_idx, k: int):
    # === Config ===
    key = "common_items"
    # === Retrieve Common Items === 
    common_items = r.zrevrange(key, 0, 250, withscores=False)

    # === Providing Recommendations ===
    common_items_1 = []
    if common_items:
        common_items_1 = [int(i) for i in common_items]
    
    # === Retrieve Interacted and watched Items ===
    key_interacted_items = f"saver_interaction:{user_idx}:interacted_items"
    
    interacted_items = r.lrange(key_interacted_items, 0, -1)

    # === Filter Common Items ===
    interacted_items = set([int(i) for i in interacted_items])

    common_items = []
    for common_item in common_items_1:
        if common_item not in interacted_items:
            common_items.append(common_item)
            if len(common_items) == k:
                break

    if len(common_items) < k:
        missing_num = k - len(common_items)
        all_items = set(mapping_items_idx_to_id.keys())
        filtered_items = list(all_items - interacted_items)
        if filtered_items >= missing_num:
            common_items.extend(random.sample(filtered_items, k=missing_num))
        else:
            common_items.extend(filtered_items)

    result = []
    for idx in  common_items: 
        img_dict = mapping_item_idx_to_image.get(idx, {})
        id = mapping_items_idx_to_id.get(idx, -1)
        if id == -1:
            continue
        result.append({
            "item_id": id,
            "title": mapping_item_idx_to_title.get(idx, "not-found"),
            "image_s": img_dict.get("image_url_s", "not-found"),
            "image_m": img_dict.get("image_url_m", "not-found"),
            "image_l": img_dict.get("image_url_l", "not-found")
        }) 

    return result