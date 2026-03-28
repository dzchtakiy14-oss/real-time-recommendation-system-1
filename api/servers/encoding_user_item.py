from joblib import load 

# ============
# Load Mapping
# ============
mapping_user_id_to_idx = load("tools/encoders/user_id_to_idx.pkl")
mapping_item_id_to_idx = load("tools/encoders/items_id_to_idx.pkl")

def encoding_user_id(user_id):
    user_idx = mapping_user_id_to_idx.get(user_id, f"{user_id}-unknown")
    return user_idx

def encoding_item_id(item_id):
    item_idx = mapping_item_id_to_idx.get(item_id, f"{item_id}-unknown")
    return item_idx