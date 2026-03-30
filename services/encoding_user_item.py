from joblib import load 
from fastapi import HTTPException

# ============
# Load Mapping
# ============
mapping_user_id_to_idx = load("storage/store/user_id_to_idx.pkl")
mapping_item_id_to_idx = load("storage/store/items_id_to_idx.pkl")

def encoding_user_id(user_id):
    try:
        user_idx = mapping_user_id_to_idx.get(user_id, f"{user_id}-unknown")
        return user_idx
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Encode User ID: {e}")


def encoding_item_id(item_id):
    try:
        item_idx = mapping_item_id_to_idx.get(item_id, f"{item_id}-unknown")
        return item_idx
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Encode Item ID: {e}")
