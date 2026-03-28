from joblib import load
from fastapi import HTTPException 
import pandas as pd
import numpy as np
import torch

from model.model_structure import TwoTowerModel

# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============
# Prepare Model
# =============
config = load("model/model_weights/model_config.pt")
model = TwoTowerModel(
    config["num_users"],
    config["num_categories_ages"],
    config["num_locations"],
    config["num_publishers"],
    config["num_periods"],
    config["num_authors"]
)

model.load_state_dict(torch.load("model/model_weights/model_weights.pt", map_location=device))
model.eval()

# ======================
# Compute Context Vector
# ======================
def compute_context_vec(curr_time: int, status=True):
    if status is None:
        return None
    try:
        # === Compute Context Features ===
        hour = (curr_time // 3600) % 24
        hour_cos = np.cos(np.pi * 2 * hour / 24)
        hour_sin = np.sin(np.pi * 2 * hour / 24)

        day = (curr_time // 3600 * 24) % 7
        day_cos = np.cos(np.pi * 2 * day / 7)
        day_sin = np.sin(np.pi * 2 * day / 7)

        month = pd.to_datetime(curr_time, unit="s").month 
        month_cos = np.cos(np.pi * 2 * month / 12)
        month_sin = np.sin(np.pi * 2 * month / 12)

        # === Transforming to Tensors 
        hour_cos_tens = torch.tensor(hour_cos, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        hour_sin_tens = torch.tensor(hour_sin, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        day_cos_tens = torch.tensor(day_cos, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        day_sin_tens = torch.tensor(day_sin, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        month_cos_tens = torch.tensor(month_cos, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        hmonth_sin_tens = torch.tensor(month_sin, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

        # === Create Context Vector ===
        with torch.no_grad():
            context_vec = model.compute_context(hour_cos_tens, hour_sin_tens, day_cos_tens, day_sin_tens, month_cos_tens, hmonth_sin_tens)

        return context_vec

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed To Compute Context Vector: {e}") 
