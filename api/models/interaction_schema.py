from pydantic import BaseModel, Field
from typing import List

# =========================
# Saving Interactions Model
# =========================
class InteractionRequest(BaseModel):
    user_id: int = Field(ge=0)
    item_id: int = Field(ge=0)
    event_type: str 