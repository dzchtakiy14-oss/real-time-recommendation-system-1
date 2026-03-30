from pydantic import BaseModel, Field
from typing import List

# =========================
# Saving Interactions Model
# =========================
class InteractionRequest(BaseModel):
    user_id: int = Field(ge=0)
    item_id: str
    event_type: str 