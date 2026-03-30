from pydantic import BaseModel, Field
from typing import List

# ============================
# Recommendation Model
# ============================
class RecommendationRequest(BaseModel):
    user_id: int = Field(ge=0)
    
class RecommendedItemFeatures(BaseModel):
    item_id: str 
    title: str 
    image_s: str
    image_m: str
    image_l: str

class RecommendationResponse(BaseModel):
    user_id: int 
    recommendation: List[RecommendedItemFeatures]