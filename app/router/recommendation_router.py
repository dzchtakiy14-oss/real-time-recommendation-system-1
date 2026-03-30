import time 
from fastapi import APIRouter 

from app.models.recommendation_schemas import RecommendationRequest
from app.models.recommendation_schemas import RecommendationResponse
from app.models.interaction_schema import InteractionRequest
from services.recommendation_engine import recommendation_engine
from services.saver_interactions import saving_interactions

# ==============
# Prepare Router
# ==============
router = APIRouter()

# =====================
# Recommendation Router
# =====================
@router.post("/recommendations", response_model=RecommendationResponse)
def recommendation_service(request: RecommendationRequest):
    st = time.perf_counter()
    recommendation = recommendation_engine(request.user_id)
    en = time.perf_counter()
    print(f"total_time_recommendation: {en - st}")
    return {
        "user_id": request.user_id,
        "recommendation": recommendation
    }

@router.post("/interactions")
def interaction_service(interaction: InteractionRequest):
    st = time.perf_counter()
    msg = saving_interactions(interaction.user_id, interaction.item_id, interaction.event_type)
    en = time.perf_counter()
    print(f"total_time_saving_interaction: {en - st}")
    return msg