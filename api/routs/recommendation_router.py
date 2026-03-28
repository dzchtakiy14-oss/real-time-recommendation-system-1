from fastapi import APIRouter 

from api.models.recommendation_schemas import RecommendationRequest
from api.models.recommendation_schemas import RecommendationResponse
from api.models.interaction_schema import InteractionRequest
from api.servers.recommendation_engine import recommendation_engine
from api.servers.saver_interactions import saving_interactions

# ==============
# Prepare Router
# ==============
router = APIRouter()

# =====================
# Recommendation Router
# =====================
@router.post("/recommendations", response_model=RecommendationResponse)
def recommendation_service(request: RecommendationRequest):
    recommendation = recommendation_engine(request.user_id)
    return {
        "user_id": request.user_id,
        "recommendation": recommendation
    }

@router.post("/interactions")
def interaction_service(interaction: InteractionRequest):
    msg = saving_interactions(interaction.user_id, interaction.item_id, interaction.event_type)
    return msg