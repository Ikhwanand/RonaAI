from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class Concern(BaseModel):
    name: str
    severity: str
    type: Optional[str]
    confidence: float

class Recommendation(BaseModel):
    title: str
    description: str
    priority: str

class AnalysisMetrics(BaseModel):
    skin_hydration: int
    texture_uniformity: int
    pore_visibility: int
    overall_score: int

class AnalysisCreate(BaseModel):
    overall_health: str
    skin_type: str
    concerns: List[Concern]
    recommendations: List[Recommendation]
    analysis_metrics: AnalysisMetrics

class AnalysisResponse(AnalysisCreate):
    id: int
    user_id: int
    image_url: str
    created_at: datetime

    class Config:
        from_attributes = True