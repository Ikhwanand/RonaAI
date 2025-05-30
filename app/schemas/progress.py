from pydantic import BaseModel
from datetime import datetime
from typing import List

class ProgressMetric(BaseModel):
    metric_name: str
    current_value: float
    previous_value: float
    improvement: float
    date: datetime

class ConcernProgress(BaseModel):
    concern_name: str
    severity_history: List[dict]
    current_severity: str
    improvement_percentage: float

class ProgressResponse(BaseModel):
    overall_improvement: float
    metrics: List[ProgressMetric]
    concerns_progress: List[ConcernProgress]
    comparison_dates: List[datetime]