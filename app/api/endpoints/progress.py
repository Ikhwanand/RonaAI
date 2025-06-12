from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.analysis import Analysis
from app.models.users import User
from app.core.security import get_current_user
from app.schemas.responses import APIResponse


router = APIRouter()

@router.get("/metrics/{analysis_id}", response_model=APIResponse)
async def get_progress_metrics(analysis_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get progress metrics comparing with previous analysis"""
    # Get current analysis
    current_analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.user_id == current_user.id
    ).first()

    if not current_analysis:
        return APIResponse(
            success=False,
            message="Analysis not found",
            data=None
        )
    
    # Get previous analysis for comparison
    previous_analysis = db.query(Analysis).filter(
        Analysis.user_id == current_user.id,
        Analysis.created_at < current_analysis.created_at
    ).order_by(Analysis.created_at.desc()).first()

    if not previous_analysis:
        # Return current analysis data without comparison
        return APIResponse(
            success=True,
            message="First analysis - no previous data for comparison",
            data={
                "analysis_id": current_analysis.id,
                "is_first_analysis": True,
                "current_metrics": {
                    "skin_hydration": current_analysis.analysis_metrics.get("skin_hydration", 0),
                    "texture_uniformity": current_analysis.analysis_metrics.get("texture_uniformity", 0),
                    "pore_visibility": current_analysis.analysis_metrics.get("pore_visibility", 0),
                    "overall_score": current_analysis.analysis_metrics.get("overall_score", 0)
                },
                "concerns": current_analysis.concerns
            }
        )
    
    # Calculate improvement
    improvement_areas = []
    if current_analysis.analysis_metrics.get("skin_hydration", 0) > previous_analysis.analysis_metrics.get("skin_hydration", 0):
        improvement_areas.append("Hydration")
    if current_analysis.analysis_metrics.get("texture_uniformity", 0) > previous_analysis.analysis_metrics.get("texture_uniformity", 0):
        improvement_areas.append("Texture")
    if current_analysis.analysis_metrics.get("pore_visibility", 0) > previous_analysis.analysis_metrics.get("pore_visibility", 0):
        improvement_areas.append("Pores")
    if current_analysis.analysis_metrics.get("overall_score", 0) > previous_analysis.analysis_metrics.get("overall_score", 0):
        improvement_areas.append("Overall Score")
    
    # Format concerns progress
    concerns_progress = []
    for current_concern in current_analysis.concerns:
        previous_concern = next(
            (c for c in previous_analysis.concerns if c["name"] == current_concern["name"]),
            None
        )
        if previous_concern:
            concerns_progress.append({
                "name": current_concern["name"],
                "previous_severity": previous_concern["severity"],
                "current_severity": current_concern["severity"],
                "improved": current_concern["severity"] < previous_concern["severity"]
            })
        else:
            # New concern that wasn't in previous analysis
            concerns_progress.append({
                "name": current_concern["name"],
                "current_severity": current_concern["severity"],
                "is_new": True
            })
    
    progress_data = {
        "analysis_id": current_analysis.id,
        "comparison_date": previous_analysis.created_at.strftime("%Y-%m-%d"),
        "improvement_areas": improvement_areas,
        "concerns_progress": concerns_progress,
        "metrics_comparison": {
            "skin_hydration": {
                "previous": previous_analysis.analysis_metrics.get("skin_hydration", 0),
                "current": current_analysis.analysis_metrics.get("skin_hydration", 0),
                "improved": current_analysis.analysis_metrics.get("skin_hydration", 0) > previous_analysis.analysis_metrics.get("skin_hydration", 0)
            },
            "texture_uniformity": {
                "previous": previous_analysis.analysis_metrics.get("texture_uniformity", 0),
                "current": current_analysis.analysis_metrics.get("texture_uniformity", 0),
                "improved": current_analysis.analysis_metrics.get("texture_uniformity", 0) > previous_analysis.analysis_metrics.get("texture_uniformity", 0)
            },
            "pore_visibility": {
                "previous": previous_analysis.analysis_metrics.get("pore_visibility", 0),
                "current": current_analysis.analysis_metrics.get("pore_visibility", 0),
                "improved": current_analysis.analysis_metrics.get("pore_visibility", 0) > previous_analysis.analysis_metrics.get("pore_visibility", 0)
            },
            "overall_score": {
                "previous": previous_analysis.analysis_metrics.get("overall_score", 0),
                "current": current_analysis.analysis_metrics.get("overall_score", 0),
                "improved": current_analysis.analysis_metrics.get("overall_score", 0) > previous_analysis.analysis_metrics.get("overall_score", 0)
            }
        }
    }

    return APIResponse(
        success=True,
        message="Progress metrics retrieved successfully.",
        data=progress_data
    )