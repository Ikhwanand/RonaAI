from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session 

from app.db.database import get_db
from app.models.skin import Skin
from app.models.users import User 
from app.schemas.skin import SkinCreate, SkinResponse, SkinUpdate
from app.schemas.responses import APIResponse
from app.core.security import get_current_user
from fastapi import HTTPException, status


router = APIRouter()


@router.get("/get-profile-skin", response_model=APIResponse)
async def get_profile_skin(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user profile skin"""
    query = db.query(Skin).filter(
        Skin.user_id == current_user.id
    ).first()

    if not query:
        return APIResponse(
            success=False,
            message="Profile skin not found",
            data=None
        )

    return APIResponse(
        success=True,
        message="Profile skin retrieved successfully",
        data={
            "id": query.id,
            "user_id": query.user_id,
            "skin_type": query.skin_type,
            "concerns": query.concerns.split(", ") if isinstance(query.concerns, str) else query.concerns,
            "created_at": query.created_at,
            "updated_at": query.updated_at
        }
    )


@router.post("/create-profile-skin", response_model=APIResponse)
async def create_profile_skin(skin_create: SkinCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create new skin profile"""
    existing_skin = db.query(Skin).filter(Skin.user_id == current_user.id).first()
    if existing_skin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skin profile already exists"
        )
    
    try:
        concerns_str = (
            skin_create.concerns
            if isinstance(skin_create.concerns, str)
            else ", ".join(skin_create.concerns)
        )
        skin = Skin(
            user_id=current_user.id,
            skin_type=skin_create.skin_type,
            concerns=concerns_str
        )
        db.add(skin)
        db.commit()
        db.refresh(skin)

        return APIResponse(
            success=True,
            message="Skin profile created successfully",
            data=SkinResponse(
                id=skin.id,
                user_id=skin.user_id,
                skin_type=skin.skin_type,
                concerns=skin.concerns.split(", ") if isinstance(skin.concerns, str) else skin.concerns,
                created_at=skin.created_at,
                updated_at=skin.updated_at
            )
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Creation failed: {str(e)}"
        )

@router.put("/update-skin-profile", response_model=APIResponse)
async def update_skin_profile(
    skin_update: SkinUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    skin = db.query(Skin).filter(Skin.user_id == current_user.id).first()
    if not skin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Skin profile not found"
        )
    
    try:
        skin.skin_type = skin_update.skin_type
        skin.concerns = skin_update.concerns if isinstance(skin_update.concerns, str) else ", ".join(skin_update.concerns)
        db.commit()
        db.refresh(skin)

        return APIResponse(
            success=True,
            message="Skin profile updated successfully",
            data=SkinResponse(
                id=skin.id,
                user_id=skin.user_id,
                skin_type=skin.skin_type,
                concerns=skin.concerns,
                created_at=skin.created_at,
                updated_at=skin.updated_at
            )
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )