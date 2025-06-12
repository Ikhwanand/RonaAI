from fastapi import APIRouter, Depends, HTTPException, status, Request 
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.users import User 
from app.core.security import get_current_user, pwd_context
from pydantic import BaseModel, EmailStr, validator
from app.schemas.responses import APIResponse
from typing import Optional
import re
from fastapi import UploadFile, File 
import os 
from datetime import datetime 

class UserProfileUpdate(BaseModel):
    name: Optional[str]
    email: Optional[EmailStr]
    current_password: Optional[str]
    new_password: Optional[str]
    
    @validator('new_password')
    def password_complexity(cls, v):
        if v is not None:
            if len(v) < 8:
                raise ValueError("Password must be at least 8 characters long")
            if not re.search(r"[A-Z]", v):
                raise ValueError("Password must contain at least one uppercase letter")
            if not re.search(r"[a-z]", v):
                raise ValueError("Password must contain at least one lowercase letter")
            if not re.search(r"\d", v):
                raise ValueError("Password must contain at least one digit")
        return v

router = APIRouter()

@router.put("/update", response_model=APIResponse)
async def update_profile(
    user_update: UserProfileUpdate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Update user profile including password"""
    # Validate current password if trying to change password or email
    if (user_update.new_password or user_update.email) and not user_update.current_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is required to change password or email"
        )

    # Verify current password if provided
    if user_update.current_password:
        if not pwd_context.verify(user_update.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

    # Update email if provided
    if user_update.email and user_update.email != current_user.email:
        existing_user = db.query(User).filter(User.email == user_update.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        current_user.email = user_update.email

    # Update name if provided
    if user_update.name:
        current_user.name = user_update.name

    # Update password if provided
    if user_update.new_password:
        current_user.hashed_password = pwd_context.hash(user_update.new_password)

    try:
        db.commit()
        db.refresh(current_user)

        return APIResponse(
            success=True,
            message="Profile updated successfully",
            data={
                "id": current_user.id,
                "name": current_user.name,
                "email": current_user.email,
                "profile_updated": True
            }
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the profile"
        )

@router.get("/me", response_model=APIResponse)
async def get_profile(request: Request, current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return APIResponse(
        success=True,
        message="Profile retrieved successfully",
        data={
            "id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
            "profile_image": f"{str(request.base_url)[:-1]}{current_user.profile_image}" if current_user.profile_image else None
        }
    )



@router.delete("/delete-account", response_model=APIResponse)
async def delete_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete user account permanently"""
    try:
        # Delete profile image if exists
        if current_user.profile_image:
            image_path = current_user.profile_image.lstrip('/')
            # Remove loading slash
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Delete the user (cascading delete will handle related records)
        db.delete(current_user)
        db.commit()

        return APIResponse(
            success=True,
            message="Account and all associated data deleted successfully",
            data=None
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the account"
        )
    

@router.post("/upload-profile-image", response_model=APIResponse)
async def upload_profile_image(request: Request, file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Upload profile image for user"""
    # Validate image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Create upload directory if not exists
    upload_dir = "uploads/profile-images"
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{current_user.id}_{timestamp}{file_extension}"
    filepath = os.path.join(upload_dir, filename)

    # Save the file
    try:
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Update user profile image
    file_path_url = f"/{filepath.replace(os.sep, '/')}"  # Convert path to URL format
    current_user.profile_image = file_path_url
    db.commit()

   
    base_url_image = f"{str(request.base_url)[:-1]}{file_path_url}"
    return APIResponse(
        success=True,
        message="Profile image uploaded successfully",
        data={"profile_image_url": base_url_image}
    )

