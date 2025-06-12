from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.analysis import Analysis
from app.models.skin import Skin
from app.services.agent import analyze_skin
from app.services.load_model import visualize_prediction
from app.core.security import get_current_user
from app.models.users import User
from app.schemas.responses import APIResponse
from app.models.journals import Journals
import aiofiles
import os
from datetime import datetime
import json

router = APIRouter()

# @router.post("/upload-image", response_model=APIResponse)
# async def upload_image(image: UploadFile = File(...), current_user: User = Depends(get_current_user)):
#     """Upload skin image for analysis"""
#     # Validate image format
#     if not image.content_type.startswith("image/"):
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image")
    
#     # Save image
#     upload_dir = "uploads/skin-images"
#     os.makedirs(upload_dir, exist_ok=True)

#     # Generate unique filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_extension = os.path.splitext(image.filename)[1]
#     filename = f"{current_user.id}_{timestamp}.{file_extension}"

#     filepath = os.path.join(upload_dir, filename)
#     async with aiofiles.open(filepath, 'wb') as out_file:
#         content = await image.read()
#         await out_file.write(content)
    
#     # TODO: AI Integration Point 1
#     # Here we would:
#     # 1. Send image to AI model for initial processing
#     # 2. Get preliminary results
#     # 3. Store initial analysis data
    
#     analysis_id = f"analysis_{timestamp}"

#     return APIResponse(
#         success=True,
#         message="Image uploaded successfully",
#         data={"analysis_id": analysis_id}
#     )

@router.post("/analyze", response_model=APIResponse)
async def analyze_skin_image(request: Request, image: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Upload and analyze skin image"""
    # Validate image format
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Save image 
    
    upload_dir = "uploads/skin-images"
    classify_dir = "uploads/classify"
    os.makedirs(upload_dir.split('/')[0], exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(classify_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(image.filename)[1]
    filename = f"{current_user.id}_{timestamp}{file_extension}"
    filepath = os.path.join(upload_dir, filename)

    # Save the uplaoded file
    with open(filepath, "wb") as buffer:
        buffer.write(await image.read())

    filename_classify = filename.replace(".", "_classify.")
    filepath_classify = os.path.join(classify_dir, filename_classify)
    
    # with open(filepath_classify, "wb") as buffer:
    #     buffer.write(await visualize_prediction(image_path=filepath).read())
    
    # Perform AI analysis
    try:
        # Called visualize_prediction and catch the result
        saved_classify_path = visualize_prediction(filepath, filepath_classify)
        
        if not saved_classify_path or not os.path.exists(saved_classify_path):
            print("Warning: Visualization failed, using original image")
            import shutil
            shutil.copy(filepath, filepath_classify)
            saved_classify_path = filepath_classify
        
        query = db.query(Journals).filter(Journals.user_id == current_user.id)

        journals = query.order_by(Journals.created_at.desc()).all()
        # Convert SQLAlchemy model to dictionary
        journals_list = []
        for journal in journals:
            journals_list.append({
                "title": journal.title,
                "content": journal.content,
                "created_at": journal.created_at,
                "updated_at": journal.updated_at
            })

        
        user_country = current_user.country
        analysis_result = analyze_skin(image_url=filepath_classify, country=user_country, journals=journals_list)

        # Validate and convert AI response
        if isinstance(analysis_result, str):
            analysis_result = json.loads(analysis_result)

        # Validate required fields
        required_fields = ["overall_health", "skin_type", "concerns", "recommendations", "analysis_metrics", "skincare_products"]
        if not all(field in analysis_result for field in required_fields):
            raise ValueError("Invalid AI response structure")
        
        # Create analysis record
        analysis = Analysis(
            user_id=current_user.id,
            image_url=str(request.base_url)[:-1] + f"/uploads/skin-images/{filename}",
            overall_health=analysis_result["overall_health"],
            skin_type=analysis_result["skin_type"],
            concerns=analysis_result["concerns"],
            recommendations=analysis_result["recommendations"],
            analysis_metrics=analysis_result["analysis_metrics"],
            skincare_products=analysis_result["skincare_products"]
        )
        db.add(analysis)

        # Update or create skin profile
        skin = db.query(Skin).filter(Skin.user_id == current_user.id).first()
        concerns_list = [concern["name"] for concern in analysis_result["concerns"]]

        if not skin:
            skin = Skin(
                user_id=current_user.id,
                skin_type=analysis_result["skin_type"],
                concerns=", ".join(concerns_list)
            )
            db.add(skin)
        else:
            skin.skin_type = analysis_result["skin_type"]
            skin.concerns = ", ".join(concerns_list)
        
        db.commit()
        db.refresh(analysis)

        # # Convert SQLAlchemy model to Pydantic model for serialization
        # analysis_response = AnalysisResponse(
        #     id=analysis.id,
        #     user_id=analysis.user_id,
        #     image_url=analysis.image_url,
        #     created_at=analysis.created_at,
        #     overall_health=analysis.overall_health,
        #     skin_type=analysis.skin_type,
        #     concerns=analysis.concerns,
        #     recommendations=analysis.recommendations,
        #     analysis_metrics=analysis.analysis_metrics
        # )

        return APIResponse(
            success=True,
            message="Image analyzed successfully",
            data={
                "analysis": {
                    "id": analysis.id,
                    "image_url": analysis.image_url,
                    "overall_health": analysis.overall_health,
                    "skin_type": analysis.skin_type,
                    "concerns": analysis.concerns,
                    "recommendations": analysis.recommendations,
                    "analysis_metrics": analysis.analysis_metrics,
                    "skincare_products": analysis.skincare_products,
                    "created_at": analysis.created_at.isoformat()
                },
                "skin_profile": {
                    "skin_type": analysis_result["skin_type"],
                    "concerns": concerns_list
                }
            }
        )
    except Exception as e:
        # Clean up the uploaded file if analysis fails
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(filepath_classify):
            os.remove(filepath_classify)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

    
    # # Mock response for development
    # analysis_data = {
    #     "overall_health": "Good",
    #     "skin_type": "Combination",
    #     "concerns": [
    #         {
    #             "name": "Acne Scars",
    #             "severity": "Moderate",
    #             "type": "Box Scars",
    #             "confidence": 0.89
    #         },
    #         {
    #             "name": "Dryness",
    #             "severity": "Mild",
    #             "type": None,
    #             "confidence": 0.75
    #         }
    #     ],
    #     "recommendations": [
    #         {
    #             "title": "Use a Gentle Cleanser",
    #             "description": "Use a pH-balanced cleanser twice daily.",
    #             "priority": "High"
    #         },
    #         {
    #             "title": "Apply Vitamin C Serum",
    #             "description": "Help reduce the appearance of acne scars.",
    #             "priority": "Medium"
    #         }
    #     ],
    #     "analysis_metrics": {
    #         "skin_hydration": 65,
    #         "texture_uniformity": 78,
    #         "pore_visibility": 45,
    #         "overall_score": 72
    #     }
    # }
    


@router.get("/history", response_model=APIResponse)
async def get_analysis_history(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 10
):
    """Get users' analysis history with pagination"""
    total = db.query(Analysis).filter(Analysis.user_id == current_user.id).count()
    analyses = db.query(Analysis)\
        .filter(Analysis.user_id == current_user.id)\
        .order_by(Analysis.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    # Convert SQLAlchemy models to dictionaries
    analysis_list = []
    for analysis in analyses:
        analysis_list.append({
            "id": analysis.id,
            "user_id": analysis.user_id,
            "image_url": analysis.image_url,
            "overall_health": analysis.overall_health,
            "skin_type": analysis.skin_type,
            "concerns": analysis.concerns,
            "recommendations": analysis.recommendations,
            "analysis_metrics": analysis.analysis_metrics,
            "skincare_products": analysis.skincare_products,
            "created_at": analysis.created_at.isoformat()
        })
    
    return APIResponse(
        success=True,
        message="Analysis history retrieved successfully",
        data={
            "items": analysis_list,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    )

@router.get("/get-analysis/{analysis_id}", response_model=APIResponse)
async def get_analysis(analysis_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get specific analysis results"""
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id, 
        Analysis.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Analysis not found"
        )
    
    # Convert SQLAlchemy model to dictionary
    analysis_data = {
        "id": analysis.id,
        "user_id": analysis.user_id,
        "image_url": analysis.image_url,
        "overall_health": analysis.overall_health,
        "skin_type": analysis.skin_type,
        "concerns": analysis.concerns,
        "recommendations": analysis.recommendations,
        "analysis_metrics": analysis.analysis_metrics,
        "skincare_products": analysis.skincare_products,
        "created_at": analysis.created_at.isoformat()
    }
    
    return APIResponse(
        success=True,
        message="Analysis retrieved successfully",
        data=analysis_data
    )

@router.delete("/delete-analysis/{analysis_id}", response_model=APIResponse)
async def delete_analysis(analysis_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete an analysis record"""
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id, 
        Analysis.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Analysis not found"
        )
    
    # Extract image path from URL and delete the files
    if analysis.image_url:
        # Get the relative path from the URL
        image_path = analysis.image_url.split("/uploads/", 1)[-1]
        if image_path:
            # Construct the full path for both original and classified images
            original_image = os.path.join("uploads", image_path)
            classified_image = os.path.join("uploads", "classify", os.path.basename(image_path).replace(".", "_classify."))
            
            print(original_image)
            print(classified_image)
            # Delete original image if exists
            if os.path.exists(original_image):
                os.remove(original_image)
            
            # Delete classified image if exists
            if os.path.exists(classified_image):
                os.remove(classified_image)

    db.delete(analysis)
    db.commit()

    return APIResponse(
        success=True,
        message="Analysis deleted successfully"
    )