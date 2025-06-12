from fastapi import APIRouter, Depends, HTTPException, status 
from sqlalchemy.orm import Session 
from typing import List 

from app.db.database import get_db
from app.models.products import Products
from app.models.users import User 
from app.schemas.product import ProductCreate, ProductResponse
from app.schemas.responses import APIResponse
from app.core.security import get_current_user


router = APIRouter()

@router.post("/create-product", response_model=APIResponse)
async def create_product(product_data: ProductCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    product = Products(
        user_id=current_user.id,
        product_name=product_data.product_name,
        product_category=product_data.product_category,
        ai_recommendation=product_data.ai_recommendation
    )

    db.add(product)
    db.commit()
    db.refresh(product)

    # Convert SQLAlchemy model to dictionary
    product_dict = {
        "id": product.id,
        "user_id": product.user_id,
        "product_name": product.product_name,
        "product_category": product.product_category,
        "ai_recommendation": product.ai_recommendation,
        "created_at": product.created_at,
        "updated_at": product.updated_at
    }

    return APIResponse(
        success=True,
        message="Product created successfully",
        data=product_dict
    )
    


@router.get("/get-products", response_model=APIResponse)
async def get_products(current_user: User = Depends(get_current_user), db: Session = Depends(get_db), skip: int = 0, limit: int = 10):
    total = db.query(Products).filter(Products.user_id == current_user.id).count()
    products = db.query(Products).filter(Products.user_id == current_user.id).order_by(Products.created_at.desc()).offset(skip).limit(limit).all()

    # Convert SQLAlchemy models to dictionaries
    products_list = []
    for product in products:
        products_list.append({
            "id": product.id,
            "user_id": product.user_id,
            "product_name": product.product_name,
            "product_category": product.product_category,
            "ai_recommendation": product.ai_recommendation,
            "created_at": product.created_at,
            "updated_at": product.updated_at
        })

    return APIResponse(
        success=True,
        message="Products retrieved successfully",
        data={
            "items": products_list,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    )


# @router.get("/{product_id}", response_model=APIResponse)
# async def get_product(product_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
#     product = db.query(Products).filter(Products.id == product_id, Products.user_id == current_user.id).first()

#     if not product:
#         return APIResponse(
#             success=False,
#             message="Product not found",
#             data=None
#         )
    
#     return APIResponse(
#         success=True,
#         message="Product retrieved successfully",
#         data=ProductResponse.from_orm(product)
#     )



@router.put("/update-product/{product_id}", response_model=APIResponse)
async def update_product(product_id: int, product_data: ProductCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    product = db.query(Products).filter(
        Products.id == product_id,
        Products.user_id == current_user.id
    ).first()

    if not product:
        return APIResponse(
            status=False, 
            message="Product not found",
            data=None
        )
    
    product.product_name = product_data.product_name 
    product.product_category = product_data.product_category
    product.ai_recommendation = product_data.ai_recommendation

    db.commit()
    db.refresh(product)

    # Convert SQLAlchemy model to dictionary
    product_dict = {
        "id": product.id,
        "user_id": product.user_id,
        "product_name": product.product_name,
        "product_category": product.product_category,
        "ai_recommendation": product.ai_recommendation,
        "created_at": product.created_at,
        "updated_at": product.updated_at
    }

    return APIResponse(
        success=True,
        message="Product updated successfully",
        data=product_dict
    )


@router.delete("/delete-product/{product_id}", response_model=APIResponse)
async def delete_product(product_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    product = db.query(Products).filter(
        Products.id == product_id,
        Products.user_id == current_user.id
    ).first()

    if not product:
        return APIResponse(
            success=False,
            message="Product not found",
            data=None
        )

    db.delete(product)
    db.commit()

    return APIResponse(
        success=True,
        message="Product deleted successfully"
    )