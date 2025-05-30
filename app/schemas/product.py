from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class ProductBase(BaseModel):
    product_name: str 
    product_category: str
    ai_recommendation: bool 


class ProductCreate(ProductBase):
    pass 


class ProductResponse(ProductBase):
    id: int
    user_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None 

    class Config:
        from_attributes = True
        alias_generator = lambda field_name: "update_at" if field_name == "updated_at" else field_name