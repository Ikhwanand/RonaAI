from pydantic import BaseModel, Field
from typing import Union, List


class SkinBase(BaseModel):
    skin_type: str = Field(..., description="Skin type (Oily/Dry/Combination/Normal)")
    concerns: Union[str, List[str]] = Field(..., description="List of skin concerns")

class SkinCreate(SkinBase):
    pass 

class SkinUpdate(SkinBase):
    pass

class SkinResponse(BaseModel):
    id: int 
    user_id: int
    skin_type: str
    concerns: Union[str, List[str]]
    created_at: str 
    updated_at: str 

    