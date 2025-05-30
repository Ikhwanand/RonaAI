from datetime import datetime 
from pydantic import BaseModel
from typing import Optional

class JournalBase(BaseModel):
    title: str 
    content: str 
    


class JournalCreate(JournalBase):
    pass 

class JournalResponse(JournalBase):
    id: int 
    user_id: int 
    created_at: datetime 
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        alias_generator = lambda field_name: "update_at" if field_name == "updated_at" else field_name 
        