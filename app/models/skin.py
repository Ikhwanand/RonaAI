from sqlalchemy import Column, String, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.database import Base 
from datetime import datetime 


class Skin(Base):
    __tablename__ = "skin"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    skin_type = Column(String, nullable=False, default="")
    concerns = Column(Text, nullable=False, default="")
    created_at = Column(String, nullable=False, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_at = Column(String, nullable=False, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), onupdate=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    user = relationship("User", back_populates="skin")