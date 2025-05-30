from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey 
from sqlalchemy.orm import relationship
from app.db.database import Base 
from datetime import datetime 



class Products(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    product_name = Column(String, nullable=False)
    product_category = Column(String, nullable=False)
    ai_recommendation = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="products")