from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.database import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_url = Column(String, nullable=True)
    overall_health = Column(String, nullable=True)
    skin_type = Column(String, nullable=True)
    concerns = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    analysis_metrics = Column(JSON, nullable=True)
    skincare_products = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relasi
    user = relationship("User", back_populates="analyses")