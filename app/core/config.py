from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    DATABASE_URL: str = "sqlite:///./rona_ai.db"
    PROJECT_NAME: str = "RonaAI"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.urandom(32).hex()
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    ALLOWED_ORIGINS: list = ["http://localhost:5173"]

    class Config:
        case_sensitive = True 

    
settings = Settings()