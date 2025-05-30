from fastapi import APIRouter 
from app.api.endpoints import auth_router, profile_router, progress_router, journal_router, products_router, skin_router

router = APIRouter()

router.include_router(auth_router, prefix='/auth', tags=['Auth'])
router.include_router(progress_router, prefix='/progress', tags=['Progress'])
router.include_router(profile_router, prefix='/profile', tags=['Profile'])
router.include_router(journal_router, prefix='/journals', tags=['Journals'])
router.include_router(products_router, prefix='/products', tags=['Products'])
router.include_router(skin_router, prefix='/skin', tags=['Skin'])