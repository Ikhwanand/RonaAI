from .auth import router as auth_router 
from .analysis import router as analysis_router
from .profile import router as profile_router 
from .progress import router as progress_router 
from .journal import router as journal_router 
from .products import router as products_router
from .skin import router as skin_router

__all__ = [
    "auth_router",
    "analysis_router",
    "profile_router",
    "progress_router",
    "journal_router",
    "products_router",
    'skin_router',
]