from fastapi import HTTPException, Request 
from fastapi.responses import JSONResponse
from typing import Union


class AppException(HTTPException):
    def __init__(self, status_code: int, message: Union[str, list], internal_code: str = None):
        super().__init__(status_code=status_code, detail=message)
        self.internal_code = internal_code 


async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "internal_code": exc.internal_code,
            "data": None
        }
    )