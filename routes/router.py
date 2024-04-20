from fastapi import APIRouter

from handler.face import faceRouter


def setup_router() -> APIRouter:
    router = APIRouter()

    router.include_router(faceRouter, prefix="/face", tags=["Face"])

    return router
