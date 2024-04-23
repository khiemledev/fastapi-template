from fastapi import APIRouter

from handler.face import FaceHandler
from util.logger import get_logger

logger = get_logger()


def setup_router(face_handler: FaceHandler) -> APIRouter:
    logger.info("setting up routers...")

    router = APIRouter()

    logger.info("setting up face router...")
    face_router = APIRouter()
    face_router.add_api_route(
        "/detect",
        endpoint=face_handler.face_detect_handler,
        methods=["POST"],
    )
    logger.info("face router setup successfully")

    logger.info("adding routers...")
    router.include_router(face_router, prefix="/face", tags=["Face"])
    logger.info("routers setup successfully")

    return router
