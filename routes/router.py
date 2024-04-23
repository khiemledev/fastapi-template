from fastapi import APIRouter

from handler.face import FaceHandler


def endpoint():
    return {}


def setup_router(face_handler: FaceHandler) -> APIRouter:
    router = APIRouter()

    face_router = APIRouter()
    face_router.add_api_route(
        "/detect",
        endpoint=face_handler.face_detect_handler,
        methods=["POST"],
    )

    router.include_router(face_router, prefix="/face", tags=["Face"])

    return router
