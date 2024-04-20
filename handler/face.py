import http

from fastapi import APIRouter

from schemas.api import APIResponse

faceRouter = APIRouter()


class FaceHandler:
    def __init__(self):
        pass

    @faceRouter.get("/detect")
    def face_detection_handler() -> APIResponse:
        return APIResponse(
            status=http.HTTPStatus.OK,
            message="Face detection handler is running",
        )

    @faceRouter.get("/recognize")
    def face_recognition_handler() -> APIResponse:
        return APIResponse(
            status=http.HTTPStatus.OK,
            message="Face recognition handler is running",
        )
