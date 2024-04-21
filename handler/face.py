import http
from io import BytesIO

from fastapi import File, UploadFile
from PIL import Image

from schema.api import APIResponse
from service.face_analysis import FaceAnalysisService
from util.logger import get_logger

logger = get_logger()


class FaceHandler:
    def __init__(self, *, face_analyzer_service: FaceAnalysisService):
        self.face_analyzer_service = face_analyzer_service

    async def face_detect_handler(
        self,
        images: list[UploadFile] = File(...),
    ) -> APIResponse:
        imgs: list[Image.Image] = []
        for image in images:
            img_bytes = await image.read()
            img = Image.open(BytesIO(img_bytes))
            imgs.append(img)

        result = self.face_analyzer_service.detect_face(imgs)

        # Remove faces from result because it cannot be json serialized
        for r in result:
            r.faces = []

        return APIResponse(
            status=http.HTTPStatus.OK.value,
            message="Face detection handler is running",
            data=result,
        )

    async def face_recognize_handler(self) -> APIResponse:
        return APIResponse(
            status=http.HTTPStatus.OK.value,
            message="Face recognition handler is running",
        )
