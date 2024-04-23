import http
import io
from base64 import b64encode
from io import BytesIO

import numpy as np
from fastapi import File, UploadFile
from PIL import Image

from schema.detection import DetectionAPIResponse, DetectionDataAPIResponse
from service.face_analysis import FaceAnalysisService
from util.logger import get_logger

logger = get_logger()


class FaceHandler:
    def __init__(self, *, face_analyzer_service: FaceAnalysisService):
        self.face_analyzer_service = face_analyzer_service

    async def face_detect_handler(
        self,
        return_b64_images: bool = False,
        max_face_width: int = 256,
        images: list[UploadFile] = File(...),
    ) -> DetectionAPIResponse:
        imgs: list[Image.Image] = []
        for image in images:
            img_bytes = await image.read()
            img = Image.open(BytesIO(img_bytes))
            imgs.append(img)

        result = self.face_analyzer_service.detect_face(imgs)

        resp: list[DetectionDataAPIResponse] = []
        for r in result:
            data = DetectionDataAPIResponse(
                count=r.count,
                boxes=r.boxes,
            )

            if return_b64_images:
                data.b64_images = []
                for face in r.faces:
                    if isinstance(face, np.ndarray):
                        # Convert to RGB Image
                        face = Image.fromarray(face)

                    if max_face_width > 0:
                        imw, imh = face.size
                        if imw > max_face_width:
                            new_w = max_face_width
                            new_h = int((new_w / imw) * imh)

                            face = face.resize((new_w, new_h))

                    buf = io.BytesIO()
                    face.save(buf, format="JPEG")
                    buf.seek(0)
                    data.b64_images.append(b64encode(buf.getvalue()).decode())
                    buf.flush()
                    buf.close()

            resp.append(data)

        return DetectionAPIResponse(
            status=http.HTTPStatus.OK.value,
            message="Face detection handler is running",
            data=resp,
        )
