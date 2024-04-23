import http
import io
from base64 import b64encode
from io import BytesIO
from typing import Optional

import numpy as np
import shortuuid
from fastapi import File, UploadFile
from PIL import Image

from schema.api import APIResponse
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
        trace_id: Optional[str] = None,
    ) -> DetectionAPIResponse:
        if trace_id is None:
            trace_id = shortuuid.uuid()

        log_ctx = logger.bind(trace_id=trace_id)
        log_ctx.info(
            "request received with %s images; return_b64_images: %s; max_face_width: %s"
            % (len(images), return_b64_images, max_face_width),
        )

        try:
            imgs: list[Image.Image] = []
            for image in images:
                img_bytes = await image.read()
                img = Image.open(BytesIO(img_bytes))
                imgs.append(img)
        except Exception as err:
            log_ctx.error(
                "error while parsing image from uploaded files",
            )
            log_ctx.exception(err)
            return APIResponse(
                status=http.HTTPStatus.BAD_REQUEST.value,
                message=http.HTTPStatus.BAD_REQUEST.name,
            )

        log_ctx.info("start detecting faces...")
        result = self.face_analyzer_service.detect_face(imgs)
        log_ctx.info("finished detecting faces")

        log_ctx.info("post-processing for result...")
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
        log_ctx.info(
            "finished post-processing for result",
        )

        return DetectionAPIResponse(
            status=http.HTTPStatus.OK.value,
            message="Face detection handler is running",
            data=resp,
        )
