import numpy as np
from PIL import Image

from ai_model.detector.base_detector import BaseDetector
from ai_model.extractor.base_extractor import BaseExtractor
from schema.face import FaceDetectionResult
from util.config.ai_model import YOLOFaceConfig


class FaceAnalysisService:
    def __init__(
        self,
        face_detector: BaseDetector,
        face_recognizer: BaseExtractor,
        expend_percentage=0,
        crop_faces: bool = YOLOFaceConfig.CropFaces.value,
    ) -> None:
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.expand_percentage = expend_percentage
        self.crop_faces = crop_faces

    def detect_face(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[FaceDetectionResult]:
        """Detect face within images

        Args:
            images (list[Image.Image | np.ndarray] | np.ndarray): list of images

        Returns:
            list[FaceDetectionResult]: result of face detection
        """  # noqa: E501

        result: list[FaceDetectionResult] = []

        predictions = self.face_detector(images)
        for j in range(len(predictions)):
            pred = predictions[j]
            boxes = pred.boxes

            face_det = FaceDetectionResult(
                **pred.model_dump(),
                faces=[],
            )

            for i in range(len(boxes)):
                box = boxes[i]

                if self.expand_percentage > 0:
                    x1, y1, x2, y2, _, __, imw, imh = box.to_list()

                    w = x2 - x1
                    h = y2 - y1
                    # Expand the facial region height and width by the provided percentage # noqa
                    # ensuring that the expanded region stays within img.shape limits # noqa
                    expanded_w = w + int(w * self.expand_percentage / 100)
                    expanded_h = h + int(h * self.expand_percentage / 100)
                    w_added = int((expanded_w - w) / 2)
                    h_added = int((expanded_h - h) / 2)

                    x1 = max(0, x1 - w_added)
                    y1 = max(0, y1 - h_added)
                    x2 = min(imw - 1, x2 + w_added)
                    y2 = min(imh - 1, y2 + h_added)

                    box.x_min = x1
                    box.y_min = y1
                    box.x_max = x2
                    box.y_max = y2

                    boxes[i] = box

                if self.crop_faces:
                    x1, y1, x2, y2, _, __, imw, imh = box.to_list()
                    img = pred.orin_image
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img[:, :, ::-1])
                    face = img.crop((x1, y1, x2, y2))
                    face_det.faces.append(face)

            face_det.boxes = boxes

            result.append(face_det)

        return result
