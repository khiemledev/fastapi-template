import numpy as np
from PIL import Image

from ai_model.detector.base_detector import BaseDetector
from ai_model.extractor.base_extractor import BaseExtractor
from constant.face_analysis import DistanceMetric
from schema.detection import BBox
from schema.face import FaceDetectionResult


class FaceAnalysisService:
    def __init__(
        self,
        face_detector: BaseDetector,
        face_recognizer: BaseExtractor,
        expend_percentage=0,
    ) -> None:
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.expand_percentage = expend_percentage

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
        for pred in predictions:
            cls_ids = pred.boxes.cls.cpu().detach().numpy().tolist()
            confs = pred.boxes.conf.cpu().detach().numpy().tolist()
            boxes = pred.boxes.xyxy.cpu().detach().numpy().tolist()

            imh, imw = pred.orig_shape

            _boxes: list[BBox] = []
            _faces: list[Image.Image] = []
            for cls_id, conf, box in zip(cls_ids, confs, boxes):
                x1, y1, x2, y2 = box

                if self.expand_percentage > 0:
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

                    box = [x1, y1, x2, y2]

                _boxes.append(
                    BBox(
                        x_min=x1,
                        y_min=y1,
                        x_max=x2,
                        y_max=y2,
                        cls_id=int(cls_id),
                        score=conf,
                        im_width=imw,
                        im_height=imh,
                    ),
                )

                # Crop face region
                face_region = Image.fromarray(pred.orig_img[:, :, ::-1]).crop(
                    box,
                )
                _faces.append(face_region)

                face_region.save("face.jpg")

            result.append(
                FaceDetectionResult(
                    faces=_faces,
                    boxes=_boxes,
                    count=len(_boxes),
                ),
            )

        return result

    def get_embedding(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        """Extract embedding vectors from face images

        Args:
            images (list[Image.Image | np.ndarray] | np.ndarray): list of images

        Returns:
            np.ndarray: embedding vector of the image
        """  # noqa: E501
        raise NotImplementedError

    def compare_embedding(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> float:
        """Compare two embedding vectors

        Args:
            embedding1 (np.ndarray): embedding vector 1
            embedding2 (np.ndarray): embedding vector 2
            distance_metric (DistanceMetric, optional): distance metric. Defaults to DistanceMetric.COSINE.

        Returns:
            float: distance between two embedding vectors
        """  # noqa: E501
        raise NotImplementedError

    def compare_face(
        self,
        image1: Image.Image,
        image2: Image.Image,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        threshold: float = 0.5,
    ) -> bool:
        """Compare two face images

        Args:
            image1 (Image.Image): image 1
            image2 (Image.Image): image 2
            distance_metric (DistanceMetric, optional): distance metric. Defaults to DistanceMetric.COSINE.
            threshold (float, optional): threshold. Defaults to 0.5.

        Returns:
            bool: result of comparison
        """  # noqa: E501
        raise NotImplementedError
