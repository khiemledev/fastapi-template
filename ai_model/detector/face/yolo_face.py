import numpy as np
from PIL import Image
from ultralytics import YOLO

from ai_model.detector.base_detector import BaseDetector
from util.config.ai_model import YOLOFaceConfig
from util.logger import get_logger

logger = get_logger()


class YOLOFace(BaseDetector):
    def __init__(self):
        logger.info("Loading YOLO Face model")
        self.model = YOLO(str(YOLOFaceConfig.WeightsPath.value))
        logger.info("Loaded YOLO Face model")

    def __call__(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[dict]:
        return self.predict(images)

    def predict(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[dict]:
        predictions = self.model(images)
        return predictions
