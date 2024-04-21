import numpy as np
from PIL import Image

from ai_model.base import BaseAIModel
from schema.detection import DetectionResult


class BaseDetector(BaseAIModel):
    def __init__(self):
        self.model = None
        raise NotImplementedError

    def __call__(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[DetectionResult]:
        return self.predict(images)

    def predict(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[DetectionResult]:
        raise NotImplementedError
