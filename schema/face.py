import numpy as np
from PIL import Image

from schema.detection import DetectionResult


class FaceDetectionResult(DetectionResult):
    faces: list[Image.Image | np.ndarray]

    class Config:
        arbitrary_types_allowed = True
