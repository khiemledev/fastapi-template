import numpy as np
from PIL import Image

from ai_model.base import BaseAIModel


class BaseExtractor(BaseAIModel):
    def __init__(self):
        self.model = None
        raise NotImplementedError

    def __call__(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[np.ndarray] | np.ndarray:
        self.predict(images)

    def predict(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[np.ndarray] | np.ndarray:
        raise NotImplementedError
