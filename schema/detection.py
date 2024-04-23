from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel

from schema.api import APIResponse


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    cls_id: int
    score: float
    im_width: int
    im_height: int

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "cls_id": self.cls_id,
            "score": self.score,
            "im_width": self.im_width,
            "im_height": self.im_height,
        }

    def to_list(self) -> list:
        return [
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
            self.cls_id,
            self.score,
            self.im_width,
            self.im_height,
        ]


class DetectionResult(BaseModel):
    count: int
    boxes: list[BBox]
    orin_image: Optional[Image.Image | np.ndarray | None] = None

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "boxes": [box.to_dict() for box in self.boxes],
            "orin_image": self.orin_image,
        }

    def to_list(self) -> list:
        """Only the bounding boxes return"""

        return [box.to_list() for box in self.boxes]


class DetectionDataAPIResponse(BaseModel):
    count: int
    boxes: list[BBox]
    b64_images: Optional[list[str]] = None


class DetectionAPIResponse(APIResponse):
    data: list[DetectionDataAPIResponse]
