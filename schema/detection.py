from pydantic import BaseModel


class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    cls_id: int
    score: float
    im_width: int
    im_height: int


class DetectionResult(BaseModel):
    count: int
    boxes: list[BBox]
