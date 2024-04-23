from enum import Enum


class YOLOFaceConfig(str, Enum):
    WeightsPath = "weights/yolov8n-face.pt"
    ExpandPercentage = 5
    CropFaces = True
