import numpy as np
from PIL import Image
from ultralytics import YOLO

from ai_model.detector.base_detector import BaseDetector
from schema.detection import BBox, DetectionResult
from util.config.ai_model import YOLOFaceConfig
from util.logger import get_logger

logger = get_logger()


class YOLOFace(BaseDetector):
    def __init__(self):
        self.log_ctx = logger.bind(name="yolov8_face")

        self.log_ctx.info(
            "load model YOLOv8-Face with weights %s..."
            % (YOLOFaceConfig.WeightsPath.value),
        )
        self.model = YOLO(str(YOLOFaceConfig.WeightsPath.value))
        self.log_ctx.info("finished loading model")

    def __call__(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[DetectionResult]:
        return self.predict(images)

    def predict(
        self,
        images: list[Image.Image | np.ndarray] | np.ndarray,
    ) -> list[DetectionResult]:
        result: list[DetectionResult] = []

        self.log_ctx.info(
            "performing prediction for %d images..." % len(images),
        )
        predictions = self.model(images)
        self.log_ctx.info("finished prediction")

        # convert predictions
        self.log_ctx.info("converting predictions...")
        for pred in predictions:
            cls_ids = pred.boxes.cls.cpu().detach().numpy().tolist()
            confs = pred.boxes.conf.cpu().detach().numpy().tolist()
            boxes = pred.boxes.xyxy.cpu().detach().numpy().tolist()

            imh, imw = pred.orig_shape

            _boxes: list[BBox] = []
            for cls_id, conf, box in zip(cls_ids, confs, boxes):
                x1, y1, x2, y2 = box

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

            result.append(
                DetectionResult(
                    boxes=_boxes,
                    count=len(_boxes),
                    orin_image=pred.orig_img,
                ),
            )
        self.log_ctx.info("finished converting predictions")

        return result
