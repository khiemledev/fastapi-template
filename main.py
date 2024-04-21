import dotenv

from ai_model.detector.face.yolo_face import YOLOFace
from api.api import setup_app
from constant.app import Mode
from handler.face import FaceHandler
from routes.router import setup_router
from service.face_analysis import FaceAnalysisService
from util.config.ai_model import YOLOFaceConfig
from util.config.config import Config
from util.logger import get_logger

dotenv.load_dotenv()
logger = get_logger()

app = setup_app()

# Models
yolo_face = YOLOFace()

# Services
face_analyzer_service = FaceAnalysisService(
    face_detector=yolo_face,
    face_recognizer=yolo_face,
    expend_percentage=int(YOLOFaceConfig.ExpandPercentage),
)

# Handlers
face_handler = FaceHandler(
    face_analyzer_service=face_analyzer_service,
)

# Routes
router = setup_router(face_handler)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=Config.Host.value,
        port=Config.Port.value,
        workers=Config.Workers.value
        if Config.Mode.value == Mode.PRODUCTION
        else 1,
    )
