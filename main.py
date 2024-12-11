import dotenv

from api.api import setup_app
from config import AppConfig
from handler.general import GeneralHandler
from route.router import setup_router
from util.logger import get_logger

dotenv.load_dotenv()
logger = get_logger()

app = setup_app()

# Handlers
general_handler = GeneralHandler()

# Routes
router = setup_router(handler=general_handler)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=AppConfig().Host,
        port=AppConfig().Port,
        workers=AppConfig().Workers
        if AppConfig().Mode == "PRODUCTION" else 1,
    )
