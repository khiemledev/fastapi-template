import dotenv

from api.api import setup_app
from constant.app import Mode
from util.config import Config
from util.logger import get_logger

dotenv.load_dotenv()
logger = get_logger()

app = setup_app()

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
