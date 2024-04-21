import http
import time
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from constant.app import Mode
from routes.router import setup_router
from schemas.api import APIResponse
from util.config import Config
from util.logger import get_logger, setup_logger

dotenv.load_dotenv()
setup_logger()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up")
    logger.info("Start application in %s mode" % str(Config.Mode))

    yield
    # Shutdown
    logger.info("Shutting down")


print(str(Config.SwaggerURL))


def setup_app() -> FastAPI:
    app = FastAPI()

    app = FastAPI(
        docs_url=Config.SwaggerURL.value
        if Config.Mode == Mode.DEVELOPMENT
        else None,
        redoc_url=Config.ReDocURL.value
        if Config.Mode == Mode.DEVELOPMENT
        else None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    @app.exception_handler(status.HTTP_500_INTERNAL_SERVER_ERROR)
    async def internal_exception_handler(request: Request, exc: Exception):
        # Handle 500 exception
        logger.exception(exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse(
                status=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )

    @app.exception_handler(ValidationError)
    @app.exception_handler(RequestValidationError)
    async def pydantic_request_validation_error(
        request: Request,
        err: RequestValidationError | ValidationError,
    ):
        logger.exception(err)
        return APIResponse(
            status=http.HTTPStatus.BAD_REQUEST.value,
            message=str(http.HTTPStatus.BAD_REQUEST),
            data=err.errors(),
        )

    @app.get("/")
    async def root():
        return APIResponse(
            status=http.HTTPStatus.OK.value,
            message="Running (Healthy)",
        )

    @app.get("/health-check")
    async def health_check():
        return APIResponse(
            status=http.HTTPStatus.OK.value,
            message="Running (Healthy)",
        )

    router = setup_router()
    app.include_router(router)

    return app
