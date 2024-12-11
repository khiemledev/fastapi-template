import http
import time
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config import AppConfig
from schema.api import APIResponse
from util.logger import get_logger, setup_logger

dotenv.load_dotenv()
setup_logger()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up")
    logger.info("Start application in %s mode" % str(AppConfig().Mode))

    yield
    # Shutdown
    logger.info("Shutting down")


def setup_app() -> FastAPI:
    app = FastAPI()

    is_dev = AppConfig().Mode == "DEVELOPMENT"
    app = FastAPI(
        docs_url=AppConfig().SwaggerURL if is_dev else None,
        redoc_url=AppConfig().ReDocURL if is_dev else None,
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
            status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=APIResponse(
                status=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                message=http.HTTPStatus.INTERNAL_SERVER_ERROR.name,
                data=None,
            ).model_dump(),
        )

    @app.exception_handler(ValidationError)
    @app.exception_handler(RequestValidationError)
    async def pydantic_request_validation_error(
        request: Request,
        err: RequestValidationError | ValidationError,
    ):
        logger.exception(err)
        return JSONResponse(
            status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=APIResponse(
                status=http.HTTPStatus.BAD_REQUEST.value,
                message=str(http.HTTPStatus.BAD_REQUEST),
                data=err.errors(),
            ).model_dump(),
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

    return app
