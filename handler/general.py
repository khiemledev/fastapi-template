from http import HTTPStatus

from schema.api import APIResponse


class GeneralHandler:
    def __init__(self) -> None:
        pass

    async def ping_handler(self) -> APIResponse:
        return APIResponse(
            status=HTTPStatus.OK.value,
            message="Running (Healthy)",
            data="ping",
        )
