from enum import Enum

from constant.app import Mode


def get_v(self) -> any:
    return getattr(self.name)


class Config(str, Enum):
    # This is the global config
    Mode = Mode.DEVELOPMENT.value

    SwaggerURL = "/docs"
    ReDocURL = "/redoc"

    Host = "0.0.0.0"
    Port = 8080
    Workers = 1


class LoggerConfig(str, Enum):
    # This is the logger config
    LogDir = "./log/"
    BackTrace = False
    MaxBytes = 10485760  # 10MB ~ 10485760
    MaxBackupCount = 10
    SerializeJSON = True
    Diagnose = False  # This should be disabled in production
