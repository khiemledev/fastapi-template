
class AppConfig:
    def __init__(self) -> None:
        # This is the global config
        self.Mode = "DEVELOPMENT"

        self.SwaggerURL = "/docs"
        self.ReDocURL = "/redoc"

        self.Host = "0.0.0.0"
        self.Port = 8080
        self.Workers = 1
