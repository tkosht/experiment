from codeboxapi.box.localbox import LocalBox


class CustomLocalBox(LocalBox):
    def __new__(cls, *args, **kwargs):
        # NOTE: sigleton
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, port: int = 8888) -> None:
        super().__init__()
        self.port = port
