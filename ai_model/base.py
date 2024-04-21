class BaseAIModel:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
