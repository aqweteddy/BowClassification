class BasePipeline:
    def __init__(self, args) -> None:
        super().__init__()

    def get_result(self):
        raise NotImplementedError

    @staticmethod
    def add_parser(parser):
        raise NotImplementedError