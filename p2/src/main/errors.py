class BaseError(Exception):
    pass

class InvalidModelInputError(BaseError):
    pass

class InvalidResultOfProcess(BaseError):
    pass
