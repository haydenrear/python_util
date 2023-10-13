from python_util.delegate.delegates import PythonDelegate


class Delegator:
    def __init__(self, hello: str):
        self.hello = hello

    def return_hello(self) -> str:
        return self.hello


class TestDelegate(object, metaclass=PythonDelegate):
    delegate: Delegator



class RandomObj:
    pass


class AnotherObject(RandomObj, TestDelegate):
    delegate: Delegator
