from abc import ABCMeta, abstractmethod
import json
import abc

class FromJsonClass(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def from_dict(cls, message: dict):
        pass

    @classmethod
    def fromJSON(cls, message: str):
        return cls.from_dict(json.loads(message))


class ToJsonClass(metaclass=ABCMeta):

    @abc.abstractmethod
    def toJSON(self) -> str:
        pass

