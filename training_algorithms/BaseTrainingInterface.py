import abc

class BaseTrainingInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "train") and 
                callable(subclass.train) or NotImplemented)

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
