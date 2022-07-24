import abc

class BaseModelInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "forward") and 
                callable(subclass.forward) or NotImplemented)

    @abc.abstractmethod
    def forward(self, X):
        raise NotImplementedError
