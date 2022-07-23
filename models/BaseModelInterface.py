import abc

class BaseModelInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "fit") and 
                callable(subclass.fit) and 
                hasattr(subclass, "predict") and 
                callable(subclass.predict) or NotImplemented)

    @abc.abstractmethod
    def fit(X, y):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(X):
        raise NotImplementedError
