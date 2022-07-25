import abc

class GameInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "get_actions") and 
                callable(subclass.get_actions) and 
                hasattr(subclass, "get_current_state") and 
                callable(subclass.get_current_state) or NotImplemented)

    @abc.abstractmethod
    def get_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def play_step():
        raise NotImplementedError
