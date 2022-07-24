from turtle import forward
import torch.nn as nn
from BaseModelInterface import BaseModelInterface

class ANN(BaseModelInterface):

    def __init__(self, nn_config) -> None:
        super().__init__()
        self.model = self.__create_nn(nn_config)

    def __create_nn(nn_config):
        return

    def forward(self, X):
        return self.model(X)
