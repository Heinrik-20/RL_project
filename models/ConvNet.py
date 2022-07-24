from BaseModelInterface import BaseModelInterface
import torch.nn as nn

class ConvNet(BaseModelInterface):

    def __init__(self, height, width, output, model_specs=None) -> None:
        super().__init__()
        self.model = self.__build_model(height, width, output, model_specs)

    def forward(self, X):
        return self.model(X)

    def __build_model(height, width, output, model_specs):

        #TODO: implement with a simple step
        if model_specs:
            return 

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32)
        )

        convW = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convH = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        model.append(nn.Linear(convW * convH * 32, output))

        return model
