import BaseModelInterface

class ModelEnsemble(BaseModelInterface):

    def __init__(self, model_list) -> None:
        super().__init__()

        self.models = []
        self.model_names = model_list

    def fit(X, y):
        return

    def predict(X, y):
        return 