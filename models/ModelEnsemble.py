import BaseModelInterface

class ModelEnsemble(BaseModelInterface):

    def __init__(self, model_list, ensemble_method) -> None:
        super().__init__()

        self.models = []
        self.model_names = model_list
        self.method = ensemble_method

    def fit(self, X, y):
        return

    def forward(self, X):
        return

    def predict(self, X):
        return 
