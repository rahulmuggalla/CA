from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
import numpy as np
from .base import BaseModel
seed = 0  # Assuming seed is defined elsewhere

class RandomForest(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = ClassifierChain(
            RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample'),
            order=[0, 1, 2]  # Predict y2, then y3, then y4
        )
        self.predictions = None

    def data_transform(self):
            """Implement the abstract method. No transformation needed for this model."""
            pass

    def train(self, data) -> None:
        self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: np.ndarray):
        self.predictions = self.mdl.predict(X_test)

    def print_chained_accuracy(self, y_true, y_pred):
        scores = []
        for true, pred in zip(y_true, y_pred):
            k = 0
            for i in range(3):
                if true[i] == pred[i]:
                    k += 1
                else:
                    break
            scores.append(k / 3.0)
        accuracy = np.mean(scores)
        print(f"Chained Accuracy: {accuracy * 100:.2f}%")

    def print_stagewise_accuracy(self, y_true, y_pred):
        n = len(y_true)
        correct_type2 = np.sum(y_true[:, 0] == y_pred[:, 0])
        correct_type2_type3 = np.sum((y_true[:, 0] == y_pred[:, 0]) & (y_true[:, 1] == y_pred[:, 1]))
        correct_all = np.sum((y_true[:, 0] == y_pred[:, 0]) & (y_true[:, 1] == y_pred[:, 1]) & (y_true[:, 2] == y_pred[:, 2]))
        print(f"Accuracy for Type2: {correct_type2 / n * 100:.2f}%")
        print(f"Accuracy for Type2 + Type3: {correct_type2_type3 / n * 100:.2f}%")
        print(f"Accuracy for Type2 + Type3 + Type4: {correct_all / n * 100:.2f}%")

        