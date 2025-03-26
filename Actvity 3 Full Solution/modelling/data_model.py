from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from Config import Config

class Data:
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        self.X = X
        # Extract multiple labels
        self.y = df[Config.TYPE_COLS].values  # Shape: (n_samples, 3), Config.TYPE_COLS = ['y2', 'y3', 'y4']
        
        # Encode each label column
        self.label_encoders = []
        for i in range(3):
            le = LabelEncoder()
            self.y[:, i] = le.fit_transform(self.y[:, i])
            self.label_encoders.append(le)
        self.y = self.y.astype(int)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0
        )