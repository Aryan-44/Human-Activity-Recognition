"""Loads a trained activity model (see train.py) and runs predictions."""

from pathlib import Path

import joblib

from features import FEATURE_NAMES


DEFAULT_MODEL_PATH = Path("models") / "activity_model.joblib"


class ActivityModel:
    def __init__(self, path=DEFAULT_MODEL_PATH):
        bundle = joblib.load(path)
        if list(bundle["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError(
                f"{path} was trained with different features. Run train.py again."
            )
        self.classifier = bundle["classifier"]
        self.name = bundle.get("model_name", "model")
        self.labels = list(self.classifier.classes_)

    def predict(self, features):
        """Return (label, probability) for one feature vector."""
        probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
        best = probabilities.argmax()
        return self.labels[best], float(probabilities[best])
