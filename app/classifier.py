import joblib
from . import config

class Classifier:
    def __init__(self, path=config.MODEL_PATH):
        self.model = joblib.load(path)
        self.labels = list(self.model.classes_)

    def predict(self, text: str):
        proba = self.model.predict_proba([text])[0]
        probs = {lab: round(float(p), 4) for lab, p in zip(self.labels, proba)}
        label = max(probs, key=probs.get)
        conf = probs[label]
        return {"label": label, "confidence": conf, "yakin": conf >= config.CONF_THRESHOLD,
                "probabilities": dict(sorted(probs.items(), key=lambda kv: -kv[1]))}
