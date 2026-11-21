from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import joblib

from config import MODEL_PATH, REPORT_DIR, VECTORIZER_PATH
from preprocessing import clean_text


class PredictorError(Exception):
    """Raised when the predictor cannot complete a request."""


@dataclass
class PredictionResult:
    label: str
    confidence: float
    real_probability: float
    fake_probability: float
    word_count: int
    prediction_time_ms: float
    model_used: str = "TF-IDF + Logistic Regression"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "real_probability": round(self.real_probability, 4),
            "fake_probability": round(self.fake_probability, 4),
            "word_count": self.word_count,
            "prediction_time_ms": round(self.prediction_time_ms, 2),
            "model_used": self.model_used,
        }


class FakeNewsPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, vectorizer_path: Path = VECTORIZER_PATH) -> None:
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self._model = None
        self._vectorizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None and self._vectorizer is not None:
                return

            if not self.model_path.exists() or not self.vectorizer_path.exists():
                self._train_missing_artifacts()

            try:
                self._model = joblib.load(self.model_path)
                self._vectorizer = joblib.load(self.vectorizer_path)
            except Exception as exc:  # corrupt / incompatible pickle
                raise PredictorError(
                    "Model files could not be loaded (they may come from a different scikit-learn "
                    "version). Delete model.pkl and vectorizer.pkl, then run `python train.py`."
                ) from exc

    @staticmethod
    def _train_missing_artifacts() -> None:
        """Train on first use so a fresh clone / container works out of the box."""
        try:
            from train import TrainingError, main as train_main

            train_main()
        except TrainingError as exc:
            raise PredictorError(str(exc)) from exc
        except Exception as exc:
            raise PredictorError(f"Automatic training failed: {exc}") from exc

    def predict(self, raw_text: str) -> PredictionResult:
        if not raw_text or not raw_text.strip():
            raise ValueError("Please enter news content or upload a TXT file before predicting.")

        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("The provided content became empty after preprocessing. Please use meaningful news text.")

        self.load()
        start_time = time.perf_counter()
        features = self._vectorizer.transform([cleaned])
        predicted_label = int(self._model.predict(features)[0])
        probabilities = self._model.predict_proba(features)[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        fake_probability = float(probabilities[0])
        real_probability = float(probabilities[1])
        label = "Real News" if predicted_label == 1 else "Fake News"
        confidence = max(fake_probability, real_probability)

        return PredictionResult(
            label=label,
            confidence=confidence,
            real_probability=real_probability,
            fake_probability=fake_probability,
            word_count=len(raw_text.split()),
            prediction_time_ms=elapsed_ms,
        )


def load_training_metrics() -> dict:
    metrics_path = REPORT_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}

    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
