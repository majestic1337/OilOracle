"""ML models for inference and training."""

from src.models.ml.inference import predict
from src.models.ml.train import train_model

__all__ = ["predict", "train_model"]
