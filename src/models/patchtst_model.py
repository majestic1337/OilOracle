from __future__ import annotations
from typing import Any
import inspect
from loguru import logger
from src.models.dl_wrapper import DeepLearningForecasterWrapper

class PatchTSTForecaster(DeepLearningForecasterWrapper):
    def __init__(
        self, 
        horizon: int, 
        input_size: int = 64,
        max_steps: int = 100,
        learning_rate: float = 1e-3
    ) -> None:
        self.horizon = horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        try:
            from neuralforecast.models import PatchTST
        except ImportError as exc:
            raise ImportError("neuralforecast is required") from exc

        logger.info("Initializing PatchTST: h={h}, input={i}", h=horizon, i=input_size)

        # Формуємо kwargs, виключаючи ті, що вже є в сигнатурі враппера
        model_kwargs = {
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "val_check_steps": 1,
            "early_stop_patience_steps": -1,
            "accelerator": "cpu",
            "devices": 1,
            "random_seed": 42,
        }

        super().__init__(
            model_class=PatchTST,
            horizon=horizon,
            input_size=input_size,
            local_scaler_type=None, # Для лог-прибутків
            **model_kwargs,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params: Any) -> "PatchTSTForecaster":
        for key, value in params.items():
            setattr(self, key, value)
        return self