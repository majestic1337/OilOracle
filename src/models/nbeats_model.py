from __future__ import annotations
from typing import Any
import logging
from loguru import logger
from src.models.dl_wrapper import DeepLearningForecasterWrapper

# Приглушуємо логі Lightning
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

class NBEATSForecaster(DeepLearningForecasterWrapper):
    def __init__(
        self, 
        horizon: int, 
        input_size: int = 30,
        max_steps: int = 100,
        learning_rate: float = 1e-3
    ) -> None:
        self.horizon = horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        try:
            from neuralforecast.models import NBEATS
        except ImportError as exc:
            raise ImportError("neuralforecast is required for NBEATSForecaster") from exc

        # Логіка вибору стеку (trend/seasonality)
        if horizon == 1:
            stack_types = ["identity"]
            n_blocks = [3]
            mlp_units = [[512, 512]]
        elif horizon <= 3:
            stack_types = ["trend"]
            n_blocks = [3]
            mlp_units = [[512, 512]]
        else:
            stack_types = ["trend", "seasonality"]
            n_blocks = [3, 3]
            mlp_units = [[512, 512], [512, 512]]

        logger.info("Initializing N-BEATS: h={h}, input={i}", h=horizon, i=input_size)

        model_kwargs = {
            "stack_types": stack_types,
            "n_blocks": n_blocks,
            "mlp_units": mlp_units,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "early_stop_patience_steps": -1,
            "random_seed": 42,
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "val_check_steps": 1,
        }

        super().__init__(
            model_class=NBEATS,
            horizon=horizon,
            input_size=input_size,
            local_scaler_type=None,  # Працюємо з лог-прибутками
            **model_kwargs,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params: Any) -> "NBEATSForecaster":
        for key, value in params.items():
            setattr(self, key, value)
        # Sync model_kwargs so _build_model uses updated values on next fit
        if hasattr(self, "model_kwargs"):
            for key in ("max_steps", "learning_rate"):
                if key in params:
                    self.model_kwargs[key] = params[key]
        return self