"""NBEATS deep-learning forecaster."""

from __future__ import annotations

import logging # Додано для приглушення логів Lightning
from loguru import logger

from src.models.dl_wrapper import DeepLearningForecasterWrapper

# Приглушуємо системні повідомлення PyTorch Lightning
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)


class NBEATSForecaster(DeepLearningForecasterWrapper):
    """NBEATS forecasting model (Oreshkin et al., 2019)."""

    def __init__(self, horizon: int, input_size: int | None = None) -> None:
        if input_size is None:
            input_size = 5 * horizon if horizon > 1 else 10

        try:
            from neuralforecast.models import NBEATS
        except Exception as exc:  # noqa: BLE001
            raise ImportError("neuralforecast is required for NBEATSForecaster") from exc

        if horizon == 1:
            stack_types = ["identity"]
            n_blocks = [3]
            mlp_units = [[512, 512]]
            logger.info("h=1 detected: using identity stack for N-BEATS")
        elif horizon <= 3:
            stack_types = ["trend"]
            n_blocks = [3]
            mlp_units = [[512, 512]]
        else:
            stack_types = ["trend", "seasonality"]
            n_blocks = [3, 3]
            mlp_units = [[512, 512], [512, 512]]

        model_kwargs = dict(
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            max_steps=300,
            learning_rate=1e-3,
            early_stop_patience_steps=-1,
            random_seed=42,
            accelerator="cpu",
            enable_progress_bar=False,   # Вимикає ProgressBar Lightning для кожного фолду
            enable_model_summary=False,  # Вимикає таблицю параметрів для кожного фолду
        )

        super().__init__(
            model_class=NBEATS,
            horizon=horizon,
            input_size=input_size,
            **model_kwargs,
        )