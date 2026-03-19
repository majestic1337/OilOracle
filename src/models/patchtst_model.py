"""PatchTST deep-learning forecaster."""

from __future__ import annotations

from loguru import logger

from src.models.dl_wrapper import DeepLearningForecasterWrapper


class PatchTSTForecaster(DeepLearningForecasterWrapper):
    """PatchTST forecaster (Nie et al., 2023).

    PatchTST is a patch-based transformer that processes each channel
    independently, reducing sequence length while preserving local structure.
    This design directly addresses critiques of vanilla transformers for time
    series (Zeng et al., 2023) by using patches to capture short-term patterns
    without overwhelming attention costs.
    """

    def __init__(self, horizon: int, input_size: int | None = None) -> None:
        if input_size is None:
            input_size = max(4 * horizon, 64)

        try:
            from neuralforecast.models import PatchTST
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("neuralforecast is required for PatchTSTForecaster") from exc

        logger.info("Initializing PatchTST forecaster with horizon {h}", h=horizon)

        params = dict(
            h=horizon,
            input_size=input_size,
            max_steps=300,
            learning_rate=1e-4,
            val_check_steps=50,
            early_stop_patience_steps=10,
            random_seed=42,
            accelerator="cpu",
        )
        self.model_instance = PatchTST(**params)

        super().__init__(
            model_class=PatchTST,
            horizon=horizon,
            input_size=input_size,
            **params,
        )
