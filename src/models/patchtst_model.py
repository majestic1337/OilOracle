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

        params = {
            "h": horizon,
            "input_size": input_size,
            "patch_len": 16,
            "stride": 8,
            "d_model": 128,
            "n_heads": 4,
            "d_ff": 256,
            "dropout": 0.1,
            "max_steps": 300,
            "learning_rate": 1e-4,
            "val_check_steps": 50,
            "early_stop_patience_steps": 10,
            "random_seed": 42,
            "accelerator": "cpu",
            "devices": 1,
            "futr_exog_list": None,
            "hist_exog_list": None,
        }

        try:
            self.model_instance = PatchTST(**params)
        except TypeError as exc:
            logger.warning(f"PatchTST init failed with full params: {exc}")
            logger.warning("Retrying with minimal params")
            # Прибираємо dropout параметри і пробуємо знову
            params.pop("dropout", None)
            params.pop("head_dropout", None)
            self.model_instance = PatchTST(**params)

        params.pop("h", None)
        params.pop("input_size", None)

        super().__init__(
            model_class=PatchTST,
            horizon=horizon,
            input_size=input_size,
            **params,
        )
