"""NBEATS deep-learning forecaster."""

from __future__ import annotations

from loguru import logger

from src.models.dl_wrapper import DeepLearningForecasterWrapper


class NBEATSForecaster(DeepLearningForecasterWrapper):
    """NBEATS forecasting model (Oreshkin et al., 2019).

    The architecture won the M4 competition by combining trend and seasonality
    basis expansions in a fully-connected stack. On stationary log-returns the
    seasonality component is less pronounced, but the trend block still provides
    a strong autoregressive baseline. We choose NBEATS over LSTM to avoid
    vanishing-gradient issues in long sequences while retaining interpretability.
<<<<<<< HEAD
    Note: stack_types are adapted automatically based on horizon; no manual
    tuning needed.
=======
>>>>>>> origin/main
    """

    def __init__(self, horizon: int, input_size: int | None = None) -> None:
        if input_size is None:
            input_size = 5 * horizon

        try:
            from neuralforecast.models import NBEATS
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("neuralforecast is required for NBEATSForecaster") from exc

        logger.info("Initializing NBEATS forecaster with horizon {h}", h=horizon)

<<<<<<< HEAD
        if horizon == 1:
            # На h=1 trend/seasonality декомпозиція безглузда —
            # використовуємо identity стек (чистий AR baseline)
            logger.warning("h=1 detected: using identity stack for N-BEATS")
            stack_types = ["identity"]
            n_blocks = [3]
            mlp_units = [[512, 512]]
        elif horizon <= 3:
            # На короткому горизонті тільки trend компонент
            stack_types = ["trend"]
            n_blocks = [3]
            mlp_units = [[512, 512]]
        else:
            # h=7+ — повна декомпозиція
            stack_types = ["trend", "seasonality"]
            n_blocks = [3, 3]
            mlp_units = [[512, 512], [512, 512]]

=======
>>>>>>> origin/main
        super().__init__(
            model_class=NBEATS,
            horizon=horizon,
            input_size=input_size,
<<<<<<< HEAD
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
=======
            stack_types=["trend", "seasonality"],
            n_blocks=[3, 3],
            mlp_units=[[512, 512], [512, 512]],
>>>>>>> origin/main
            dropout_prob_theta=0.1,
            max_steps=1000,
            learning_rate=1e-3,
            val_check_steps=50,
            early_stop_patience_steps=10,
            random_seed=42,
        )
