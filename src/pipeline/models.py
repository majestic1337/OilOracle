"""Model Zoo — unified factory for Statistical, ML, and DL models.

All models expose a consistent .fit() / .predict() / .predict_proba() interface.
ML and DL models use MIMO / Direct multi-step forecasting.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Change 1 · Global seed helper
# ═════════════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for full reproducibility.

    Call once at the start of each experiment (notebook first cell) and
    again before each DL multi-run (change 2) with the run-specific seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import optuna
        optuna.samplers.TPESampler(seed=seed)
    except ImportError:
        pass
    logger.debug("Global seed set to %d", seed)


# ═════════════════════════════════════════════════════════════════════════════
# Abstract Base
# ═════════════════════════════════════════════════════════════════════════════

class BaseModel(ABC):
    """Unified interface for all model families."""

    name: str

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        """Train and return metadata dict (train_time, loss_history, params…)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted price (regression) or class label array."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class (direction = UP).

        Default raises NotImplementedError.  Override in classifier subclasses
        and DL models that output a sigmoid/softmax probability.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba()."
        )

    def get_params(self) -> Dict[str, Any]:
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# Statistical Models
# ═════════════════════════════════════════════════════════════════════════════

class ARIMAModel(BaseModel):
    """Auto-ARIMA via pmdarima — univariate, fit on Close only."""

    name = "ARIMA"

    def __init__(self) -> None:
        self._model: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        import pmdarima as pm

        close_series = X_train[:, 0] if X_train.ndim > 1 else X_train
        t0 = time.time()
        self._model = pm.auto_arima(
            close_series, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action="ignore",
            max_p=5, max_q=5, max_d=2,
        )
        train_time = time.time() - t0
        return {
            "train_time": train_time,
            "params": {"order": self._model.order},
            "loss_history": [],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if X.ndim > 1 else len(X)
        return self._model.predict(n_periods=n)

    def get_params(self) -> Dict[str, Any]:
        if self._model is not None:
            return {"order": self._model.order}
        return {}


class HoltWintersModel(BaseModel):
    """Holt-Winters Exponential Smoothing — univariate."""

    name = "HoltWinters"

    def __init__(self) -> None:
        self._model: Any = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        close_series = X_train[:, 0] if X_train.ndim > 1 else X_train
        t0 = time.time()
        self._model = ExponentialSmoothing(
            close_series, trend="add", seasonal=None,
        ).fit(optimized=True, use_brute=False)
        train_time = time.time() - t0
        return {"train_time": train_time, "params": {}, "loss_history": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if X.ndim > 1 else len(X)
        return self._model.forecast(n)

    def get_params(self) -> Dict[str, Any]:
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# Tree-based ML Models — Regressors (unchanged interface)
# ═════════════════════════════════════════════════════════════════════════════

class XGBoostModel(BaseModel):
    """XGBoost with Direct / MIMO multi-output wrapper (regression)."""

    name = "XGBoost"

    def __init__(self, horizon: int = 1, cfg: Optional[PipelineConfig] = None, **params: Any) -> None:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        if cfg is not None:
            set_global_seed(cfg.GLOBAL_SEED)
        self.horizon = horizon
        self._params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "tree_method": "hist",
            "verbosity": 0,
        }
        base = XGBRegressor(**self._params)
        self._model = MultiOutputRegressor(base) if horizon > 1 else base
        self._scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        X_s = self._scaler.fit_transform(X_train)
        t0 = time.time()
        self._model.fit(X_s, y_train)
        train_time = time.time() - t0
        return {"train_time": train_time, "params": self._params, "loss_history": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        pred = self._model.predict(X_s)
        if pred.ndim == 2:
            return pred[:, -1]
        return pred

    def get_params(self) -> Dict[str, Any]:
        return self._params


class LightGBMModel(BaseModel):
    """LightGBM with Direct / MIMO multi-output wrapper (regression)."""

    name = "LightGBM"

    def __init__(self, horizon: int = 1, cfg: Optional[PipelineConfig] = None, **params: Any) -> None:
        import lightgbm as lgb
        from sklearn.multioutput import MultiOutputRegressor

        if cfg is not None:
            set_global_seed(cfg.GLOBAL_SEED)
        self.horizon = horizon
        self._params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "verbosity": -1,
        }
        base = lgb.LGBMRegressor(**self._params)
        self._model = MultiOutputRegressor(base) if horizon > 1 else base
        self._scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        X_s = self._scaler.fit_transform(X_train)
        t0 = time.time()
        self._model.fit(X_s, y_train)
        train_time = time.time() - t0
        return {"train_time": train_time, "params": self._params, "loss_history": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        pred = self._model.predict(X_s)
        if pred.ndim == 2:
            return pred[:, -1]
        return pred

    def get_params(self) -> Dict[str, Any]:
        return self._params


# ═════════════════════════════════════════════════════════════════════════════
# Tree-based ML Models — Classifiers with Isotonic Calibration (Change 2)
# ═════════════════════════════════════════════════════════════════════════════

class XGBoostClassifierModel(BaseModel):
    """XGBoost binary classifier with optional CalibratedClassifierCV.

    Outputs probability of direction = UP (class 1).
    Regression metrics (RMSE/MAE/MAPE) are not applicable to this model;
    classification and calibration metrics (Accuracy, F1, Brier, ECE) apply.
    """

    name = "XGBoost_cls"

    def __init__(self, cfg: PipelineConfig, **params: Any) -> None:
        from xgboost import XGBClassifier

        set_global_seed(cfg.GLOBAL_SEED)
        self.cfg = cfg
        self._params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
        }
        self._base = XGBClassifier(**self._params)
        self._model: Any = None  # set in fit() after calibration
        self._scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        from sklearn.calibration import CalibratedClassifierCV

        X_s = self._scaler.fit_transform(X_train)
        t0 = time.time()
        self._base.fit(X_s, y_train)
        if self.cfg.use_calibration:
            self._model = CalibratedClassifierCV(
                self._base, method="isotonic", cv="prefit"
            )
            self._model.fit(X_s, y_train)
        else:
            self._model = self._base
        train_time = time.time() - t0
        return {"train_time": train_time, "params": self._params, "loss_history": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        return self._model.predict(X_s)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(direction = UP) for each sample."""
        X_s = self._scaler.transform(X)
        return self._model.predict_proba(X_s)[:, 1]

    def get_params(self) -> Dict[str, Any]:
        return self._params


class LightGBMClassifierModel(BaseModel):
    """LightGBM binary classifier with optional CalibratedClassifierCV.

    Outputs probability of direction = UP (class 1).
    """

    name = "LightGBM_cls"

    def __init__(self, cfg: PipelineConfig, **params: Any) -> None:
        import lightgbm as lgb

        set_global_seed(cfg.GLOBAL_SEED)
        self.cfg = cfg
        self._params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "verbosity": -1,
        }
        self._base = lgb.LGBMClassifier(**self._params)
        self._model: Any = None
        self._scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        from sklearn.calibration import CalibratedClassifierCV

        X_s = self._scaler.fit_transform(X_train)
        t0 = time.time()
        self._base.fit(X_s, y_train)
        if self.cfg.use_calibration:
            self._model = CalibratedClassifierCV(
                self._base, method="isotonic", cv="prefit"
            )
            self._model.fit(X_s, y_train)
        else:
            self._model = self._base
        train_time = time.time() - t0
        return {"train_time": train_time, "params": self._params, "loss_history": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        return self._model.predict(X_s)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(direction = UP) for each sample."""
        X_s = self._scaler.transform(X)
        return self._model.predict_proba(X_s)[:, 1]

    def get_params(self) -> Dict[str, Any]:
        return self._params


# ═════════════════════════════════════════════════════════════════════════════
# Deep Learning Models (PyTorch) — MIMO / Direct architecture
# ═════════════════════════════════════════════════════════════════════════════

class _PyTorchBase(BaseModel):
    """Shared PyTorch training loop for sequence models."""

    def __init__(self, lookback: int, horizon: int, cfg: PipelineConfig, **kw: Any) -> None:
        import torch
        import torch.nn as nn

        set_global_seed(cfg.GLOBAL_SEED)  # Change 1: seed on init
        self.lookback = lookback
        self.horizon = horizon
        self.cfg = cfg
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: nn.Module | None = None
        self._loss_history: Dict[str, List[float]] = {"train": [], "val": []}

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Any:
        """Convert tabular data to (batch, seq_len, features) tensors."""
        import torch

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,F)
        if y is not None:
            yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            return Xt, yt
        return Xt

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kw: Any) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        set_global_seed(self.cfg.GLOBAL_SEED)  # Change 1: seed on fit
        X_val = kw.get("X_val")
        y_val = kw.get("y_val")

        # Scale
        Xs = self._scaler_x.fit_transform(X_train)
        ys = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        Xt, yt = self._make_sequences(Xs, ys)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False)

        # Validation set
        val_dl = None
        if X_val is not None and y_val is not None:
            Xvs = self._scaler_x.transform(X_val)
            yvs = self._scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            Xvt, yvt = self._make_sequences(Xvs, yvs)
            val_dl = DataLoader(TensorDataset(Xvt, yvt), batch_size=256, shuffle=False)

        assert self._net is not None
        self._net.to(self._device)
        opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val = float("inf")
        patience_ctr = 0
        best_state = None
        epoch = 0

        t0 = time.time()
        for epoch in range(self.cfg.max_epochs):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                out = self._net(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(ds)
            self._loss_history["train"].append(epoch_loss)

            # Validation
            if val_dl is not None:
                self._net.eval()
                vloss = 0.0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb, yb = xb.to(self._device), yb.to(self._device)
                        vloss += criterion(self._net(xb), yb).item() * xb.size(0)
                vloss /= len(val_dl.dataset)  # type: ignore[arg-type]
                self._loss_history["val"].append(vloss)
                if vloss < best_val:
                    best_val = vloss
                    patience_ctr = 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.cfg.patience:
                        logger.info("%s early stop @ epoch %d", self.name, epoch + 1)
                        break
            else:
                self._loss_history["val"].append(epoch_loss)

        if best_state is not None:
            self._net.load_state_dict(best_state)
        train_time = time.time() - t0
        return {
            "train_time": train_time,
            "params": {"epochs_run": epoch + 1, "lookback": self.lookback, "horizon": self.horizon},
            "loss_history": self._loss_history,
            "early_stop_epoch": epoch + 1 if patience_ctr >= self.cfg.patience else None,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        assert self._net is not None
        self._net.eval()
        Xs = self._scaler_x.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1).to(self._device)
        with torch.no_grad():
            out = self._net(Xt).cpu().numpy()
        return self._scaler_y.inverse_transform(out.reshape(-1, 1)).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid-transformed probability of direction = UP.

        The network outputs a raw scalar; sigmoid maps it to [0, 1].
        No additional calibration step is needed (use_calibration=False
        in config for DL models).
        """
        import torch

        assert self._net is not None
        self._net.eval()
        Xs = self._scaler_x.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1).to(self._device)
        with torch.no_grad():
            logits = self._net(Xt).cpu().numpy().ravel()
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid


class LSTMModel(_PyTorchBase):
    """LSTM direct forecaster."""

    name = "LSTM"

    def __init__(self, lookback: int, horizon: int, cfg: PipelineConfig, n_features: int = 1, **kw: Any) -> None:
        super().__init__(lookback, horizon, cfg, **kw)
        import torch.nn as nn

        hidden = kw.get("hidden_size", 64)

        class _Net(nn.Module):
            def __init__(self, inp: int, hid: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, num_layers=2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hid, 1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                _, (h, _) = self.lstm(x)
                return self.fc(h[-1])

        self._net = _Net(n_features, hidden)


class TCNBlock:
    """Helper — builds a causal dilated conv block."""
    pass


class TCNModel(_PyTorchBase):
    """Temporal Convolutional Network — direct forecast."""

    name = "TCN"

    def __init__(self, lookback: int, horizon: int, cfg: PipelineConfig, n_features: int = 1, **kw: Any) -> None:
        super().__init__(lookback, horizon, cfg, **kw)
        import torch
        import torch.nn as nn

        channels = kw.get("channels", 64)

        class _CausalConv1d(nn.Module):
            def __init__(self, in_c: int, out_c: int, k: int, d: int) -> None:
                super().__init__()
                self.pad = (k - 1) * d
                self.conv = nn.Conv1d(in_c, out_c, k, dilation=d)
                self.bn = nn.BatchNorm1d(out_c)
                self.act = nn.ReLU()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                x = nn.functional.pad(x, (self.pad, 0))
                return self.act(self.bn(self.conv(x)))

        class _Net(nn.Module):
            def __init__(self, inp: int, ch: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    _CausalConv1d(inp, ch, 3, 1),
                    _CausalConv1d(ch, ch, 3, 2),
                    _CausalConv1d(ch, ch, 3, 4),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Linear(ch, 1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                x = x.permute(0, 2, 1)  # (B, F, T)
                x = self.layers(x).squeeze(-1)
                return self.fc(x)

        self._net = _Net(n_features, channels)


class NBEATSModel(_PyTorchBase):
    """N-BEATS — direct multi-step forecasting."""

    name = "NBEATS"

    def __init__(self, lookback: int, horizon: int, cfg: PipelineConfig, n_features: int = 1, **kw: Any) -> None:
        super().__init__(lookback, horizon, cfg, **kw)
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self, inp: int, hid: int) -> None:
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(inp, hid), nn.ReLU(),
                    nn.Linear(hid, hid), nn.ReLU(),
                    nn.Linear(hid, hid), nn.ReLU(),
                )
                self.back = nn.Linear(hid, inp)
                self.fore = nn.Linear(hid, 1)

            def forward(self, x: "torch.Tensor") -> "tuple":
                h = self.fc(x)
                return x - self.back(h), self.fore(h)

        class _Net(nn.Module):
            def __init__(self, inp: int, hid: int = 128, n_blocks: int = 3) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([_Block(inp, hid) for _ in range(n_blocks)])

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                x = x.squeeze(1)  # (B, F)
                forecast = 0.0
                for blk in self.blocks:
                    x, f = blk(x)
                    forecast = forecast + f
                return forecast

        self._net = _Net(n_features)


class TFTModel(_PyTorchBase):
    """Simplified Temporal Fusion Transformer — attention + gating."""

    name = "TFT"

    def __init__(self, lookback: int, horizon: int, cfg: PipelineConfig, n_features: int = 1, **kw: Any) -> None:
        super().__init__(lookback, horizon, cfg, **kw)
        import torch
        import torch.nn as nn

        d_model = kw.get("d_model", 64)

        class _Net(nn.Module):
            def __init__(self, inp: int, d: int) -> None:
                super().__init__()
                self.proj = nn.Linear(inp, d)
                encoder_layer = nn.TransformerEncoderLayer(
                    d, nhead=4, dim_feedforward=d * 2, batch_first=True, dropout=0.1
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = nn.Linear(d, 1)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                x = self.proj(x)
                x = self.encoder(x)
                return self.fc(x[:, -1, :])

        self._net = _Net(n_features, d_model)


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

class ModelFactory:
    """Instantiate any model from the zoo by name."""

    _REGISTRY = {
        "ARIMA": ARIMAModel,
        "HoltWinters": HoltWintersModel,
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "XGBoost_cls": XGBoostClassifierModel,   # Change 2: classifier variant
        "LightGBM_cls": LightGBMClassifierModel,  # Change 2: classifier variant
        "LSTM": LSTMModel,
        "TCN": TCNModel,
        "NBEATS": NBEATSModel,
        "TFT": TFTModel,
    }

    @classmethod
    def create(
        cls,
        name: str,
        lookback: int,
        horizon: int,
        cfg: PipelineConfig,
        n_features: int = 1,
        **params: Any,
    ) -> BaseModel:
        if name not in cls._REGISTRY:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._REGISTRY)}")
        klass = cls._REGISTRY[name]
        # Statistical models take no extra args
        if name in ("ARIMA", "HoltWinters"):
            return klass()
        # Classifier models need cfg for calibration flag
        if name in ("XGBoost_cls", "LightGBM_cls"):
            return klass(cfg=cfg, **params)
        # Regressor ML models need horizon + optional params
        if name in ("XGBoost", "LightGBM"):
            return klass(horizon=horizon, cfg=cfg, **params)
        # DL models need lookback, horizon, cfg, n_features
        return klass(lookback=lookback, horizon=horizon, cfg=cfg, n_features=n_features, **params)
