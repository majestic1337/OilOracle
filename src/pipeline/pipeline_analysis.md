# OilOracle: Brent Crude Oil Forecasting Pipeline — Documentation & Analysis

This document provides a comprehensive overview of the `src/pipeline` architecture, implementation details, and a critical analysis of its current shortcomings as of March 2026.

---

## 🏗️ Architecture Overview

The pipeline follows a modular, decoupled design where data acquisition, model orchestration, validation, and reporting are separate concerns.

### 1. Configuration (`config.py`)
- **Role**: Centralized source of truth using Python `dataclasses`.
- **Implementation**:
    - **Temporal Settings**: Defines training/dev/test boundaries.
    - **Model Registry**: Separate lists for Statistical, ML, ML Classifier, and DL models.
    - **Hyperparameter Anchors**: Lookback candidates, Optuna trials, and DL training constants.
    - **Methodology Flags**: Control ε-thresholding, feature ablation, and multi-run seeding.

### 2. Data Infrastructure (`data.py`)
- **Acquisition**: Uses `yfinance` to fetch OHLCV data. Standardizes columns and handles basic gaps via ffill/bfill.
- **Feature Engineering**: Strictly causal (backward-looking).
    - **Core**: Returns (1d, 5d, 20d), Log-returns, Rolling SMA/STD/Vol/ATR (5, 10, 20, 60 windows).
    - **Technical (Ablation-gated)**: RSI(14), MACD(12,26,9), Bollinger Band Width, OBV.
- **Target Formulation**:
    - Supports both **Regression** (absolute future price) and **Classification** (direction).
    - **Dynamic ε-Thresholding**: Implements a "flat zone" using `ATR(14) × 0.5`. Samples where price change ≤ ε are labeled as `NaN` to optimize model signal-to-noise ratio.

### 3. Model Zoo (`models.py`)
- **Factory Pattern**: `ModelFactory` instantiates all model types via a unified interface (`BaseModel`).
- **Reproducibility**: Global seeding logic (Random, NumPy, PyTorch, Optuna) implemented at the module level and called during model initialization and fitting.
- **Model Families**:
    - **Statistical**: `Auto-ARIMA` (pmdarima) and `Holt-Winters` (statsmodels). Univariate execution on Close price only.
    - **Tree-based ML (Regressors)**: `XGBoost` and `LightGBM` using Multi-Output Regressor wrappers for multi-step forecasting ($h > 1$).
    - **Tree-based ML (Classifiers)**: `XGBoost_cls` and `LightGBM_cls` with **Isotonic Calibration** via `CalibratedClassifierCV`.
    - **Deep Learning (PyTorch)**: `LSTM`, `TCN`, `NBEATS`, and `Simplified TFT`.
        - All support `predict_proba()` (sigmoid-based direction probability).
        - Integrated "DL Multi-run" support to mitigate variance from random initialization.

### 4. Validation Framework (`validation.py`)
- **Hold-out**: Simple 80/20 temporal split for rapid baseline checking.
- **Expanding Window CV**: Uses **Optuna** to optimize hyperparameters within the development set before proceeding to OOS.
- **Walk-Forward Tester**: Simulates production by stepping forward (refitting every $N$ steps) and recording real-time predictions. This is the primary verification tool for OOS performance.

### 5. Metrics & Calculation (`evaluation.py`)
- **Comprehensive Suite**:
    - **Classification**: Accuracy, F1, ROC_AUC, Brier Score, ECE (Expected Calibration Error).
    - **Regression**: RMSE, MAE, MAPE.
    - **Statistical**: `diebold_mariano_test` with Newey-West HAC variance for model-to-model comparison.
- **Financial Simulation**: Long/Flat directional strategy with configurable transaction costs and slippage.

### 6. Reporting Engine (`reporting.py`)
- **Aggregation**: Handles multi-seed results for DL models (Mean/Std/Median).
- **Visualization**: Styled tables and loss curve plotting.
- **Export**: Generates master results CSV and $N \times N$ Diebold-Mariano p-value matrices for each horizon.

---

## 🔍 Critical Analysis: Shortcomings & Omissions

Despite the robust architecture, the following areas represent technical debt or missing professional features:

### 🔴 Data & Features
1.  **Asset Isolation**: The model only searches for signals within Brent crude's own history. It lacks **inter-market features** (WTI, Dollar Index DXY, Gold, S&P 500) and **macro indicators** (OPEC output, interest rates).
2.  **No Sentiment/Alternative Data**: In oil markets, geopolitical news and sentiment often override technical patterns. The pipeline does not incorporate news scrapers or LLM-based sentiment signals.
3.  **Basic Feature Engineering**: Features are manually selected. There is no automated feature selection (e.g., Boruta, LASSO) or automated feature transformation (e.g., Yeo-Johnson).

### 🟠 Models & Optimization
1.  **Limited Search Space**: Optuna only tunes a handful of parameters (mainly tree depth and hidden size). Learning rates, optimizers (Adam vs. SGD), and sequence architecture choices are mostly fixed.
2.  **Modern SOTA Gap**: While N-BEATS and TCN are present, modern Transformer-based SOTA like **PatchTST** or **iTransformer** are missing. The TFT implementation is a non-standard "simplified" version.
3.  **No Ensemble Layer**: There is no dedicated **Meta-Learner** or Stacking ensemble module to combine strengths of ARIMA (trend) and DL (complex patterns).

### 🟡 Training & Infrastructure
1.  **Execution Serialization**: The pipeline relies on in-memory execution. There is no logic for **Model Checkpointing** (saving `.pth` or `.pkl` files), meaning a crash mid-OOS run loses all progress.
2.  **Compute Parallelization**: Walk-Forward testing and CV are sequential. On modern hardware, multi-horizon and multi-model runs could be parallelized across GPUs/cores.
3.  **Data Validation**: Missing "Great Expectations" style checks. No logic to detect price outliers, data drift, or anomalous volume spikes before training.

### 🔵 Financial & Strategy
1.  **Naive Backtest**: The current strategy is Long/Flat. It lacks **Short-selling**, **Stop-loss management**, **Position sizing**, or **Kelly Criterion** logic.
2.  **Risk Metrics**: Missing common hedge-fund metrics like **Sharpe Ratio**, **Sortino Ratio**, or **Value at Risk (VaR)**.

---

## 🚀 Potential Next Steps
- [ ] Integrate **WTI-Brent Spread** as a core feature.
- [ ] Implement **Meta-Stacking** (e.g., using a Ridge regressor to blend XGBoost and LSTM).
- [ ] Add **Model Checkpointing** to prevent loss of long-running WF simulations.
- [ ] Upgrade Financial Backtester to include **Rolling Sharpe Ratio** and **Drawdown Duration**.
