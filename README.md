# Bachelor Work

Trading & ML system with multi-agent architecture.

## Structure

- `src/agents/` - Analyst, Manager, Risk, Trader agents
- `src/core/` - Constants, exceptions, logger
- `src/environment/` - Market and backtest environments
- `src/models/` - LLM client and ML inference/training
- `src/pipelines/` - ETL and training pipelines
- `src/tools/` - DB, news/social parsers, time series analyzer

## Run

```bash
uv run python -m src.main --help
```
