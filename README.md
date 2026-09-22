# ASTRA FUSION QUANT

> **Research / Unvalidated** — This is a TradingView decision-support system.  
> It is **not** an autonomous trading bot. All thresholds are unvalidated hypotheses.  
> No profitability is claimed or implied.

## What It Does

ASTRA FUSION QUANT produces BUY / SELL / STRONG BUY / STRONG SELL / NO TRADE decisions
on standard OHLC candlestick charts by combining:

- Smart Money Concepts (structure, FVG, order blocks, sweeps)
- Trend, momentum, divergence, volume, and volatility families
- Four setup state machines (trend pullback, sweep reversal, breakout retest, range reversal)
- Multi-timeframe context (HTF1 + HTF2 confirmed bars)
- Hard no-trade filters with rejection reason codes

## Target Instruments (Initial)

| Instrument | Preset | Default Chart / HTF1 / HTF2 |
|---|---|---|
| XAUUSD | Commodities | 15m / 1H / 4H |
| BTCUSDT | Crypto | 15m / 1H / 4H |

## Quick Setup — TradingView Indicator

1. Open TradingView → open a standard candlestick chart of XAUUSD or BTCUSDT
2. Open **Pine Editor** (bottom panel)
3. Paste contents of `pine/astra_indicator.pine` → **Save** → **Add to chart**
4. In indicator settings, select **Asset Preset** matching your instrument
5. Use the **General / Intraday** style for 15-minute charts
6. Wait for **WARMING_UP** to clear (needs 600+ chart bars)
7. Read signals in the dashboard overlay

## Quick Setup — Strategy Backtest

1. Paste `pine/astra_strategy.pine` into Pine Editor as a **separate** script
2. Set **Properties**: capital, commission, slippage per your actual venue
3. Run from **Strategy Tester** tab

## Python Research Engine

```bash
pip install -r requirements.txt
python tools/build.py          # verify Pine engine hash
python -m pytest tests/ -v     # run all fixtures
python research/validation/baselines.py --instrument XAUUSD
```

Data is fetched via `yfinance` by default. See `docs/user_guide.md` for CSV/Parquet ingestion.

## Project Structure

```
config/          Configuration registry (presets, asset metadata, schema)
pine/engine/     Canonical Pine engine modules
pine/            Assembled standalone indicator + strategy
research/        Python research engine (features, setups, scoring, validation)
tests/fixtures/  Synthetic timing and arithmetic fixtures
tools/           Build, parity-check, export utilities
docs/            User guide, alert setup, paper trading, limitations
reports/         Validation outputs (NOT RUN until data provided)
experiment_ledger.csv  All attempted configs and trial decisions
```

## Evidence Status

| Component | Status |
|---|---|
| Pine indicator compiles | NOT RUN — paste into TradingView to verify |
| Pine strategy compiles | NOT RUN — paste into TradingView to verify |
| Python fixtures pass | Run `pytest tests/ -v` |
| Backtest results | NOT RUN — requires historical data |
| Paper trading | NOT RUN — requires 30 sessions / 50 signals |
| Final holdout | SEALED — do not unseal until parameters frozen |

## Limitations

See `docs/limitations.md` for full list. Key constraints:
- Signals are for decision support only; not broker-connected
- Short signals require instrument to actually permit shorting
- No news filter, no bid/ask data, no tick-level fills
- Backtest fills are broker-emulator estimates, not real execution

## License

Research use only. Not financial advice.
