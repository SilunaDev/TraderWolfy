"""
ASTRA FUSION QUANT — Export Fixtures
Exports python calculation intermediate results to CSV for Pine Script validation.
"""
import pandas as pd
import sys

def export_fixtures(df: pd.DataFrame, out_path: str):
    """
    Exports required columns for Pine validation.
    Expects df to have columns: open, high, low, close, volume,
    plus calculated fields like atr, rsi, ema20, ema50, ema200, 
    signal_side, net_rr, etc.
    """
    cols = ["open", "high", "low", "close", "volume"]
    extra_cols = ["atr", "rsi", "trend_score", "signal_side", "signal_tier", "net_rr"]
    
    available = cols + [c for c in extra_cols if c in df.columns]
    df[available].to_csv(out_path)
    print(f"Exported fixtures to {out_path}")

if __name__ == "__main__":
    print("Fixture exporter ready.")
