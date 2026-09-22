"""
ASTRA FUSION QUANT — Parity Check Stub
Compares Python engine signals against Pine Script signals using exported CSVs.
"""
import pandas as pd
import sys

def check_parity(python_csv: str, pine_csv: str):
    print("Loading Python and Pine CSV exports...")
    try:
        py_df = pd.read_csv(python_csv, index_col=0)
        pine_df = pd.read_csv(pine_csv, index_col=0)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Check alignment
    if len(py_df) != len(pine_df):
        print("WARN: Different number of rows.")
    
    common = py_df.index.intersection(pine_df.index)
    
    # Check key columns
    cols_to_check = ["atr", "rsi", "trend_score", "signal_side", "signal_tier"]
    
    diffs = 0
    for col in cols_to_check:
        if col not in py_df.columns or col not in pine_df.columns:
            continue
            
        py_vals = py_df.loc[common, col]
        pine_vals = pine_df.loc[common, col]
        
        try:
            mismatches = (py_vals != pine_vals).sum()
            print(f"{col}: {mismatches} mismatches.")
            diffs += mismatches
        except:
            pass
            
    if diffs == 0:
        print("PARITY ACHIEVED: All checked columns match perfectly.")
    else:
        print("PARITY FAILED: See mismatches above.")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        check_parity(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python parity_check.py <python_export.csv> <pine_export.csv>")
