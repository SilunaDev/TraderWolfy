# ASTRA FUSION QUANT - Limitations & Risks

1. **Unvalidated Thresholds:** All hardcoded engine configurations (score floors, cooldowns) are *hypotheses* from the research blueprint. They must be validated using the Python engine before live trading.
2. **Execution Slippage:** Backtests assume market fills at the next open. Adverse entry slippage can destroy edge in fast markets.
3. **Black Swan Events:** Volatility regime filters out "EXTREME" volatility, but gaps over stops are not protected against in this model. Use standard account risk management.
