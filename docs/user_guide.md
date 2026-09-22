# ASTRA FUSION QUANT - User Guide

## Overview
Astra Fusion Quant is a decision-support trading framework implementing the Astra system. It is available as a Python research engine for historical validation and a Pine Script v6 indicator for real-time signaling.

## Setup Instructions
1. Open TradingView.
2. Open the Pine Editor at the bottom of the screen.
3. Paste the contents of `pine/astra_indicator.pine`.
4. Click **Add to Chart**.
5. Wait for the engine to warm up. A dashboard will appear saying "WARMING UP" until 600 chart bars and 250 HTF bars have processed.

## Features
- **Dashboard**: Live engine state, regime detection, volatility status, HTF bias, family scores.
- **Signals**: Buy/Sell markers based on the ensemble scoring thresholds.
- **Alerts**: Fully automated JSON alerts configurable in TradingView.
