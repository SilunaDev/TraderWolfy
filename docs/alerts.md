# ASTRA FUSION QUANT - Alerts Integration

Astra outputs signals in JSON format for easy routing to webhooks (e.g., 3Commas, Make, custom bots).

## JSON Schema
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "entry": 65000.0,
  "stop": 64500.0,
  "tp": 66500.0
}
```

## Setup in TradingView
1. Add the indicator to your chart.
2. Click the **Alerts** icon.
3. Select "ASTRA FUSION QUANT" as the condition.
4. Set Webhook URL to your bot.
5. In the message box, use `{{strategy.order.alert_message}}` or leave it to emit the internal alert JSON directly.
