# Dry Run Mode with Price Action Strategy

This implementation adds a comprehensive dry run mode to the pump.fun trading bot that simulates trading new meme coins without spending real SOL, while implementing sophisticated price action strategies for automated trading decisions.

## Features

### 🎯 Core Dry Run Functionality
- **Risk-Free Testing**: Simulate trading without spending real SOL
- **Real-Time Price Analysis**: Monitor actual price movements and apply strategies
- **Performance Tracking**: Comprehensive reporting on simulated trading performance
- **Position Management**: Track multiple simulated positions with automatic exit strategies

### 📈 Price Action Strategy
- **Multi-Signal Analysis**: Combines price movement, volatility, momentum, and volume
- **Dynamic Signal Generation**: Generates BUY/SELL/HOLD signals with confidence scores
- **Technical Indicators**: 
  - Price change percentage analysis
  - Volatility measurement (standard deviation)
  - Momentum scoring (rate of change acceleration)
  - Volume surge detection
- **Risk Management**: Automatic stop-loss and take-profit execution

### 🤖 Automated Trading Logic
- **Smart Entry**: Only buys tokens with positive price action signals
- **Exit Strategies**: Multiple exit conditions (stop-loss, take-profit, price action reversal)
- **Position Monitoring**: Continuous monitoring of open positions
- **Portfolio Management**: Balance tracking and position limits

## Configuration

### Bot Configuration (`bots/bot-dry-run-meme-sniper.yaml`)

```yaml
# Enable dry run mode
dry_run:
  enabled: true
  initial_balance: 1.0     # Starting SOL balance
  trade_amount: 0.1        # SOL per trade
  max_positions: 5         # Max concurrent positions
  stop_loss_pct: 25.0      # 25% stop loss
  take_profit_pct: 100.0   # 100% take profit
  
  # Price action strategy settings
  price_action:
    min_price_points: 3         # Min data points for analysis
    analysis_window: 180        # 3 minutes analysis window
    volatility_threshold: 0.15  # 15% volatility threshold
    momentum_threshold: 0.08    # 8% momentum threshold
    volume_surge_multiplier: 3.0 # 3x volume surge detection
```

### Key Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `initial_balance` | Starting SOL for simulation | 1.0 |
| `trade_amount` | SOL amount per trade | 0.1 |
| `max_positions` | Maximum concurrent positions | 5 |
| `stop_loss_pct` | Stop loss percentage | 25.0% |
| `take_profit_pct` | Take profit percentage | 100.0% |
| `analysis_window` | Price analysis time window | 180s |
| `volatility_threshold` | Volatility threshold for signals | 15% |

## Usage

### Running the Dry Run Bot

```bash
# 1. Enable the dry run bot in the config
# Edit bots/bot-dry-run-meme-sniper.yaml and set enabled: true

# 2. Run the bot
uv run pump_bot

# 3. Monitor the logs for trading activity
tail -f logs/dry-run-meme-sniper_*.log
```

### Running the Demo

```bash
# Test the dry run functionality with simulated data
uv run learning-examples/dry_run_demo.py
```

## Price Action Strategy Details

### Signal Generation

The price action strategy analyzes multiple factors to generate trading signals:

1. **Price Movement Analysis**
   - Tracks price changes over the analysis window
   - Identifies bullish/bearish trends
   - Detects breakouts and breakdowns

2. **Volatility Assessment**
   - Calculates standard deviation of price changes
   - High volatility + positive movement = BREAKOUT signal
   - High volatility + negative movement = BREAKDOWN signal

3. **Momentum Scoring**
   - Measures rate of price change acceleration
   - Positive momentum strengthens bullish signals
   - Negative momentum strengthens bearish signals

4. **Volume Analysis**
   - Detects volume surges during price movements
   - High volume confirms price action validity
   - Adjusts signal confidence based on volume

### Trading Signals

| Signal | Description | Action |
|--------|-------------|--------|
| `STRONG_BUY` | High confidence bullish signal | Immediate buy |
| `BUY` | Moderate confidence bullish signal | Buy if conditions met |
| `HOLD` | Neutral or uncertain signal | No action |
| `SELL` | Moderate confidence bearish signal | Sell position |
| `STRONG_SELL` | High confidence bearish signal | Immediate sell |

### Price Actions

| Action | Trigger | Strategy |
|--------|---------|----------|
| `BULLISH` | Price increase > 5% | Positive momentum |
| `BEARISH` | Price decrease > 5% | Negative momentum |
| `BREAKOUT` | High volatility + bullish | Strong buy signal |
| `BREAKDOWN` | High volatility + bearish | Strong sell signal |
| `CONSOLIDATION` | High volatility + neutral | Wait for direction |

## Output and Reporting

### Real-Time Logs

The bot provides detailed logging of all activities:

```
DRY RUN BUY: MoonCoin at 0.00000150 SOL (Signal: strong_buy, Confidence: 0.85)
DRY RUN SELL: MoonCoin at 0.00000225 SOL (PnL: 0.005000 SOL, 50.00%)
```

### Performance Reports

Detailed performance reports are generated in `dry_run_results/`:

```json
{
  "summary": {
    "initial_balance": 1.0,
    "current_balance": 0.95,
    "portfolio_value": 1.15,
    "total_return_pct": 15.0,
    "total_pnl": 0.15
  },
  "trading_stats": {
    "total_trades": 10,
    "winning_trades": 7,
    "losing_trades": 3,
    "win_rate_pct": 70.0,
    "best_trade": 0.08,
    "worst_trade": -0.025
  }
}
```

### Trade History

Individual trades are logged in `dry_run_results/dry_run_trades.jsonl`:

```json
{"timestamp": "2024-01-15T10:30:00Z", "action": "buy", "token": "MOON", "price": 0.00000150, "analysis": {...}}
{"timestamp": "2024-01-15T10:35:00Z", "action": "sell", "token": "MOON", "price": 0.00000225, "pnl_pct": 50.0}
```

## Architecture

### Key Components

1. **`DryRunTrader`**: Base class for simulated trading
2. **`PriceActionStrategy`**: Price analysis and signal generation
3. **`DryRunBuyer`/`DryRunSeller`**: Specialized traders for buy/sell operations
4. **`UniversalTrader`**: Enhanced to support dry run mode

### Integration Points

- **Bot Runner**: Modified to pass dry run configuration
- **Universal Trader**: Enhanced with dry run mode detection
- **Configuration**: Extended YAML config support for dry run settings

## Benefits

### For Development
- **Safe Testing**: Test strategies without financial risk
- **Strategy Validation**: Validate price action strategies with real data
- **Performance Analysis**: Understand strategy effectiveness before deployment

### For Research
- **Backtesting**: Analyze historical performance patterns
- **Strategy Optimization**: Tune parameters for better performance
- **Market Understanding**: Learn meme coin price patterns

### For Production
- **Risk Assessment**: Evaluate strategies before committing real funds
- **Parameter Tuning**: Optimize settings based on dry run results
- **Confidence Building**: Gain confidence in automated trading strategies

## Next Steps

1. **Enable the dry run bot** by setting `enabled: true` in the config
2. **Run the demo** to see the functionality in action
3. **Monitor performance** using the generated reports
4. **Tune parameters** based on dry run results
5. **Graduate to live trading** once confident in the strategy

## Safety Notes

- Dry run mode uses simulated prices - actual market conditions may vary
- Always validate strategies with paper trading before using real funds
- Monitor market conditions and adjust parameters accordingly
- The bot is designed for educational and testing purposes

---

Happy trading! 🚀 Remember: This is simulation mode - perfect for learning and testing strategies risk-free.