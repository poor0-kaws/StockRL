# StockRL

`StockRL` is a small reinforcement learning project for trading `SPY`.

The big idea is simple:

1. Load daily price data.
2. Turn prices into easy-to-read features like `RSI`, `MACD`, and moving averages.
3. Let an agent decide whether to be in cash or in `SPY`.
4. Compare the agent against plain old buy-and-hold.

This repo is built to be easy to read.

No clever magic.
No giant framework.
Just a small pipeline you can understand from first principles.

## How the project is shaped

- `stockrl/data_loader.py`
  Reads data, checks the columns, and splits time in the right order.
- `stockrl/features.py`
  Builds indicators from past prices only.
- `stockrl/portfolio_core.py`
  Holds the one true version of the portfolio math.
- `stockrl/trading_env.py`
  Wraps the market into a Gymnasium environment for RL training.
- `stockrl/train.py`
  Trains a PPO agent.
- `stockrl/evaluate.py`
  Runs the trained agent and compares it with buy-and-hold.

## Why next-bar execution matters

This project uses a simple safety rule:

- The agent looks at day `t`.
- The trade happens on day `t + 1`.

That rule matters because otherwise the agent could cheat.
If it sees the end-of-day price and also trades at that same price, it is using information that would not be available in real life.

## Quick start

Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Download `SPY` and train:

```bash
python -m stockrl.train --ticker SPY --start 2012-01-01 --end 2024-12-31
```

Evaluate a saved model:

```bash
python -m stockrl.evaluate --ticker SPY --start 2012-01-01 --end 2024-12-31 --model-path artifacts/ppo_spy.zip
```

## What the agent can do

The action space is intentionally tiny:

- `0` = sell, go flat
- `1` = hold
- `2` = buy, go long

The position space is also tiny:

- `0` = flat, holding cash
- `1` = long, holding `SPY`

This makes the behavior much easier to inspect.

## What gets tested

The tests are designed to catch the quiet bugs that make trading projects lie:

- leakage from the future into today's features
- wrong train/validation/test splits
- broken trade timing
- wrong fee math
- drift between environment math and evaluation math

That is the whole game.
