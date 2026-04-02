# tda-backtester

Algorithmic trading backtesting framework with statistical analysis.

## Features

* Built-in strategies (SMA Crossover, Bollinger Mean Reversion, Breakout Momentum).
* FastAPI REST API (`api/`).
* Statistical metrics calculation (`backtester/stats/`).
* React/Vite dashboard (`dashboard/`).

## Installation

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### Example script

Run the provided example script to backtest built-in strategies:

```bash
python examples/run_backtest.py
```

### API

Start the FastAPI REST API:

```bash
uvicorn api.main:app --reload --port 8000
```

## Testing

Run the test suite using `pytest`:

```bash
pytest
```
