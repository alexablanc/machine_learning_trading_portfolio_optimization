# LLM + Q-Learner Daily Trading Pipeline (Airflow / Astronomer)

This project implements a daily trading pipeline using Apache Airflow, combining an LLM for signal generation with a Dyna-Q Learner for position sizing.

---

## Architecture

```
fetch_market_data
      |
generate_llm_signal
      |
execute_q_learner
      |
log_trade_and_metrics
```

| Task | Description |
|---|---|
| `fetch_market_data` | Downloads 60 days of JPM OHLCV data via `yfinance` and computes BB %B, RSI, and Momentum |
| `generate_llm_signal` | Sends the indicator values to GPT-4o-mini and receives a directional signal: 0=Bearish, 1=Neutral, 2=Bullish |
| `execute_q_learner` | Discretizes the indicator + LLM state into one of 3,000 states, updates the Q-table with yesterday's reward (Dyna-Q with 200 hallucinations), and queries for today's action |
| `log_trade_and_metrics` | Writes the trade to `/tmp/trades_log.csv` and logs cumulative return, Sharpe ratio, and mean daily return to `/tmp/metrics.json` |

---

## File Structure

```
airflow_project/
├── dags/
│   └── llm_qlearner_trading_dag.py       # Main DAG definition
├── plugins/
│   └── trading_pipeline/
│       ├── QLearner.py                   # Your Dyna-Q implementation
│       ├── indicators.py                 # BB%B, RSI, Momentum (yfinance-compatible)
│       ├── llm_signal.py                 # OpenAI API signal generation
│       ├── agent.py                      # State discretization + Q-Learner wrapper
│       └── performance.py               # Cumulative return, Sharpe ratio
└── README.md
```

---

## Setup Instructions

### 1. Add to your Astronomer project

Copy the `dags/` and `plugins/` folders into your existing Astronomer project directory (the one you used for your exoplanet pipeline).

```bash
cp -r dags/* ~/your-astronomer-project/dags/
cp -r plugins/* ~/your-astronomer-project/plugins/
```

### 2. Add the OpenAI API key as an Airflow Variable or Environment Variable

In the Astronomer UI, go to **Admin → Variables** and add:

| Key | Value |
|---|---|
| `OPENAI_API_KEY` | `your-openai-api-key` |

Or add it to your `.env` file in the Astronomer project:

```
OPENAI_API_KEY=your-openai-api-key
```

### 3. Add Python dependencies

Add the following to your `requirements.txt` in the Astronomer project:

```
yfinance
openai
```

### 4. Start the local environment

```bash
astro dev start
```

### 5. Trigger the DAG

In the Airflow UI at `http://localhost:8080`, find `llm_qlearner_trading_pipeline` and toggle it ON. It will run automatically at 6 PM UTC on weekdays (after US market close).

To trigger it manually for testing, click the **Trigger DAG** button.

---

## How the Q-Learner Works in This Context

The Q-Learner is your exact Dyna-Q implementation from the ML4T course, adapted for a daily Airflow loop:

- **State space (3,000 states):** Combines 10 bins for BB %B + 10 bins for RSI + 10 bins for Momentum + 3 LLM signal values.
- **Actions (3):** 0 = Short (-1000 shares), 1 = Cash (0 shares), 2 = Long (+1000 shares).
- **Reward:** The daily return of JPM, signed by the position held: Long gets +return, Short gets -return, Cash gets 0.
- **Dyna-Q (200 hallucinations):** Since real market data arrives only once per day, Dyna-Q uses the learned transition model to simulate 200 additional updates per real step, dramatically accelerating learning.
- **Persistence:** The Q-table is saved to `/tmp/qlearner_model.pkl` after each run so it accumulates learning across days.

---

## Output Files

| File | Description |
|---|---|
| `/tmp/trades_log.csv` | Cumulative log of all trades: Date, Symbol, Order, Price, Target_Position |
| `/tmp/metrics.json` | Latest performance metrics: Cumulative Return, Sharpe Ratio, Mean Daily Return |
| `/tmp/qlearner_model.pkl` | Serialized Q-Learner model (persists across daily runs) |
| `/tmp/qlearner_state.json` | Previous day's state and action (used to compute reward) |

---

## Notes

- The pipeline trades **JPM only** to match your course project. Change the `symbol` variable in the DAG to trade any ticker supported by `yfinance`.
- The Q-Learner starts with `rar=0.5` (50% random actions) and decays at `radr=0.99` per step. It will take several weeks of daily runs to converge to a stable policy.
- This is a **paper trading** pipeline — it logs intended trades but does not connect to a brokerage. To execute real trades, integrate the Alpaca API in the `log_trade_and_metrics` task.
