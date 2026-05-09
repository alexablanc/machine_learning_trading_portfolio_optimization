from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
import os
import csv
import json

# Add plugins path to sys.path so we can import our modules
import sys
sys.path.insert(0, '/usr/local/airflow/plugins')

# Import our trading pipeline modules
from trading_pipeline.indicators import add_indicators
from trading_pipeline.llm_signal import get_llm_signal
from trading_pipeline.agent import discretize_state, get_q_action, update_q_learner
from trading_pipeline.performance import compute_metrics

# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True, # We want to train the Q-learner sequentially
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# The DAG
with DAG(
    'llm_qlearner_trading_pipeline',
    default_args=default_args,
    description='A daily trading pipeline using LLM signal generation and Q-Learner position sizing',
    schedule_interval='0 18 * * 1-5', # Run at 6 PM UTC Monday-Friday (after market close)
    catchup=False,
    tags=['trading', 'ml4t'],
) as dag:

    def fetch_market_data(**kwargs):
        """Fetch the latest JPM data and compute indicators."""
        symbol = 'JPM'
        # We fetch 60 days to ensure we have enough history for 20-day BB%B and 14-day RSI
        df = yf.download(symbol, period='60d', interval='1d')
        
        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = add_indicators(df)
        
        # Get the latest day's data
        latest_data = df.iloc[-1].to_dict()
        latest_date = df.index[-1].strftime('%Y-%m-%d')
        
        # Also pass yesterday's close to calculate today's return for the Q-Learner reward
        yesterday_close = df['Close'].iloc[-2] if len(df) > 1 else latest_data['Close']
        
        # Push to XCom for the next tasks
        kwargs['ti'].xcom_push(key='latest_data', value=latest_data)
        kwargs['ti'].xcom_push(key='latest_date', value=latest_date)
        kwargs['ti'].xcom_push(key='yesterday_close', value=yesterday_close)
        
        print(f"Fetched data for {latest_date}. Close: {latest_data['Close']}")

    def generate_llm_signal(**kwargs):
        """Ask the LLM for a directional signal based on the latest indicators."""
        ti = kwargs['ti']
        latest_data = ti.xcom_pull(task_ids='fetch_market_data', key='latest_data')
        
        # Extract relevant indicators
        bb = latest_data.get('BB_PB', 0.5)
        rsi = latest_data.get('RSI', 50)
        mom = latest_data.get('Momentum', 0)
        
        # Get signal from LLM (0=Bearish, 1=Neutral, 2=Bullish)
        signal = get_llm_signal(bb, rsi, mom)
        
        ti.xcom_push(key='llm_signal', value=signal)
        print(f"LLM generated signal: {signal}")

    def execute_q_learner(**kwargs):
        """Discretize state, update Q-Learner with yesterday's reward, and get today's action."""
        ti = kwargs['ti']
        latest_data = ti.xcom_pull(task_ids='fetch_market_data', key='latest_data')
        yesterday_close = ti.xcom_pull(task_ids='fetch_market_data', key='yesterday_close')
        latest_date = ti.xcom_pull(task_ids='fetch_market_data', key='latest_date')
        llm_signal = ti.xcom_pull(task_ids='generate_llm_signal', key='llm_signal')
        
        # Extract indicators
        bb = latest_data.get('BB_PB', 0.5)
        rsi = latest_data.get('RSI', 50)
        mom = latest_data.get('Momentum', 0)
        today_close = latest_data.get('Close', 100)
        
        # 1. Calculate reward for yesterday's action (if any)
        # We store the previous state and action in a local JSON file to persist across DAG runs
        state_file = '/tmp/qlearner_state.json'
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                prev_state_data = json.load(f)
                
            prev_action = prev_state_data.get('action', 1) # Default to Cash (1)
            
            # Calculate daily return
            daily_return = (today_close / yesterday_close) - 1
            
            # Reward: 
            # Action 0 (Short): -daily_return
            # Action 1 (Cash): 0
            # Action 2 (Long): +daily_return
            if prev_action == 0:
                reward = -daily_return
            elif prev_action == 2:
                reward = daily_return
            else:
                reward = 0
                
            # Update the Q-Learner with the reward
            update_q_learner(reward)
            print(f"Updated Q-Learner with reward: {reward:.4f} for previous action: {prev_action}")
        
        # 2. Get today's state and action
        state = discretize_state(bb, rsi, mom, llm_signal)
        action = get_q_action(state)
        
        # 3. Save current state and action for tomorrow's reward calculation
        with open(state_file, 'w') as f:
            json.dump({'state': state, 'action': action, 'date': latest_date}, f)
            
        # 4. Map action to trade size
        # 0 = Short (-1000), 1 = Cash (0), 2 = Long (+1000)
        position = (action - 1) * 1000
        
        ti.xcom_push(key='target_position', value=position)
        ti.xcom_push(key='close_price', value=today_close)
        
        print(f"Current State: {state} -> Action: {action} -> Target Position: {position}")

    def log_trade_and_metrics(**kwargs):
        """Save the trade to a CSV and compute performance metrics."""
        ti = kwargs['ti']
        latest_date = ti.xcom_pull(task_ids='fetch_market_data', key='latest_date')
        target_position = ti.xcom_pull(task_ids='execute_q_learner', key='target_position')
        close_price = ti.xcom_pull(task_ids='execute_q_learner', key='close_price')
        
        trades_file = '/tmp/trades_log.csv'
        
        # 1. Determine the actual order needed to reach the target position
        current_position = 0
        if os.path.exists(trades_file):
            df_trades = pd.read_csv(trades_file)
            if not df_trades.empty:
                # Sum all past orders to get current position
                current_position = df_trades['Order'].sum()
                
        order = target_position - current_position
        
        # 2. Log the trade if non-zero
        if order != 0 or not os.path.exists(trades_file):
            file_exists = os.path.exists(trades_file)
            with open(trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Date', 'Symbol', 'Order', 'Price', 'Target_Position'])
                writer.writerow([latest_date, 'JPM', order, close_price, target_position])
                
            print(f"Logged trade: {order} shares at {close_price}")
        else:
            print(f"Holding current position: {current_position}")
            
        # 3. Compute metrics
        if os.path.exists(trades_file):
            metrics = compute_metrics(trades_file)
            print(f"Current Performance: {metrics}")
            
            # Save metrics
            with open('/tmp/metrics.json', 'w') as f:
                json.dump(metrics, f)

    # Define tasks
    t1 = PythonOperator(
        task_id='fetch_market_data',
        python_callable=fetch_market_data,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id='generate_llm_signal',
        python_callable=generate_llm_signal,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id='execute_q_learner',
        python_callable=execute_q_learner,
        dag=dag,
    )

    t4 = PythonOperator(
        task_id='log_trade_and_metrics',
        python_callable=log_trade_and_metrics,
        dag=dag,
    )

    # Set dependencies
    t1 >> t2 >> t3 >> t4
