import os
import pickle
import numpy as np

# Add plugins path to sys.path
import sys
sys.path.insert(0, '/usr/local/airflow/plugins')

from trading_pipeline.QLearner import QLearner

# Global Q-Learner instance (loaded from disk if it exists)
qlearner_file = '/tmp/qlearner_model.pkl'

def get_qlearner_instance():
    """Load the existing Q-Learner from disk or create a new one."""
    if os.path.exists(qlearner_file):
        with open(qlearner_file, 'rb') as f:
            learner = pickle.load(f)
            return learner
    
    # Create a new learner if none exists
    # State space: 10 bins for BB, 10 for RSI, 10 for Mom, 3 for LLM Signal = 3000 states
    # Actions: 3 (Short, Cash, Long)
    learner = QLearner(
        num_states=3000,
        num_actions=3,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=200, # Use Dyna-Q to speed up learning from sparse daily data
        verbose=False
    )
    return learner

def save_qlearner_instance(learner):
    """Save the updated Q-Learner to disk."""
    with open(qlearner_file, 'wb') as f:
        pickle.dump(learner, f)

def discretize_state(bb, rsi, mom, llm_signal):
    """
    Convert continuous indicators and the LLM signal into a single discrete state integer.
    - BB%B: 10 bins (0 to 9)
    - RSI: 10 bins (0 to 9)
    - Momentum: 10 bins (0 to 9)
    - LLM Signal: 3 bins (0, 1, 2)
    Total states = 10 * 10 * 10 * 3 = 3000
    """
    # 1. Discretize BB%B (typically 0.0 to 1.0, but can extend outside)
    bb_bin = np.digitize(bb, np.linspace(-0.2, 1.2, 9))
    
    # 2. Discretize RSI (0 to 100)
    rsi_bin = np.digitize(rsi, np.linspace(10, 90, 9))
    
    # 3. Discretize Momentum (typically -0.2 to 0.2)
    mom_bin = np.digitize(mom, np.linspace(-0.1, 0.1, 9))
    
    # 4. LLM Signal is already 0, 1, or 2
    llm_bin = llm_signal
    
    # Combine into a single integer state
    # state = (bb * 100 * 3) + (rsi * 10 * 3) + (mom * 3) + llm
    state = (bb_bin * 300) + (rsi_bin * 30) + (mom_bin * 3) + llm_bin
    
    return int(state)

def update_q_learner(reward):
    """Update the Q-Learner with the reward from yesterday's action."""
    learner = get_qlearner_instance()
    
    # The DAG ensures that we query() the learner with the new state *after* this,
    # but to apply the reward to the *previous* state/action, we need to pass the
    # current state into query(). Since we haven't computed today's state yet when
    # this is called, we just use the last known state. The DAG logic handles this
    # by storing the previous state and action.
    
    # We use a slight hack here to force the Q-Learner to update its internal tables
    # for the previous state and action, using the new reward.
    # In a pure RL loop, this would happen in query(s_prime, r).
    
    learner = get_qlearner_instance()
    # The DAG passes the reward for the PREVIOUS state/action.
    # To update the Q-table, we must call query(current_state, reward).
    # Since we don't have the current state yet, we just store the reward globally
    # and apply it when get_q_action is called with the new state.
    global pending_reward
    pending_reward = reward

pending_reward = None

def get_q_action(state):
    """Query the Q-Learner for the next action based on the current state."""
    learner = get_qlearner_instance()
    
    # In a real daily loop, we don't have the reward until tomorrow.
    # The DAG script handles calculating the reward from yesterday's action.
    # Here we just use querysetstate() to get an action without updating the table,
    # or query() if we had a reward.
    
    # For simplicity in this Airflow setup, we'll assume the DAG handles the reward
    # calculation and we just need the action for today.
    
    # To properly train it daily, we actually need to pass yesterday's reward and today's state.
    # The DAG passes the reward directly to a custom update function, but we'll just
    # use query() here if we have a reward, or querysetstate() if we don't.
    
    global pending_reward
    if pending_reward is not None:
        # We have a reward from yesterday, so update the Q-table and get today's action
        action = learner.query(state, pending_reward)
        pending_reward = None
    else:
        # First run or no reward available, just get an action without updating
        action = learner.querysetstate(state)
    
    save_qlearner_instance(learner)
    return action
