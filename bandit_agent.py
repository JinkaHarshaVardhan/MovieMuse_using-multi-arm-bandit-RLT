import numpy as np
import random
import pickle
import os

class MultiArmBanditAgent:
    """
    Multi-Arm Bandit agent for movie recommendations.
    
    Uses Upper Confidence Bound (UCB) algorithm to balance exploration and exploitation.
    """
    
    def __init__(self, n_arms, alpha=0.1, exploration_weight=1.0):
        """
        Initialize the Multi-Arm Bandit agent.
        
        Args:
            n_arms: Number of arms (movies)
            alpha: Learning rate for updating value estimates
            exploration_weight: Weight for the exploration term in UCB
        """
        self.n_arms = n_arms
        self.alpha = alpha
        self.exploration_weight = exploration_weight
        
        # Initialize value estimates for each arm
        self.value_estimates = np.zeros(n_arms)
        
        # Count of times each arm was pulled
        self.arm_counts = np.zeros(n_arms)
        
        # Total number of actions taken
        self.total_actions = 0
        
    def act(self, state, valid_actions=None):
        """
        Choose an action using Upper Confidence Bound (UCB) algorithm.
        
        Args:
            state: Current state (not used in MAB, but kept for API compatibility)
            valid_actions: List of valid actions to choose from
            
        Returns:
            Selected action (movie index)
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_arms))
            
        if not valid_actions:
            return None
            
        # Increment total actions
        self.total_actions += 1
        
        # Calculate UCB values for valid actions
        ucb_values = np.zeros(len(valid_actions))
        
        for i, action in enumerate(valid_actions):
            # If arm never pulled, prioritize it
            if self.arm_counts[action] == 0:
                return action
                
            # Calculate UCB value
            exploitation = self.value_estimates[action]
            exploration = self.exploration_weight * np.sqrt(
                np.log(self.total_actions) / self.arm_counts[action]
            )
            ucb_values[i] = exploitation + exploration
            
        # Choose action with highest UCB value
        best_action_idx = np.argmax(ucb_values)
        return valid_actions[best_action_idx]
        
    def update(self, action, reward):
        """
        Update value estimates based on observed reward.
        
        Args:
            action: The action (arm) that was taken
            reward: The reward received
        """
        # Increment the count for this arm
        self.arm_counts[action] += 1
        
        # Update value estimate using incremental average
        self.value_estimates[action] += self.alpha * (reward - self.value_estimates[action])
    
    def save(self, filepath):
        """Save the agent's state to a file"""
        state = {
            'value_estimates': self.value_estimates,
            'arm_counts': self.arm_counts,
            'total_actions': self.total_actions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, filepath):
        """Load the agent's state from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            self.value_estimates = state['value_estimates']
            self.arm_counts = state['arm_counts']
            self.total_actions = state['total_actions']