import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

from data_preprocessing import DataPreprocessor
from rl_environment import MovieRecommendationEnv
from bandit_agent import MultiArmBanditAgent

# Create a Thompson Sampling agent (as an example of a different algorithm)
class ThompsonSamplingAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Success counts
        self.beta = np.ones(n_arms)   # Failure counts
        
    def act(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.n_arms))
            
        if not valid_actions:
            return None
            
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha[valid_actions], self.beta[valid_actions])
        
        # Choose arm with highest sample
        best_action_idx = np.argmax(samples)
        return valid_actions[best_action_idx]
        
    def update(self, action, reward):
        # Update Beta distribution parameters
        if reward > 0:
            self.alpha[action] += reward
        else:
            self.beta[action] += 1
            
    def save(self, filepath):
        state = {
            'alpha': self.alpha,
            'beta': self.beta
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.alpha = state['alpha']
            self.beta = state['beta']

# Function to train an agent and return metrics
def train_agent(agent_type, env, episodes=500):
    if agent_type == 'UCB':
        agent = MultiArmBanditAgent(n_arms=env.n_movies, alpha=0.1, exploration_weight=1.0)
    elif agent_type == 'Thompson':
        agent = ThompsonSamplingAgent(n_arms=env.n_movies)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    rewards_history = []
    coverage_history = []
    exploration_rate_history = []
    diversity_history = []
    
    all_recommended_movies = set()
    
    for e in tqdm(range(episodes), desc=f"Training {agent_type}"):
        user_id = np.random.randint(0, env.n_users)
        state = env.reset(user_id)
        
        total_reward = 0
        episode_recommendations = []
        done = False
        
        while not done:
            valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
            action = agent.act(state, valid_actions)
            episode_recommendations.append(action)
            
            next_state, reward, done, info = env.step(action)
            agent.update(action, reward)
            
            state = next_state
            total_reward += reward
        
        all_recommended_movies.update(episode_recommendations)
        
        coverage = len(all_recommended_movies) / env.n_movies
        
        new_recommendations = len(set(episode_recommendations) - set(list(all_recommended_movies)[:-len(episode_recommendations)]))
        exploration_rate = new_recommendations / len(episode_recommendations) if episode_recommendations else 0
        
        diversity = np.std(episode_recommendations) if episode_recommendations else 0
        
        rewards_history.append(total_reward)
        coverage_history.append(coverage)
        exploration_rate_history.append(exploration_rate)
        diversity_history.append(diversity)
    
    # Save the agent
    os.makedirs(f"models/{agent_type}", exist_ok=True)
    agent.save(f"models/{agent_type}/final_model.pkl")
    
    return {
        'rewards': rewards_history,
        'coverage': coverage_history,
        'exploration': exploration_rate_history,
        'diversity': diversity_history,
        'final_metrics': {
            'reward': np.mean(rewards_history[-100:]),
            'coverage': coverage_history[-1],
            'exploration': np.mean(exploration_rate_history[-100:]),
            'diversity': np.mean(diversity_history[-100:])
        }
    }

# Main function
def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor('Dataset/ratings.csv', 'Dataset/movies.csv')
    preprocessor.load_data()
    train_data, test_data, user_item_matrix = preprocessor.preprocess()
    movie_features = preprocessor.get_movie_features()
    
    # Create the environment
    env = MovieRecommendationEnv(user_item_matrix, train_data, movie_features)
    
    # Train different agents
    agent_types = ['UCB', 'Thompson']
    results = {}
    
    for agent_type in agent_types:
        print(f"\nTraining {agent_type} agent...")
        results[agent_type] = train_agent(agent_type, env)
    
    # Create comparative visualizations
    create_comparative_plots(results, agent_types)

def create_comparative_plots(results, agent_types):
    # Create a dashboard with comparative plots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Rewards Comparison
    plt.subplot(3, 2, 1)
    for agent_type in agent_types:
        plt.plot(results[agent_type]['rewards'], label=agent_type)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    
    # Plot 2: Coverage Comparison
    plt.subplot(3, 2, 2)
    for agent_type in agent_types:
        plt.plot(results[agent_type]['coverage'], label=agent_type)
    plt.xlabel('Episode')
    plt.ylabel('Coverage')
    plt.title('Movie Coverage')
    plt.legend()
    
    # Plot 3: Exploration Comparison
    plt.subplot(3, 2, 3)
    for agent_type in agent_types:
        plt.plot(results[agent_type]['exploration'], label=agent_type)
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate')
    plt.title('Exploration Efficiency')
    plt.legend()
    
    # Plot 4: Diversity Comparison
    plt.subplot(3, 2, 4)
    for agent_type in agent_types:
        plt.plot(results[agent_type]['diversity'], label=agent_type)
    plt.xlabel('Episode')
    plt.ylabel('Diversity')
    plt.title('Recommendation Diversity')
    plt.legend()
    
    # Plot 5: Final Metrics Comparison (Bar Chart)
    plt.subplot(3, 2, 5)
    metrics = ['reward', 'coverage', 'exploration', 'diversity']
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, agent_type in enumerate(agent_types):
        values = [results[agent_type]['final_metrics'][metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=agent_type)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Final Performance Metrics')
    plt.xticks(x + width/2, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/comparative_analysis.png')
    plt.close()
    
    print("Comparative analysis saved to results/comparative_analysis.png")

if __name__ == "__main__":
    main()