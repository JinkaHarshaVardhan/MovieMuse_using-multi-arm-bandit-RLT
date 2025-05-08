import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import seaborn as sns

from data_preprocessing import DataPreprocessor
from rl_environment import MovieRecommendationEnv
from bandit_agent import MultiArmBanditAgent

# Set random seeds for reproducibility
np.random.seed(42)

# Create directories for models and results
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

# Get action dimensions (number of movies)
action_size = env.n_movies

print(f"Number of movies (arms): {action_size}")

# Create the Multi-Arm Bandit agent
agent = MultiArmBanditAgent(n_arms=action_size, alpha=0.1, exploration_weight=1.0)

# Training parameters
episodes = 1000

# Initialize metrics tracking
rewards_history = []
avg_rewards_history = []
coverage_history = []  # Track movie coverage
exploration_rate_history = []  # Track exploration rate
diversity_history = []  # Track recommendation diversity

# Set to track all movies recommended across episodes
all_recommended_movies = set()

print("Starting training...")
for e in tqdm(range(episodes)):
    # Reset the environment
    user_id = np.random.randint(0, env.n_users)
    state = env.reset(user_id)
    
    total_reward = 0
    episode_recommendations = []
    done = False
    
    while not done:
        # Get valid actions (movies not yet recommended)
        valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
        
        # Choose an action
        action = agent.act(state, valid_actions)
        episode_recommendations.append(action)
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Update the agent with the reward
        agent.update(action, reward)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
    
    # Add recommended movies to the overall set
    all_recommended_movies.update(episode_recommendations)
    
    # Calculate coverage (percentage of all movies recommended so far)
    coverage = len(all_recommended_movies) / env.n_movies
    
    # Calculate exploration rate (percentage of new movies in this episode)
    new_recommendations = len(set(episode_recommendations) - set(list(all_recommended_movies)[:-len(episode_recommendations)]))
    exploration_rate = new_recommendations / len(episode_recommendations) if episode_recommendations else 0
    
    # Calculate diversity (using genre diversity if available, otherwise use a simple metric)
    # For simplicity, we'll use the standard deviation of movie indices as a proxy for diversity
    diversity = np.std(episode_recommendations) if episode_recommendations else 0
    
    # Save metrics history
    rewards_history.append(total_reward)
    avg_reward = np.mean(rewards_history[-100:])
    avg_rewards_history.append(avg_reward)
    coverage_history.append(coverage)
    exploration_rate_history.append(exploration_rate)
    diversity_history.append(diversity)
    
    if e % 100 == 0:
        print(f"Episode: {e}, Avg Reward: {avg_reward:.2f}, Coverage: {coverage:.2f}")
        # Save the model
        agent.save(f"models/bandit_model_episode_{e}.pkl")

# Save the final model
agent.save("models/bandit_model_final.pkl")

# Create a comprehensive visualization dashboard
plt.figure(figsize=(15, 10))

# Plot 1: Rewards
plt.subplot(2, 3, 1)
plt.plot(rewards_history, alpha=0.3, color='blue', label='Rewards')
plt.plot(avg_rewards_history, color='red', label='Avg Rewards (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')
plt.legend()

# Plot 2: Coverage
plt.subplot(2, 3, 2)
plt.plot(coverage_history, color='green')
plt.xlabel('Episode')
plt.ylabel('Coverage')
plt.title('Movie Coverage')

# Plot 3: Exploration Rate
plt.subplot(2, 3, 3)
plt.plot(exploration_rate_history, color='purple')
plt.xlabel('Episode')
plt.ylabel('Exploration Rate')
plt.title('Exploration Efficiency')

# Plot 4: Diversity
plt.subplot(2, 3, 4)
plt.plot(diversity_history, color='orange')
plt.xlabel('Episode')
plt.ylabel('Diversity')
plt.title('Recommendation Diversity')

# Plot 5: Final Metrics Bar Chart
plt.subplot(2, 3, 5)
final_metrics = [
    np.mean(rewards_history[-100:]),  # Final avg reward
    coverage_history[-1],  # Final coverage
    np.mean(exploration_rate_history[-100:]),  # Final avg exploration
    np.mean(diversity_history[-100:])  # Final avg diversity
]
labels = ['Reward', 'Coverage', 'Exploration', 'Diversity']
plt.bar(labels, final_metrics, color=['red', 'green', 'purple', 'orange'])
plt.ylabel('Value')
plt.title('Final Performance Metrics')

# Plot 6: Value Distribution
plt.subplot(2, 3, 6)
plt.hist(agent.value_estimates, bins=30, alpha=0.7)
plt.xlabel('Estimated Value')
plt.ylabel('Count')
plt.title('Value Estimates Distribution')

plt.tight_layout()
plt.savefig('results/performance_dashboard.png')

# Also create individual high-resolution plots
# Rewards plot
plt.figure(figsize=(12, 6))
plt.plot(rewards_history, alpha=0.3, color='blue', label='Rewards')
plt.plot(avg_rewards_history, color='red', label='Avg Rewards (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.legend()
plt.savefig('results/training_progress.png')

# Coverage plot
plt.figure(figsize=(12, 6))
plt.plot(coverage_history, color='green')
plt.xlabel('Episode')
plt.ylabel('Coverage')
plt.title('Movie Coverage Over Time')
plt.savefig('results/coverage_progress.png')

# Exploration plot
plt.figure(figsize=(12, 6))
plt.plot(exploration_rate_history, color='purple')
plt.xlabel('Episode')
plt.ylabel('Exploration Rate')
plt.title('Exploration Efficiency Over Time')
plt.savefig('results/exploration_progress.png')

# Diversity plot
plt.figure(figsize=(12, 6))
plt.plot(diversity_history, color='orange')
plt.xlabel('Episode')
plt.ylabel('Diversity')
plt.title('Recommendation Diversity Over Time')
plt.savefig('results/diversity_progress.png')

plt.close('all')

print("Training completed!")
print("Performance dashboard saved to results/performance_dashboard.png")