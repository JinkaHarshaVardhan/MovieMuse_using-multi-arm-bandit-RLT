# Movie Recommendation System using Multi-Arm Bandit

This project implements a movie recommendation system using the Multi-Arm Bandit algorithm. The system learns to recommend movies to users based on their rating history and movie features.

## Dataset

The project uses the following dataset files:
- `movies.csv`: Contains movie information (ID, title, genres)
- `ratings.csv`: Contains user ratings for movies

## Project Structure

- `data_preprocessing.py`: Handles data loading and preprocessing
- `rl_environment.py`: Implements the reinforcement learning environment
- `bandit_agent.py`: Implements the Multi-Arm Bandit agent
- `train.py`: Script for training the model
- `recommend.py`: Script for generating recommendations and evaluating the model
- `main.py`: Main script to run the project

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```