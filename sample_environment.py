"""
Sample environment using stable-worldmodel Quick Start.

This script demonstrates the three stages of world model research:
1. Collecting data
2. Training a world model
3. Evaluating with model-predictive control

Usage:
    pip install 'stable-worldmodel[all]'
    python sample_environment.py
"""

import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver


def collect_data():
    """Stage 1: Collect a dataset using an expert policy."""
    # Create a world with the PushT environment
    world = swm.World("swm/PushT-v1", num_envs=8)

    # Set an expert policy (replace with your own policy)
    # For demonstration, we use a random policy
    # world.set_policy(your_expert_policy)

    # Collect demonstration data
    world.collect("data/pusht_demo.lance", episodes=100, seed=0)
    print("Data collection complete: data/pusht_demo.lance")


def load_and_train():
    """Stage 2: Load the dataset and train a world model."""
    # Load the collected dataset (format is autodetected)
    dataset = swm.data.load_dataset("data/pusht_demo.lance", num_steps=16)

    # Train your world model here
    # world_model = ...  # your model implementation

    print("Dataset loaded. Implement your world model training here.")


def evaluate():
    """Stage 3: Evaluate with model-predictive control."""
    # Create a solver (requires a trained world model)
    # solver = CEMSolver(model=world_model, num_samples=300)

    # Create a policy using the solver
    # policy = WorldModelPolicy(
    #     solver=solver,
    #     config=PlanConfig(horizon=10)
    # )

    # Set the policy and evaluate
    # world.set_policy(policy)
    # results = world.evaluate(episodes=50)
    # print(f"Success Rate: {results['success_rate']:.1f}%")

    print("Evaluation ready. Implement your world model and solver.")


if __name__ == "__main__":
    print("=== stable-worldmodel Sample Environment ===\n")
    print("This sample demonstrates the Quick Start workflow.")
    print("Uncomment and implement the stages as needed.\n")

    # Stage 1: Collect data
    # collect_data()

    # Stage 2: Load and train
    # load_and_train()

    # Stage 3: Evaluate
    # evaluate()

    print("Sample environment created successfully!")
    print("\nTo run the full pipeline:")
    print("1. Uncomment the function calls in __main__")
    print("2. Implement your expert policy and world model")
    print("3. Run: python sample_environment.py")
