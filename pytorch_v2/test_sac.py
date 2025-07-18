import os
import pickle
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from sac.py
from sac import Actor, SoftQNetwork, ImageEncoder, load_dataset, extract_action_range
from pytorch_serl.utils.device import get_device


@dataclass
class TestArgs:
    model_path: str = "models/final_model.pt"
    """path to the trained model checkpoint"""
    dataset_path: str = "../dataset/success_demo.pkl"
    """path to the test dataset"""
    num_test_samples: int = 100
    """number of samples to test"""
    visualize: bool = True
    """whether to create visualizations"""
    save_results: bool = True
    """whether to save test results"""


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model parameters
    action_dim = checkpoint["action_dim"]
    action_low = checkpoint["action_low"]
    action_high = checkpoint["action_high"]

    # Initialize actor
    actor = Actor(
        image_dim=256,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    # Load state dict
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    print(f"Loaded model with action_dim={action_dim}")
    print(f"Action range: low={action_low}, high={action_high}")

    return actor, checkpoint


def test_model_on_dataset(actor, dataset, device, num_samples=None):
    """Test the model on dataset samples"""
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    print(f"Testing model on {num_samples} samples...")

    action_errors = []
    predicted_actions = []
    true_actions = []

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            obs = sample["observations"]
            true_action = sample["actions"]

            # Convert to tensor and add batch dimension
            if isinstance(obs, dict) and "image" in obs:
                image = obs["image"]
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).unsqueeze(0).to(device)
                else:
                    image = image.unsqueeze(0).to(device)
                obs_tensor = {"image": image}
            else:
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                else:
                    obs_tensor = obs.unsqueeze(0).to(device)

            # Get predicted action
            predicted_action, _, _ = actor.get_action(obs_tensor)
            predicted_action = predicted_action.cpu().numpy().flatten()

            # Convert true action to numpy if needed
            if hasattr(true_action, "numpy"):
                true_action = true_action.numpy()
            elif isinstance(true_action, list):
                true_action = np.array(true_action)

            # Calculate error
            error = np.mean((predicted_action - true_action) ** 2)
            action_errors.append(error)
            predicted_actions.append(predicted_action)
            true_actions.append(true_action)

    predicted_actions = np.array(predicted_actions)
    true_actions = np.array(true_actions)
    action_errors = np.array(action_errors)

    return {
        "predicted_actions": predicted_actions,
        "true_actions": true_actions,
        "action_errors": action_errors,
        "mean_error": np.mean(action_errors),
        "std_error": np.std(action_errors),
        "median_error": np.median(action_errors),
    }


def visualize_results(results, save_path=None):
    """Create visualizations of test results"""
    predicted_actions = results["predicted_actions"]
    true_actions = results["true_actions"]
    action_errors = results["action_errors"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Action comparison for first few dimensions
    axes[0, 0].scatter(true_actions[:, 0], predicted_actions[:, 0], alpha=0.6)
    axes[0, 0].plot(
        [true_actions[:, 0].min(), true_actions[:, 0].max()],
        [true_actions[:, 0].min(), true_actions[:, 0].max()],
        "r--",
    )
    axes[0, 0].set_xlabel("True Action (dim 0)")
    axes[0, 0].set_ylabel("Predicted Action (dim 0)")
    axes[0, 0].set_title("True vs Predicted Actions (Dimension 0)")
    axes[0, 0].grid(True)

    # Plot 2: Error distribution
    axes[0, 1].hist(action_errors, bins=30, alpha=0.7)
    axes[0, 1].axvline(
        results["mean_error"],
        color="r",
        linestyle="--",
        label=f'Mean: {results["mean_error"]:.4f}',
    )
    axes[0, 1].axvline(
        results["median_error"],
        color="g",
        linestyle="--",
        label=f'Median: {results["median_error"]:.4f}',
    )
    axes[0, 1].set_xlabel("MSE Error")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Action Prediction Errors")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Error over samples
    axes[1, 0].plot(action_errors)
    axes[1, 0].axhline(
        results["mean_error"],
        color="r",
        linestyle="--",
        label=f'Mean: {results["mean_error"]:.4f}',
    )
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("MSE Error")
    axes[1, 0].set_title("Error over Test Samples")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Action dimension comparison (if multi-dimensional)
    if predicted_actions.shape[1] > 1:
        axes[1, 1].scatter(true_actions[:, 1], predicted_actions[:, 1], alpha=0.6)
        axes[1, 1].plot(
            [true_actions[:, 1].min(), true_actions[:, 1].max()],
            [true_actions[:, 1].min(), true_actions[:, 1].max()],
            "r--",
        )
        axes[1, 1].set_xlabel("True Action (dim 1)")
        axes[1, 1].set_ylabel("Predicted Action (dim 1)")
        axes[1, 1].set_title("True vs Predicted Actions (Dimension 1)")
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Single Dimension Action",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
        )
        axes[1, 1].set_title("Action Space Info")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    args = tyro.cli(TestArgs)
    device = get_device()

    print("SAC Model Testing")
    print(f"Args: {vars(args)}")

    # Load model
    actor, checkpoint = load_model(args.model_path, device)

    # Load dataset
    print("Loading test dataset...")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Test model
    results = test_model_on_dataset(actor, dataset, device, args.num_test_samples)

    # Print results
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Number of test samples: {len(results['action_errors'])}")
    print(f"Mean MSE Error: {results['mean_error']:.6f}")
    print(f"Std MSE Error: {results['std_error']:.6f}")
    print(f"Median MSE Error: {results['median_error']:.6f}")
    print(f"Min MSE Error: {np.min(results['action_errors']):.6f}")
    print(f"Max MSE Error: {np.max(results['action_errors']):.6f}")

    # Action space statistics
    print(f"\nAction Statistics:")
    print(f"True action mean: {np.mean(results['true_actions'], axis=0)}")
    print(f"True action std: {np.std(results['true_actions'], axis=0)}")
    print(f"Predicted action mean: {np.mean(results['predicted_actions'], axis=0)}")
    print(f"Predicted action std: {np.std(results['predicted_actions'], axis=0)}")

    # Save results
    if args.save_results:
        results_path = "test_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved detailed results to {results_path}")

    # Visualize results
    if args.visualize:
        visualize_results(
            results, "test_results_visualization.png" if args.save_results else None
        )


if __name__ == "__main__":
    main()
