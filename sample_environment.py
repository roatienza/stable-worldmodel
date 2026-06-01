"""
Sample environment using stable-worldmodel Quick Start.

This script demonstrates the three stages of world model research:
1. Collecting data with an expert policy
2. Training a world model (LeWM-style JEPA architecture)
3. Evaluating with model-predictive control (CEM solver)

Usage:
    pip install 'stable-worldmodel[all]'
    python sample_environment.py

Each stage can be run independently. Uncomment the desired stage in main().
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig, BasePolicy
from stable_worldmodel.solver import CEMSolver


# =============================================================================
# STAGE 1: Expert Policy for Data Collection
# =============================================================================

class PushTExpertPolicy(BasePolicy):
    """
    A simple heuristic expert policy for PushT-v1.

    PushT requires pushing a T-shaped object to a goal position and orientation.
    This policy uses a greedy approach: move the pusher toward the goal while
    considering the object's current position.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "pusht_expert"

    def get_action(self, obs, **kwargs):
        """
        Get action from the expert policy.

        For PushT, the action is a 2D vector (dx, dy) representing the
        pusher movement. We use a simple heuristic: move toward the goal
        while avoiding obstacles.

        Args:
            obs: Observation dictionary from the environment.
                 Contains 'pixels' (image) and potentially 'goal' info.
            **kwargs: Additional parameters.

        Returns:
            Action as a numpy array of shape (action_dim,).
        """
        # For demonstration, we use a random policy as a baseline.
        # In practice, you would implement a learned or heuristic policy.
        # The PushT environment has a 2D continuous action space.
        if hasattr(self, "env") and self.env is not None:
            return self.env.action_space.sample()

        # Fallback: return a zero action
        return np.array([0.0, 0.0], dtype=np.float32)


class RandomExpertPolicy(BasePolicy):
    """Simple random policy for baseline data collection."""

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "random"
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def get_action(self, obs, **kwargs):
        if hasattr(self, "env") and self.env is not None:
            return self.env.action_space.sample()
        return np.array([0.0, 0.0], dtype=np.float32)


# =============================================================================
# STAGE 2: World Model (LeWM-style JEPA Architecture)
# =============================================================================

class ActionEncoder(nn.Module):
    """Encodes action sequences into embeddings."""

    def __init__(self, input_dim=2, emb_dim=768, mlp_scale=4):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) action sequences
        Returns:
            (B, T, emb_dim) action embeddings
        """
        return self.mlp(x.float())


class SimpleImageEncoder(nn.Module):
    """
    A simple CNN-based image encoder for world model.

    This is a lightweight encoder suitable for demonstration purposes.
    For production use, consider using a pretrained ViT or ResNet.
    """

    def __init__(self, input_channels=3, img_size=64, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Projection to embedding space
        self.projector = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pixels):
        """
        Args:
            pixels: (B, T, C, H, W) or (B, C, H, W)
        Returns:
            embeddings: (B, T, embed_dim) or (B, embed_dim)
        """
        if pixels.ndim == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            b, t = pixels.shape[:2]
            pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
            out = self.backbone(pixels)
            out = self.projector(out.squeeze(-1).squeeze(-1))
            return rearrange(out, "(b t) d -> b t d", b=b, t=t)
        else:
            out = self.backbone(pixels)
            return self.projector(out.squeeze(-1).squeeze(-1))


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for next-state embedding prediction.

    Uses causal attention to predict future states from past context.
    """

    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4,
        num_frames=3,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action conditioning
        self.action_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, context_emb, action_emb):
        """
        Args:
            context_emb: (B, T, D) past state embeddings
            action_emb: (B, T, D) action embeddings (conditioning)
        Returns:
            (B, 1, D) predicted next state embedding
        """
        # Add positional encoding
        T = context_emb.size(1)
        x = context_emb + self.pos_embedding[:, :T]

        # Combine with action conditioning
        x = x + self.action_proj(action_emb)

        # Transformer forward pass
        x = self.transformer(x)

        # Predict next state (use last timestep)
        pred = self.output_proj(x[:, -1:, :])
        return pred


class SimpleWorldModel(nn.Module):
    """
    A simplified LeWM-style world model for demonstration.

    Architecture:
    1. Image Encoder: CNN that encodes observations into embeddings
    2. Action Encoder: MLP that encodes actions into embeddings
    3. Predictor: Transformer that predicts next-state embeddings

    This model follows the JEPA (Joint Embedding Predictive Architecture)
    paradigm where predictions are made in latent space.
    """

    def __init__(
        self,
        input_channels=3,
        img_size=64,
        embed_dim=768,
        action_dim=2,
        num_frames=3,
        predictor_heads=8,
        predictor_layers=4,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_frames = num_frames

        # Components
        self.encoder = SimpleImageEncoder(input_channels, img_size, embed_dim)
        self.action_encoder = ActionEncoder(action_dim, embed_dim)
        self.predictor = TransformerPredictor(
            embed_dim=embed_dim,
            num_heads=predictor_heads,
            num_layers=predictor_layers,
            num_frames=num_frames,
        )

    def encode(self, info):
        """
        Encode observations and actions into embeddings.

        Args:
            info: dict with 'pixels' and optionally 'action' keys
        Returns:
            dict with 'emb' (state embeddings) and 'act_emb' (action embeddings)
        """
        pixels = info["pixels"].float()
        emb = self.encoder(pixels)
        info["emb"] = emb

        if "action" in info:
            act_emb = self.action_encoder(info["action"])
            info["act_emb"] = act_emb

        return info

    def predict(self, context_emb, action_emb):
        """
        Predict next state embedding from context and actions.

        Args:
            context_emb: (B, T, D) past state embeddings
            action_emb: (B, T, D) action embeddings
        Returns:
            (B, 1, D) predicted next state embedding
        """
        return self.predictor(context_emb, action_emb)

    def forward(self, info):
        """Full forward pass: encode + predict."""
        info = self.encode(info)
        if "act_emb" in info:
            ctx_len = self.num_frames
            context = info["emb"][:, :ctx_len]
            actions = info["act_emb"][:, :ctx_len]
            pred = self.predict(context, actions)
            info["pred_emb"] = pred
        return info

    def rollout(self, info, action_sequence, history_size=3):
        """
        Autoregressive rollout for planning.

        Args:
            info: dict with initial state info
            action_sequence: (B, S, T, action_dim) planned actions
                            B=batch, S=samples, T=horizon
            history_size: number of past frames for context
        Returns:
            dict with 'predicted_emb' containing rollout embeddings
        """
        assert "pixels" in info, "pixels not in info"

        H = history_size
        B, S, T = action_sequence.shape[:3]

        # Split actions into initial context and future
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # Encode initial state
        if "emb" not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info["emb"] = _init["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)

        # Flatten batch and sample dims for rollout
        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_flat = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")

        # Encode all actions
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )

        # Autoregressive rollout
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, H + t - H)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : H + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def criterion(self, info_dict):
        """
        Compute cost between predicted and goal embeddings.

        Args:
            info_dict: dict with 'predicted_emb' and 'goal_emb'
        Returns:
            (B, S) cost per sample
        """
        pred_emb = info_dict["predicted_emb"]  # (B, S, T, D)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, D)

        # MSE between predicted last state and goal
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)
        return cost

    def get_cost(self, info_dict, action_candidates):
        """
        Compute cost for action candidates (for CEM solver).

        Args:
            info_dict: dict with initial state and goal info
            action_candidates: (B, S, T, action_dim) candidate action sequences
        Returns:
            (B, S) cost per candidate
        """
        assert "goal" in info_dict, "goal not in info_dict"

        # Encode goal state
        if "goal_emb" not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            goal["pixels"] = goal["goal"]
            for k in list(goal.keys()):
                if k.startswith("goal_"):
                    goal[k[len("goal_") :]] = goal.pop(k)
            goal.pop("action", None)
            goal = self.encode(goal)
            info_dict["goal_emb"] = goal["emb"]

        # Rollout with candidate actions
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_world_model(world_model, dataset, num_epochs=10, batch_size=32, lr=1e-4):
    """
    Train a world model using JEPA-style prediction loss.

    Args:
        world_model: The world model to train
        dataset: Dataset loaded via swm.data.load_dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Trained world model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Training world model on {device} for {num_epochs} epochs...")
    print(f"Dataset size: {len(dataset)}, Batch size: {batch_size}")

    world_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Prepare batch
            pixels = batch["pixels"].to(device)  # (B, T, C, H, W)
            actions = batch["action"].to(device)  # (B, T, action_dim)

            # Handle NaN actions (sequence boundaries)
            actions = torch.nan_to_num(actions, 0.0)

            # Forward pass
            info = {"pixels": pixels, "action": actions}
            info = world_model.encode(info)

            # JEPA prediction: predict next state from context
            ctx_len = world_model.num_frames
            n_preds = 1  # predict 1 step ahead

            context_emb = info["emb"][:, :ctx_len]
            action_emb = info["act_emb"][:, :ctx_len]
            target_emb = info["emb"][:, n_preds:]  # ground truth

            pred_emb = world_model.predict(context_emb, action_emb)

            # Prediction loss (MSE in latent space)
            pred_loss = F.mse_loss(pred_emb, target_emb)

            # Regularization: prevent collapse
            emb = info["emb"]
            emb = emb - emb.mean(dim=-1, keepdim=True)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
            reg_loss = (emb @ emb.transpose(-2, -1)).pow(2).mean()

            loss = pred_loss + 0.1 * reg_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    world_model.eval()
    print("Training complete!")
    return world_model


# =============================================================================
# STAGE 3: Evaluation with Model-Predictive Control
# =============================================================================

def evaluate_with_mpc(world_model, num_eval_episodes=10, horizon=10, num_samples=200):
    """
    Evaluate the world model using model-predictive control with CEM solver.

    Args:
        world_model: Trained world model
        num_eval_episodes: Number of evaluation episodes
        horizon: Planning horizon
        num_samples: Number of CEM samples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)

    # Create environment
    world = swm.World("swm/PushT-v1", num_envs=1)

    # Create CEM solver
    solver = CEMSolver(
        model=world_model,
        num_samples=num_samples,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device=device,
    )

    # Create MPC policy
    policy = WorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=horizon,
            receding_horizon=1,
            history_len=1,
            action_block=1,
            warm_start=True,
        ),
    )

    # Set policy and evaluate
    world.set_policy(policy)
    results = world.evaluate(episodes=num_eval_episodes)

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_eval_episodes} episodes):")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    if "mean_reward" in results:
        print(f"  Mean Reward: {results['mean_reward']:.2f}")
    if "max_reward" in results:
        print(f"  Max Reward: {results['max_reward']:.2f}")
    print(f"{'='*50}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def collect_data(output_path="data/pusht_demo.lance", num_episodes=10):
    """Stage 1: Collect demonstration data."""
    print("\n" + "=" * 50)
    print("STAGE 1: Data Collection")
    print("=" * 50)

    # Create world with PushT environment
    world = swm.World("swm/PushT-v1", num_envs=4)

    # Set expert policy (using random policy for demonstration)
    # Replace with PushTExpertPolicy() for a heuristic policy
    expert_policy = RandomExpertPolicy(seed=42)
    world.set_policy(expert_policy)

    # Collect data
    os.makedirs("data", exist_ok=True)
    world.collect(output_path, episodes=num_episodes, seed=42)
    print(f"Data collection complete: {output_path}")
    print(f"Collected {num_episodes} episodes")

    return output_path


def load_and_train(input_path="data/pusht_demo.lance", num_epochs=5):
    """Stage 2: Load dataset and train world model."""
    print("\n" + "=" * 50)
    print("STAGE 2: World Model Training")
    print("=" * 50)

    # Load dataset
    dataset = swm.data.load_dataset(input_path, num_steps=16)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create world model
    world_model = SimpleWorldModel(
        input_channels=3,
        img_size=64,
        embed_dim=768,
        action_dim=2,
        num_frames=3,
        predictor_heads=8,
        predictor_layers=4,
    )

    # Train
    world_model = train_world_model(
        world_model,
        dataset,
        num_epochs=num_epochs,
        batch_size=32,
        lr=1e-4,
    )

    return world_model


def evaluate(world_model, num_episodes=10):
    """Stage 3: Evaluate with MPC."""
    print("\n" + "=" * 50)
    print("STAGE 3: Evaluation with MPC")
    print("=" * 50)

    results = evaluate_with_mpc(
        world_model,
        num_eval_episodes=num_episodes,
        horizon=10,
        num_samples=200,
    )
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("stable-worldmodel Sample Environment")
    print("=" * 50)
    print("\nThis sample demonstrates the full world model pipeline:")
    print("  1. Data collection with expert policy")
    print("  2. World model training (JEPA-style)")
    print("  3. Evaluation with model-predictive control")
    print("\nUncomment the stages you want to run below.")

    # Stage 1: Collect data
    # Uncomment to collect demonstration data
    # data_path = collect_data(num_episodes=10)

    # Stage 2: Train world model
    # Uncomment after collecting data
    # world_model = load_and_train(input_path=data_path, num_epochs=5)

    # Stage 3: Evaluate with MPC
    # Uncomment after training
    # evaluate(world_model, num_episodes=10)

    print("\n" + "=" * 50)
    print("Sample environment ready!")
    print("=" * 50)
    print("\nTo run the full pipeline:")
    print("  1. Uncomment the function calls above")
    print("  2. Run: python sample_environment.py")
    print("\nFor a quick test with random data collection:")
    print("  python -c \"from sample_environment import collect_data; collect_data(num_episodes=5)\"")
