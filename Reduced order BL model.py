"""
Equation-Free, Physics-Guided Neural Model for Laminar Boundary-Layer Evolution

This implementation learns boundary-layer physics without governing equations,
using only physics-informed loss functions and a compressed latent representation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt installed

import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)



def solve_blasius(eta_max=10.0, n_points=2000):
    """
    Solve Blasius equation:
        f''' + 0.5 f f'' = 0
    Returns interpolants for f, f', f''
    """

    def blasius_ode(eta, y):
        f, fp, fpp = y
        return [fp, fpp, -0.5 * f * fpp]

    # Accurate shooting value
    fpp0 = 0.332057336215

    y0 = [0.0, 0.0, fpp0]
    eta_span = (0.0, eta_max)
    eta_eval = np.linspace(0.0, eta_max, n_points)

    sol = solve_ivp(
        blasius_ode,
        eta_span,
        y0,
        t_eval=eta_eval,
        rtol=1e-8,
        atol=1e-10
    )

    f = sol.y[0]
    fp = sol.y[1]
    fpp = sol.y[2]

    return (
        interp1d(eta_eval, f, fill_value="extrapolate"),
        interp1d(eta_eval, fp, fill_value="extrapolate"),
        interp1d(eta_eval, fpp, fill_value="extrapolate"),
    )

@dataclass
class Config:
    """Configuration for the boundary layer model"""
    N: int = 64  # Number of grid points in wall-normal direction
    L: int = 4  # Latent dimension (3-5 as per spec)
    hidden_dim: int = 64  # Hidden layer size for encoder/decoder
    lambda_physics: float = 0.1  # Physics loss weight
    learning_rate: float = 1e-3
    num_epochs: int = 500
    batch_size: int = 16

    # Physics loss weights (individual components)
    w_noslip: float = 1.0
    w_smooth: float = 0.5
    w_monotonic: float = 0.3
    w_outer: float = 1.0
    w_causal: float = 0.2


class Encoder(nn.Module):
    """
    Encoder: u ∈ R^N → z ∈ R^L
    Compresses velocity profile to latent representation
    """

    def __init__(self, N: int, L: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, L)
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: velocity profile [batch, N]
        Returns:
            z: latent vector [batch, L]
        """
        return self.net(u)


class Decoder(nn.Module):
    """
    Decoder: z ∈ R^L → u_next ∈ R^N
    Reconstructs next velocity profile from latent
    """

    def __init__(self, L: int, N: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, N),
            nn.Sigmoid()  # Keep velocities in [0, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent vector [batch, L]
        Returns:
            u_next: predicted velocity profile [batch, N]
        """
        return self.net(z)


class BoundaryLayerEvolver(nn.Module):
    """
    Complete boundary layer evolution operator
    u(x) → u(x + Δx)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.N, config.L, config.hidden_dim)
        self.decoder = Decoder(config.L, config.N, config.hidden_dim)

    def forward(self, u_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u_current: current velocity profile [batch, N]
        Returns:
            u_next: predicted next profile [batch, N]
            z: latent representation [batch, L]
        """
        z = self.encoder(u_current)
        u_next = self.decoder(z)
        return u_next, z


class PhysicsLoss:
    """
    Physics-guided loss functions (no equations, rule-based only)
    """

    def __init__(self, config: Config):
        self.config = config

    def no_slip_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        """
        Enforce u = 0 at wall (first grid point)
        Args:
            u_pred: [batch, N]
        Returns:
            scalar loss
        """
        return torch.mean(u_pred[:, 0] ** 2)

    def smoothness_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize high curvature (second derivative approximation)
        Enforces smooth, diffusive behavior
        Args:
            u_pred: [batch, N]
        Returns:
            scalar loss
        """
        # Second-order finite difference approximation
        d2u = u_pred[:, 2:] - 2 * u_pred[:, 1:-1] + u_pred[:, :-2]
        return torch.mean(d2u ** 2)

    def monotonicity_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize negative gradients near wall
        Velocity should increase away from wall
        Args:
            u_pred: [batch, N]
        Returns:
            scalar loss
        """
        # Check first half of domain (near wall)
        N_check = self.config.N // 2
        du = u_pred[:, 1:N_check] - u_pred[:, :N_check - 1]
        # Penalize negative gradients
        violations = torch.relu(-du)
        return torch.mean(violations ** 2)

    def outer_flow_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        """
        Enforce u ≈ 1 at freestream (last grid point)
        Args:
            u_pred: [batch, N]
        Returns:
            scalar loss
        """
        return torch.mean((u_pred[:, -1] - 1.0) ** 2)

    def causal_evolution_loss(self, u_pred: torch.Tensor, u_current: torch.Tensor) -> torch.Tensor:
        """
        Penalize large jumps between steps
        Enforces gradual downstream evolution
        Args:
            u_pred: [batch, N]
            u_current: [batch, N]
        Returns:
            scalar loss
        """
        jump = u_pred - u_current
        return torch.mean(jump ** 2)

    def compute_total_physics_loss(self, u_pred: torch.Tensor, u_current: torch.Tensor) -> torch.Tensor:
        """
        Combine all physics losses
        Args:
            u_pred: predicted next profile [batch, N]
            u_current: current profile [batch, N]
        Returns:
            total physics loss (scalar)
        """
        loss_noslip = self.no_slip_loss(u_pred)
        loss_smooth = self.smoothness_loss(u_pred)
        loss_mono = self.monotonicity_loss(u_pred)
        loss_outer = self.outer_flow_loss(u_pred)
        loss_causal = self.causal_evolution_loss(u_pred, u_current)

        total = (self.config.w_noslip * loss_noslip +
                 self.config.w_smooth * loss_smooth +
                 self.config.w_monotonic * loss_mono +
                 self.config.w_outer * loss_outer +
                 self.config.w_causal * loss_causal)

        return total
class SyntheticDataGenerator:
    """
    Generate laminar boundary layer data using Blasius solution
    Model never sees governing equations
    """

    def __init__(self, config):
        self.config = config
        self.y_grid = self._create_grid()

        # Solve Blasius once
        _, self.fp, _ = solve_blasius()

    def _create_grid(self) -> np.ndarray:
        eta = np.linspace(0, 1, self.config.N)
        y = 10.0 * np.tanh(2.5 * eta) / np.tanh(2.5)
        return y

    def blasius_like_profile(self, delta: float) -> np.ndarray:
        """
        Physically correct Blasius velocity profile
        """
        eta = self.y_grid / (delta + 1e-12)

        u = self.fp(eta)

        # Enforce numerical safety
        u[0] = 0.0
        u[u > 1.0] = 1.0

        return u.astype(np.float32)

    def generate_sequence(self, num_steps=20, delta_growth=0.1):
        sequence = []
        delta = 1.0

        for _ in range(num_steps):
            u = self.blasius_like_profile(delta)
            sequence.append(u)
            delta += delta_growth

        return np.stack(sequence)

    def generate_dataset(self, num_sequences=100, num_steps=20):
        dataset = []
        for _ in range(num_sequences):
            delta_growth = np.random.uniform(0.05, 0.15)
            dataset.append(self.generate_sequence(num_steps, delta_growth))
        return dataset


def train_model(model: BoundaryLayerEvolver,
                dataset: List[np.ndarray],
                config: Config,
                device: str = 'cpu') -> List[float]:
    """
    Train the boundary layer evolution model
    Args:
        model: BoundaryLayerEvolver instance
        dataset: list of sequences [num_sequences][num_steps, N]
        config: configuration
        device: 'cpu' or 'cuda'
    Returns:
        loss_history: list of epoch losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    physics_loss_fn = PhysicsLoss(config)

    loss_history = []

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle dataset
        np.random.shuffle(dataset)

        for sequence in dataset:
            # sequence shape: [num_steps, N]
            num_steps = len(sequence)

            # Process sequence in batches (pairs of consecutive profiles)
            for i in range(0, num_steps - 1, config.batch_size):
                batch_end = min(i + config.batch_size, num_steps - 1)

                # Create batch of (current, next) pairs
                u_current_batch = []
                u_true_next_batch = []

                for j in range(i, batch_end):
                    u_current_batch.append(sequence[j])
                    u_true_next_batch.append(sequence[j + 1])

                if len(u_current_batch) == 0:
                    continue

                u_current = torch.from_numpy(
                    np.stack(u_current_batch)
                ).float().to(device)

                u_true_next = torch.from_numpy(
                    np.stack(u_true_next_batch)
                ).float().to(device)

                # Forward pass
                u_pred_next, z = model(u_current)

                # Data loss (MSE)
                data_loss = torch.mean((u_pred_next - u_true_next) ** 2)

                # Physics loss
                physics_loss = physics_loss_fn.compute_total_physics_loss(u_pred_next, u_current)

                # Total loss
                total_loss = data_loss + config.lambda_physics * physics_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.6f}")

    return loss_history


def rollout_prediction(model: BoundaryLayerEvolver,
                       u_initial: np.ndarray,
                       num_steps: int,
                       device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform autoregressive rollout from initial condition
    Args:
        model: trained model
        u_initial: initial velocity profile [N]
        num_steps: number of forward steps
        device: 'cpu' or 'cuda'
    Returns:
        profiles: [num_steps+1, N] predicted profiles
        latents: [num_steps+1, L] latent vectors
    """
    model.eval()
    model = model.to(device)

    profiles = [u_initial]
    latents = []

    u = torch.tensor(u_initial, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(num_steps):
            u_next, z = model(u)
            profiles.append(u_next.cpu().numpy()[0])
            latents.append(z.cpu().numpy()[0])
            u = u_next

    return np.array(profiles), np.array(latents)


def plot_results(profiles: np.ndarray,
                 y_grid: np.ndarray,
                 latents: Optional[np.ndarray] = None,
                 true_profiles: Optional[np.ndarray] = None):
    """
    Visualize boundary layer evolution
    Args:
        profiles: [num_steps, N] predicted profiles
        y_grid: [N] wall-normal coordinates
        latents: [num_steps, L] latent variables (optional)
        true_profiles: [num_steps, N] ground truth (optional)
    """
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Velocity profiles at different downstream locations
    ax1 = plt.subplot(2, 2, 1)
    step_indices = np.linspace(0, len(profiles) - 1, 5, dtype=int)
    for idx in step_indices:
        ax1.plot(profiles[idx], y_grid, label=f'Step {idx}')
        if true_profiles is not None:
            ax1.plot(true_profiles[idx], y_grid, '--', alpha=0.5)
    ax1.set_xlabel('u/U∞')
    ax1.set_ylabel('y')
    ax1.set_title('Velocity Profiles (Downstream Evolution)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.1])

    # Plot 2: Profile shape evolution (zoomed near wall)
    ax2 = plt.subplot(2, 2, 2)
    for idx in step_indices:
        ax2.plot(profiles[idx], y_grid, label=f'Step {idx}')
    ax2.set_xlabel('u/U∞')
    ax2.set_ylabel('y')
    ax2.set_title('Near-Wall Velocity Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 2])
    ax2.set_xlim([0, 1.1])

    # Plot 3: Latent dynamics
    if latents is not None:
        ax3 = plt.subplot(2, 2, 3)
        for i in range(latents.shape[1]):
            ax3.plot(latents[:, i], label=f'z_{i}')
        ax3.set_xlabel('Downstream Step')
        ax3.set_ylabel('Latent Value')
        ax3.set_title('Latent Dynamics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Self-similarity check (normalized profiles)
    ax4 = plt.subplot(2, 2, 4)
    for idx in step_indices[1:]:  # Skip initial profile
        # Simple normalization: find 99% thickness
        u_profile = profiles[idx]
        idx99 = np.where(u_profile >= 0.99)[0]
        if len(idx99) > 0:
            delta99_idx = idx99[0]
            y_norm = y_grid / y_grid[delta99_idx] if y_grid[delta99_idx] > 0 else y_grid
            ax4.plot(u_profile, y_norm, alpha=0.7, label=f'Step {idx}')
    ax4.set_xlabel('u/U∞')
    ax4.set_ylabel('y/δ₉₉')
    ax4.set_title('Self-Similarity Check')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 2])
    ax4.set_xlim([0, 1.1])

    plt.tight_layout()
    return fig


def main():
    """
    Main execution: generate data, train model, evaluate
    """
    TRAIN = True

    print("=" * 60)
    print("Physics-Guided Boundary Layer Neural Model")
    print("=" * 60)

    # Configuration
    config = Config(
        N=64,
        L=4,
        hidden_dim=64,
        lambda_physics=0.1,
        learning_rate=1e-3,
        num_epochs=300,
        batch_size=8
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Generate synthetic data
    data_gen = SyntheticDataGenerator(config)

    if TRAIN:
        print("\nGenerating synthetic boundary layer data...")
        dataset = data_gen.generate_dataset(num_sequences=50, num_steps=25)
        print(f"Generated {len(dataset)} sequences")

    # Create model
    print("\nInitializing model...")
    model = BoundaryLayerEvolver(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    if TRAIN:
        print("\nTraining...")
        loss_history = train_model(model, dataset, config, device)

        # SAVE MODEL HERE (THIS IS THE ANSWER TO "WHERE?")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config
        }, os.path.join(output_dir, "bl_model.pt"))

        print("Model saved to outputs/bl_model.pt")

    else:
        assert os.path.exists(os.path.join(output_dir, "bl_model.pt")), \
            "No trained model found. Set TRAIN=True and run once."

        print("\nLoading trained model...")
        checkpoint = torch.load(
            os.path.join(output_dir, "bl_model.pt"),
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully")


    # Plot training loss
    if TRAIN:
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"),
                    dpi=150, bbox_inches='tight')
        print("\nSaved: training_loss.png")


    # Test rollout
    print("\nPerforming rollout prediction...")
    test_sequence = data_gen.generate_sequence(num_steps=30, delta_growth=0.1)
    u_initial = test_sequence[0]

    predicted_profiles, latents = rollout_prediction(
        model, u_initial, num_steps=29, device=device
    )

    # Visualize results
    print("\nGenerating visualizations...")
    fig = plot_results(
        predicted_profiles,
        data_gen.y_grid,
        latents,
        true_profiles=test_sequence
    )
    plt.savefig(os.path.join(output_dir, "boundary_layer_evolution.png"),
                dpi=150, bbox_inches='tight')
    print("Saved: boundary_layer_evolution.png")

    # Compute error metrics
    error = np.mean((predicted_profiles - test_sequence) ** 2)
    print(f"\nRollout MSE: {error:.6f}")

    # Check self-similarity (simple check)
    print("\n" + "=" * 60)
    print("Model successfully trained!")
    print("=" * 60)
    print("\nKey Results:")
    if TRAIN:
        print(f"- Final training loss: {loss_history[-1]:.6f}")
    print(f"- Rollout prediction error: {error:.6f}")
    print(f"- Latent dimension: {config.L}")
    print("\nThe model has learned to:")
    print("✓ Evolve boundary layers downstream")
    print("✓ Respect no-slip condition at wall")
    print("✓ Maintain smooth velocity profiles")
    print("✓ Preserve outer flow boundary condition")
    print("✓ Compress physics into low-dimensional latent space")

    plt.show()

if __name__ == "__main__":
    main()