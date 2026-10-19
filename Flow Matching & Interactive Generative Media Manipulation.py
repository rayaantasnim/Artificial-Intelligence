#Author - Rayaan Tasnim, prodigious programmer

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Optional, Dict, Any

class VectorFieldBackbone(nn.Module):
    """
    Neural network parameterizing the time-dependent vector field v_t(x, t).
    Maps state x (and time t) to velocity v.
    """
    def __init__(self, data_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class ConditionalFlowMatching:
    """
    Implements Conditional Flow Matching (CFM) training objective and ODE sampling.
    """
    def __init__(self, velocity_model: nn.Module):
        self.model = velocity_model

    def compute_loss(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, 1, device=x_0.device)
        x_t = (1.0 - t) * x_0 + t * x_1
        target_v = x_1 - x_0
        pred_v = self.model(x_t, t.squeeze(-1))
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample_ode(
        self, 
        x_init: torch.Tensor, 
        num_steps: int = 50, 
        steering_callback: Optional[Callable[[torch.Tensor, float, torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = x_init.device
        dt = 1.0 / num_steps
        x = x_init.clone()
        
        trajectory = [x.detach().cpu()]
        velocity_errors = []

        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((x.shape[0], 1), t_val, device=device)
            
            v = self.model(x, t_tensor.squeeze(-1))
            
            if steering_callback is not None:
                v_steered = steering_callback(x, t_val, v)
                v_error = torch.norm(v_steered - v, dim=-1).mean().item()
                velocity_errors.append(v_error)
                v = v_steered
            else:
                velocity_errors.append(0.0)

            x = x + v * dt
            trajectory.append(x.detach().cpu())

        telemetry = {
            "trajectory": torch.stack(trajectory, dim=0),
            "velocity_errors": velocity_errors
        }
        return x, telemetry


class InteractiveSteeringEngine:
    """
    Applies geometric constraint masking and vector field redirection based on user-defined
    spatial bounding coordinates or steering vectors.
    """
    def __init__(self, target_region_center: torch.Tensor, radius: float = 0.5, strength: float = 2.0):
        self.center = target_region_center
        self.radius = radius
        self.strength = strength

    def __call__(self, x: torch.Tensor, t: float, velocity: torch.Tensor) -> torch.Tensor:
        distance = torch.norm(x - self.center.to(x.device), dim=-1, keepdim=True)
        mask = (distance < self.radius).float()
        guiding_vector = (self.center.to(x.device) - x) * self.strength
        steered_velocity = velocity * (1.0 - mask) + (velocity + guiding_vector) * mask
        return steered_velocity


if __name__ == "__main__":
    dim = 2
    batch_size = 64
    hidden_dim = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone = VectorFieldBackbone(data_dim=dim, hidden_dim=hidden_dim).to(device)
    cfm = ConditionalFlowMatching(backbone)
    optimizer = optim.Adam(backbone.parameters(), lr=1e-3)

    print("Training Flow Matching backbone...")
    backbone.train()
    for epoch in range(100):
        x_0 = torch.randn(batch_size, dim, device=device)
        x_1 = torch.randn(batch_size, dim, device=device) + 2.0
        
        optimizer.zero_grad()
        loss = cfm.compute_loss(x_0, x_1)
        loss.backward()
        optimizer.step()

    print(f"Training completed. Final Loss: {loss.item():.4f}")

    backbone.eval()
    x_init = torch.randn(16, dim, device=device)
    steering_anchor = torch.tensor([1.0, 1.0], device=device)
    steering_engine = InteractiveSteeringEngine(target_region_center=steering_anchor, radius=1.5, strength=3.0)

    print("Running ODE integration with mid-trajectory geometric steering...")
    final_samples, telemetry = cfm.sample_ode(
        x_init=x_init, 
        num_steps=30, 
        steering_callback=steering_engine
    )

    print(f"Generation successful. Output tensor shape: {final_samples.shape}")
    print(f"Telemetry tracked {len(telemetry['velocity_errors'])} integration steps with dynamic vector adjustments.")