#Author - Rayaan Tasnim, prodigious programmer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any

class ActorPolicyNetwork(nn.Module):
    """
    Synthetic policy model (actor network) mapping context embeddings to action logits.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ActorPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln1(self.fc1(x)))
        h = F.gelu(self.ln2(self.fc2(h)))
        logits = self.head(h)
        return logits

class ReferencePolicyNetwork(nn.Module):
    """
    Reference policy network frozen to anchor policy drift during online updates.
    """
    def __init__(self, active_model: ActorPolicyNetwork):
        super(ReferencePolicyNetwork, self).__init__()
        self.model = ActorPolicyNetwork(
            input_dim=active_model.fc1.in_features,
            hidden_dim=active_model.fc1.out_features,
            output_dim=active_model.head.out_features
        )
        self.model.load_state_dict(active_model.state_dict())
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

class StreamingDPOTrainer:
    """
    Online Direct Preference Optimization (DPO) streaming trainer adapted for real-time telemetry.
    """
    def __init__(self, policy: ActorPolicyNetwork, ref_policy: ReferencePolicyNetwork, lr: float = 1e-4, beta: float = 0.1):
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=0.01)

    def compute_dpo_loss(self, 
                         states: torch.Tensor, 
                         chosen_actions: torch.Tensor, 
                         rejected_actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Forward pass policy
        policy_logits = self.policy(states)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        chosen_policy_log_probs = policy_log_probs.gather(1, chosen_actions.unsqueeze(1)).squeeze(1)
        rejected_policy_log_probs = policy_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)

        # Forward pass reference policy
        with torch.no_grad():
            ref_logits = self.ref_policy(states)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            chosen_ref_log_probs = ref_log_probs.gather(1, chosen_actions.unsqueeze(1)).squeeze(1)
            rejected_ref_log_probs = ref_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)

        # Implicit rewards calculation
        pi_logratios = chosen_policy_log_probs - rejected_policy_log_probs
        ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs

        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()

        # Metrics for telemetry tracking
        with torch.no_grad():
            reward_chosen = self.beta * (chosen_policy_log_probs - chosen_ref_log_probs)
            reward_rejected = self.beta * (rejected_policy_log_probs - rejected_ref_log_probs)
            accuracy = (reward_chosen > reward_rejected).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "reward_margin": (reward_chosen - reward_rejected).mean().item(),
            "accuracy": accuracy
        }
        return loss, metrics

    def step(self, states: torch.Tensor, chosen: torch.Tensor, rejected: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        loss, metrics = self.compute_dpo_loss(states, chosen, rejected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        return metrics

class SimulatedHumanEnvironment:
    """
    Simulated human-in-the-loop interaction loop generating telemetry and preference pairs.
    """
    def __init__(self, state_dim: int, output_dim: int, batch_size: int = 32):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

    def generate_streaming_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.randn(self.batch_size, self.state_dim)
        oracle_weights = torch.randn(self.state_dim, self.output_dim)
        scores = torch.matmul(states, oracle_weights)
        
        sorted_actions = torch.argsort(scores, dim=-1)
        chosen_actions = sorted_actions[:, -1]
        rejected_actions = sorted_actions[:, 0]
        
        return states, chosen_actions, rejected_actions

def run_simulation(num_steps: int = 50):
    print("Initializing Real-Time RLHF / RLAIF Simulation Framework...")
    state_dim = 64
    hidden_dim = 128
    output_dim = 10
    batch_size = 32

    actor = ActorPolicyNetwork(state_dim, hidden_dim, output_dim)
    ref_policy = ReferencePolicyNetwork(actor)
    trainer = StreamingDPOTrainer(actor, ref_policy, lr=3e-4, beta=0.1)
    env = SimulatedHumanEnvironment(state_dim, output_dim, batch_size)

    print(f"{'Step':<6} | {'Loss':<10} | {'Reward Margin':<15} | {'Preference Accuracy':<20}")
    print("-" * 60)

    for step in range(1, num_steps + 1):
        states, chosen, rejected = env.generate_streaming_batch()
        metrics = trainer.step(states, chosen, rejected)
        
        if step % 10 == 0 or step == 1:
            print(f"{step:<6} | {metrics['loss']:<10.4f} | {metrics['reward_margin']:<15.4f} | {metrics['accuracy']*100:<19.1f}%")

    print("-" * 60)
    print("Simulation completed successfully. Policy successfully updated via streaming telemetry.")

if __name__ == "__main__":
    run_simulation()