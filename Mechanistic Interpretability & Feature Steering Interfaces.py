#Author - Rayaan Tasnim, prodigious programmer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, List

class MockMLPBlock(nn.Module):
    """
    Mock Transformer MLP block that processes token embeddings and generates 
    internal hidden activations.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        h = self.activation(self.linear1(x))
        out = self.linear2(h)
        return out

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) designed to reconstruct and map hidden layer activations 
    into sparse, human-interpretable feature vectors with L1 regularization.
    """
    def __init__(self, d_in: int, d_sae: int, l1_coeff: float = 1.0):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
        
        # Encoder weights and bias
        self.W_enc = nn.Parameter(torch.Tensor(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder weights and bias
        self.W_dec = nn.Parameter(torch.Tensor(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)
            
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_in]
        acts = F.relu(torch.matmul(x, self.W_enc) + self.b_enc)
        return acts
        
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        # f: [..., d_sae]
        recon = torch.matmul(f, self.W_dec) + self.b_dec
        return recon
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.encode(x)
        x_reconstructed = self.decode(f)
        
        recon_loss = F.mse_loss(x_reconstructed, x)
        sparsity_loss = self.l1_coeff * torch.sum(torch.abs(f), dim=-1).mean()
        total_loss = recon_loss + sparsity_loss
        
        return {
            "reconstructed": x_reconstructed,
            "latent_features": f,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss
        }

class ActivationSteeringHook:
    """
    Execution hook layer acting as the steering interface. Captures hidden activations mid-forward pass,
    applies scalar multiplier modifications to specific extracted feature indices, and injects tensors back.
    """
    def __init__(self, sae: SparseAutoencoder, feature_indices: List[int], multipliers: List[float]):
        self.sae = sae
        self.feature_indices = feature_indices
        self.multipliers = multipliers
        self.last_unsteered_activation: torch.Tensor | None = None
        self.last_steered_activation: torch.Tensor | None = None
        
    def __call__(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        self.last_unsteered_activation = output.detach().clone()
        
        batch_shape = output.shape
        flat_output = output.reshape(-1, self.sae.d_in)
        
        f = self.sae.encode(flat_output)
        
        f_steered = f.clone()
        for idx, mult in zip(self.feature_indices, self.multipliers):
            if 0 <= idx < self.sae.d_sae:
                f_steered[:, idx] = f_steered[:, idx] * mult
            else:
                raise ValueError(f"Feature index {idx} out of bounds for SAE dimension {self.sae.d_sae}")
                
        steered_flat = self.sae.decode(f_steered)
        steered_output = steered_flat.reshape(batch_shape)
        
        self.last_steered_activation = steered_output.detach().clone()
        return steered_output

def compute_metrics(unsteered: torch.Tensor, steered: torch.Tensor) -> Dict[str, float]:
    """
    Computes Cosine Similarity and Mean Squared Error (MSE) between unsteered and steered activations.
    """
    unsteered_flat = unsteered.reshape(-1)
    steered_flat = steered.reshape(-1)
    
    mse = F.mse_loss(steered_flat, unsteered_flat).item()
    cosine_sim = F.cosine_similarity(unsteered_flat.unsqueeze(0), steered_flat.unsqueeze(0)).item()
    
    return {
        "mse": mse,
        "cosine_similarity": cosine_sim
    }

def main():
    print("Initializing Mechanistic Interpretability & Representation Engineering Pipeline...")
    
    batch_size = 4
    seq_len = 16
    d_model = 64
    d_ff = 256
    d_sae = 128
    l1_coeff = 0.01
    
    mlp = MockMLPBlock(d_model=d_model, d_ff=d_ff)
    sae = SparseAutoencoder(d_in=d_model, d_sae=d_sae, l1_coeff=l1_coeff)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("\n--- Phase 1: Baseline Unsteered Forward Pass ---")
    baseline_out = mlp(x)
    print(f"Baseline Output Shape: {baseline_out.shape}")
    
    print("\n--- Phase 2: Registering Steering Hook & Executing Activation Steering ---")
    feature_indices = [5, 12]
    multipliers = [3.5, 0.2]
    
    steering_hook = ActivationSteeringHook(sae=sae, feature_indices=feature_indices, multipliers=multipliers)
    handle = mlp.linear2.register_forward_hook(steering_hook)
    
    try:
        steered_out = mlp(x)
        print(f"Steered Output Shape: {steered_out.shape}")
    finally:
        handle.remove()
        
    print("\n--- Phase 3: Evaluating Continuous Tracking Metrics ---")
    if steering_hook.last_unsteered_activation is not None and steering_hook.last_steered_activation is not None:
        metrics = compute_metrics(
            steering_hook.last_unsteered_activation,
            steering_hook.last_steered_activation
        )
        print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    else:
        print("Error: Activation hook did not capture activations.")
        
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()