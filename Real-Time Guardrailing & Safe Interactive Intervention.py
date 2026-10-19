#Author - Rayaan Tasnim, prodigious programmer

import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GuardrailEngine")

@dataclass
class TelemetryRecord:
    step: int
    eval_latency_ms: float
    interception_triggered: bool
    safety_score: float
    original_token_id: int
    modified_token_id: int
    divergence_magnitude: float

@dataclass
class GuardrailConfig:
    safety_threshold: float = 0.75
    steering_strength: float = 1.5
    max_penalty_logits: float = -1e4
    safe_fallback_token_id: int = 13  # e.g., period or newline / safe token representation

class GuardrailEvaluator(nn.Module):
    """
    Evaluates hidden states and logits against a safety policy vector space.
    Simulates a lightweight classifier (e.g., Llama Guard inspired scoring head).
    """
    def __init__(self, hidden_dim: int, vocab_size: int, config: GuardrailConfig):
        super().__init__()
        self.config = config
        self.safety_projection = nn.Linear(hidden_dim, 1)
        self.steering_vector = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
        # Initialize weights deterministically for demonstration stability
        nn.init.xavier_uniform_(self.safety_projection.weight)
        nn.init.zeros_(self.safety_projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes safety alignment scores for incoming hidden states.
        Returns:
            safety_probs: Tensor of shape (batch_size, seq_len, 1) representing safety likelihood [0, 1]
            risk_logits: Raw risk scores
        """
        risk_logits = self.safety_projection(hidden_states)
        safety_probs = torch.sigmoid(-risk_logits)  # Higher values mean safer
        return safety_probs, risk_logits

class TensorHookManager:
    """
    Manages PyTorch forward hooks to intercept hidden states, evaluate safety policies,
    and apply tensor-level steering or token suppression in real time.
    """
    def __init__(self, evaluator: GuardrailEvaluator, config: GuardrailConfig):
        self.evaluator = evaluator
        self.config = config
        self.telemetry_logs: List[TelemetryRecord] = []
        self.current_step = 0
        self.active_interception_count = 0

    def get_hook(self):
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            start_time = time.perf_counter()
            
            # Handle potential tuple outputs from transformer blocks (hidden_states, attn_weights)
            is_tuple_output = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple_output else output
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Evaluate safety on the latest token representation (last sequence position)
            latest_hidden = hidden_states[:, -1:, :]
            safety_probs, risk_logits = self.evaluator(latest_hidden)
            safety_score = safety_probs.item()
            
            interception_triggered = False
            divergence = 0.0
            modified_hidden = hidden_states.clone()
            
            if safety_score < self.config.safety_threshold:
                interception_triggered = True
                self.active_interception_count += 1
                
                # Apply tensor steering: project hidden states away from prohibited vector space
                # and nudge toward safe steering vector
                steering_delta = self.evaluator.steering_vector.view(1, 1, -1) * self.config.steering_strength
                modified_hidden[:, -1:, :] = hidden_states[:, -1:, :] + steering_delta
                
                divergence = torch.norm(modified_hidden[:, -1, :] - hidden_states[:, -1, :]).item()
                logger.warning(
                    f"Step {self.current_step}: Safety violation detected! "
                    f"Safety score: {safety_score:.4f} < Threshold: {self.config.safety_threshold}. "
                    f"Interception applied with divergence magnitude: {divergence:.4f}"
                )
            
            eval_latency = (time.perf_counter() - start_time) * 1000.0
            
            # Record Telemetry
            record = TelemetryRecord(
                step=self.current_step,
                eval_latency_ms=eval_latency,
                interception_triggered=interception_triggered,
                safety_score=safety_score,
                original_token_id=0, # Populated by caller loop
                modified_token_id=0, # Populated by caller loop
                divergence_magnitude=divergence
            )
            self.telemetry_logs.append(record)
            self.current_step += 1
            
            if is_tuple_output:
                return (modified_hidden,) + output[1:]
            return modified_hidden

        return forward_hook

class SimulatedGenerativeModel(nn.Module):
    """
    Simulates a transformer-based generative language model with stacked projection blocks
    and hook attachment points.
    """
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, hook_manager: Optional[TensorHookManager] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for idx, layer in enumerate(self.layers):
            x = x + layer(x) # Residual connection
            # Attach hook to the final transformer layer if hook manager is present
            if idx == len(self.layers) - 1 and hook_manager is not None:
                # Register temporary forward hook for interception
                handle = layer.register_forward_hook(hook_manager.get_hook())
                x = layer(x) # re-run or let hook process
                handle.remove()
            else:
                x = layer(x)
                
        logits = self.lm_head(x)
        return logits

class GuardrailPipeline:
    """
    Orchestrates generation, token interception, logits manipulation, and telemetry reporting.
    """
    def __init__(self, vocab_size: int, hidden_dim: int, config: GuardrailConfig):
        self.config = config
        self.model = SimulatedGenerativeModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.evaluator = GuardrailEvaluator(hidden_dim=hidden_dim, vocab_size=vocab_size, config=config)
        self.hook_manager = TensorHookManager(evaluator=self.evaluator, config=config)
        self.vocab_size = vocab_size

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 10) -> Tuple[List[int], List[TelemetryRecord]]:
        self.model.eval()
        self.evaluator.eval()
        
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        generated_tokens = list(prompt_ids)
        
        logger.info(f"Starting guarded generation for prompt length {len(prompt_ids)}...")
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass through model with hook manager
                logits = self.model(input_tensor, hook_manager=self.hook_manager)
                next_token_logits = logits[:, -1, :]
                
                # Retrieve telemetry record just added by the hook
                latest_telemetry = self.hook_manager.telemetry_logs[-1] if self.hook_manager.telemetry_logs else None
                
                if latest_telemetry and latest_telemetry.interception_triggered:
                    # Policy Triggered: Override logits to force safe fallback token or suppress unsafe token
                    next_token_logits[0, :] = self.config.max_penalty_logits
                    next_token_logits[0, self.config.safe_fallback_token_id] = 0.0 # Force safe token probability to 1.0
                    latest_telemetry.original_token_id = torch.argmax(logits[:, -1, :]).item()
                    latest_telemetry.modified_token_id = self.config.safe_fallback_token_id
                else:
                    selected_token = torch.argmax(next_token_logits, dim=-1).item()
                    if latest_telemetry:
                        latest_telemetry.original_token_id = selected_token
                        latest_telemetry.modified_token_id = selected_token

                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_tokens.append(next_token.item())
                
                # Append to input tensor for next autoregressive step
                input_tensor = torch.cat([input_tensor, next_token], dim=1)

        logger.info(f"Generation completed. Total token interceptions: {self.hook_manager.active_interception_count}")
        return generated_tokens, self.hook_manager.telemetry_logs


if __name__ == "__main__":
    # Execution setup for Real-Time Guardrailing Framework
    VOCAB_SIZE = 1000
    HIDDEN_DIM = 256
    
    config = GuardrailConfig(
        safety_threshold=0.80,
        steering_strength=2.0,
        safe_fallback_token_id=42
    )
    
    pipeline = GuardrailPipeline(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, config=config)
    
    # Simulate an input prompt sequence
    sample_prompt = [101, 55, 230, 12, 89]
    
    output_tokens, telemetry_records = pipeline.generate(prompt_ids=sample_prompt, max_new_tokens=5)
    
    print("\n--- GENERATION TELEMETRY REPORT ---")
    print(f"Generated Token IDs: {output_tokens}")
    print(f"{'Step':<6} | {'Latency (ms)':<14} | {'Interception':<12} | {'Safety Score':<12} | {'Orig ID':<8} | {'Mod ID':<8} | {'Divergence':<10}")
    print("-" * 85)
    for rec in telemetry_records:
        print(f"{rec.step:<6} | {rec.eval_latency_ms:<14.4f} | {str(rec.interception_triggered):<12} | {rec.safety_score:<12.4f} | {rec.original_token_id:<8} | {rec.modified_token_id:<8} | {rec.divergence_magnitude:<10.4f}")