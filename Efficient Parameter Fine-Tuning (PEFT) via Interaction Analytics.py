#Author - Rayaan Tasnim, prodigious programmer

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging for telemetry and training feedback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PEFTFlywheel")


class DoRALinear(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA) Linear Layer.
    Decomposes weights into magnitude vector (m) and directional matrix (W = W_0 + B*A),
    allowing directional and magnitude updates to be decoupled.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        use_dora: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.use_dora = use_dora

        # Base linear layer (frozen weights W_0)
        self.base_layer = nn.Linear(in_features, out_features, bias=True)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Low-rank adaptation matrices A and B
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.scaling = lora_alpha / float(r)
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)
            self.scaling = 1.0

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Magnitude vector for DoRA decomposition
        if self.use_dora:
            with torch.no_grad():
                weight_matrix = self.base_layer.weight
                initial_norm = weight_matrix.norm(dim=1, keepdim=True)
            self.magnitude = nn.Parameter(initial_norm.clone())
        else:
            self.register_parameter('magnitude', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if self.r > 0:
            lora_out = self.lora_dropout(x)
            lora_out = torch.matmul(lora_out, self.lora_A.T)
            lora_out = torch.matmul(lora_out, self.lora_B.T) * self.scaling
            
            if self.use_dora:
                delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
                effective_weight = self.base_layer.weight + delta_w
                
                column_norms = effective_weight.norm(dim=1, keepdim=True).clamp(min=1e-5)
                normalized_weight = effective_weight / column_norms
                
                dora_weight_out = torch.matmul(x, normalized_weight.T)
                output = dora_weight_out * self.magnitude.squeeze(1)
                if self.base_layer.bias is not None:
                    output = output + self.base_layer.bias
                return output
            else:
                return base_out + lora_out
        return base_out


class InteractionModel(nn.Module):
    """
    Multi-Layer Perceptron model incorporating DoRA/LoRA enhanced layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, r: int = 8):
        super().__init__()
        self.layer1 = DoRALinear(input_dim, hidden_dim, r=r, use_dora=True)
        self.activation = nn.GELU()
        self.layer2 = DoRALinear(hidden_dim, output_dim, r=r, use_dora=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


@dataclass
class InteractionRecord:
    prompt_vector: List[float]
    correction_vector: List[float]
    timestamp: float = field(default_factory=time.time)


class OnlineAnalyticsTelemetryQueue:
    """
    Thread-safe online telemetry queue for capturing real-time human-edited
    corrections and input prompts.
    """
    def __init__(self, maxsize: int = 10000):
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.total_ingested = 0

    def push(self, prompt: List[float], correction: List[float]) -> None:
        record = InteractionRecord(prompt_vector=prompt, correction_vector=correction)
        try:
            self.queue.put_nowait(record)
            self.total_ingested += 1
        except queue.Full:
            logger.warning("Telemetry queue is full. Dropping incoming interaction record.")

    def fetch_batch(self, batch_size: int) -> List[InteractionRecord]:
        batch = []
        while len(batch) < batch_size and not self.queue.empty():
            try:
                batch.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return batch

    @property
    def size(self) -> int:
        return self.queue.qsize()


class PEFTFlywheelEngine:
    """
    Automated offline/online batching engine that processes telemetry data,
    computes losses against user corrections, and updates exclusively low-rank adapters.
    """
    def __init__(
        self,
        model: InteractionModel,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        update_interval_seconds: float = 2.0
    ):
        self.model = model
        self.batch_size = batch_size
        self.update_interval = update_interval_seconds
        self.telemetry_queue = OnlineAnalyticsTelemetryQueue()

        self._freeze_base_parameters()

        adaptable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(adaptable_params, lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.MSELoss()

        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

    def _freeze_base_parameters(self) -> None:
        frozen_count = 0
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name or "magnitude" in name:
                param.requires_grad = True
                trainable_count += param.numel()
                logger.info(f"Trainable Adapter Parameter: {name} (shape: {param.shape})")
            else:
                param.requires_grad = False
                frozen_count += param.numel()
                logger.info(f"Frozen Base Parameter: {name} (shape: {param.shape})")
        logger.info(f"Parameter Verification Complete -> Trainable: {trainable_count:,} | Frozen: {frozen_count:,}")

    def verify_parameter_freezing(self) -> bool:
        for name, param in self.model.named_parameters():
            is_adapter = "lora_" in name or "magnitude" in name
            if param.requires_grad != is_adapter:
                logger.error(f"Parameter constraint violation detected on {name}: requires_grad={param.requires_grad}")
                return False
        logger.info("Parameter freezing verification passed successfully.")
        return True

    def start_flywheel(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._flywheel_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Automated PEFT Interaction Flywheel background worker started.")

    def stop_flywheel(self) -> None:
        self._is_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
            logger.info("Automated PEFT Interaction Flywheel stopped.")

    def record_interaction(self, prompt: List[float], correction: List[float]) -> None:
        self.telemetry_queue.push(prompt, correction)

    def _flywheel_worker(self) -> None:
        while self._is_running:
            time.sleep(self.update_interval)
            if self.telemetry_queue.size >= self.batch_size:
                batch = self.telemetry_queue.fetch_batch(self.batch_size)
                if batch:
                    self._process_batch(batch)

    def _process_batch(self, batch: List[InteractionRecord]) -> None:
        self.model.train()
        
        prompts = torch.tensor([r.prompt_vector for r in batch], dtype=torch.float32)
        corrections = torch.tensor([r.correction_vector for r in batch], dtype=torch.float32)

        with torch.no_grad():
            preds_before = self.model(prompts)
            initial_loss = self.criterion(preds_before, corrections).item()

        self.optimizer.zero_grad()
        predictions = self.model(prompts)
        loss = self.criterion(predictions, corrections)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )

        self.optimizer.step()

        with torch.no_grad():
            preds_after = self.model(prompts)
            final_loss = self.criterion(preds_after, corrections).item()
            output_shift = torch.mean(torch.abs(preds_after - preds_before)).item()

        logger.info(
            f"[Flywheel Update] BatchProcessed={len(batch)} | "
            f"InitialLoss={initial_loss:.4f} | FinalLoss={final_loss:.4f} | "
            f"LossDelta={initial_loss - final_loss:.4f} | MeanOutputShift={output_shift:.4f}"
        )


if __name__ == "__main__":
    logger.info("Initializing Automated LoRA/DoRA Interaction Flywheel Framework...")

    INPUT_DIM = 16
    HIDDEN_DIM = 32
    OUTPUT_DIM = 8
    RANK = 4

    model = InteractionModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, r=RANK)
    flywheel = PEFTFlywheelEngine(model=model, learning_rate=5e-4, batch_size=2, update_interval_seconds=1.0)

    assert flywheel.verify_parameter_freezing(), "Critical Error: Base model weights are not fully frozen!"

    flywheel.start_flywheel()

    logger.info("Simulating real-time user correction telemetry influx...")
    for i in range(5):
        simulated_prompt = [float(i + j) * 0.1 for j in range(INPUT_DIM)]
        simulated_correction = [float(i + j + 1) * 0.12 for j in range(OUTPUT_DIM)]
        flywheel.record_interaction(prompt=simulated_prompt, correction=simulated_correction)
        time.sleep(0.4)

    time.sleep(3.5)

    flywheel.stop_flywheel()
    logger.info("PEFT Interaction Flywheel execution completed successfully.")