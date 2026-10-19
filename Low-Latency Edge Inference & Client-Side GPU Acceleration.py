#Author - Rayaan Tasnim, prodigious programmer

import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

class EdgeTransformerBlock(nn.Module):
    """
    A lightweight, production-ready Transformer-based feature extractor block
    optimized for edge deployment and low-latency client-side inference.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm Transformer architecture for numerical stability
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x


class SyntheticCalibrationDataReader(CalibrationDataReader):
    """
    Calibration Data Reader for Post-Training Quantization (PTQ) 
    supplying representative calibration batches.
    """
    def __init__(self, input_name: str, batch_size: int = 1, seq_len: int = 32, embed_dim: int = 128, num_samples: int = 64):
        self.input_name = input_name
        self.samples = [
            np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
            for _ in range(num_samples)
        ]
        self.enum_data = iter([{self.input_name: sample} for sample in self.samples])

    def get_next(self):
        try:
            return next(self.enum_data)
        except StopIteration:
            return None


class EdgeModelCompiler:
    """
    Handles model export, graph validation, optimization (node fusion),
    and Post-Training Quantization (PTQ) for ONNX Runtime.
    """
    def __init__(self, model: nn.Module, dummy_input: torch.Tensor):
        self.model = model
        self.dummy_input = dummy_input
        self.model.eval()

    def export_to_onnx(self, onnx_path: str, op_set: int = 17) -> str:
        """Exports PyTorch module dynamically to an ONNX graph schema."""
        print(f"[*] Exporting model to ONNX format at '{onnx_path}'...")
        torch.onnx.export(
            self.model,
            self.dummy_input,
            onnx_path,
            export_params=True,
            opset_version=op_set,
            do_constant_folding=True,
            input_names=["input_features"],
            output_names=["output_features"],
            dynamic_axes={
                "input_features": {0: "batch_size", 1: "sequence_length"},
                "output_features": {0: "batch_size", 1: "sequence_length"}
            }
        )
        
        # Validate graph syntax
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model successfully exported and validated.")
        return onnx_path

    def optimize_graph(self, input_onnx_path: str, output_onnx_path: str) -> str:
        """Applies node-fusion and structural graph optimizations."""
        print(f"[*] Applying graph optimizations (node-fusion) to '{input_onnx_path}'...")
        from onnxruntime.transformers.optimizer import optimize_model
        
        optimized_model = optimize_model(
            input_onnx_path,
            model_type='bert',  # Generic transformer graph optimizations
            num_heads=4,
            hidden_size=128
        )
        optimized_model.save_model_to_file(output_onnx_path)
        print(f"[+] Optimized model saved to '{output_onnx_path}'.")
        return output_onnx_path

    def quantize_model(self, input_onnx_path: str, output_quant_path: str) -> str:
        """
        Performs Post-Training Quantization (PTQ) mapping FP32 tensor weights
        down to INT8 configurations, maintaining calibrated scales and zero-points.
        """
        print(f"[*] Performing Post-Training Quantization (PTQ) on '{input_onnx_path}'...")
        calibration_data_reader = SyntheticCalibrationDataReader(input_name="input_features")
        
        quantize_static(
            model_input=input_onnx_path,
            model_output=output_quant_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True
        )
        print(f"[+] Quantized INT8 model successfully generated at '{output_quant_path}'.")
        return output_quant_path


class EdgeInferenceRunner:
    """
    Simulates client-side ultra-low latency execution loops, tracking and benchmarking
    inference speedups, memory footprints, and structural telemetry profiles.
    """
    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider"):
        self.model_path = model_path
        self.provider = provider
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(self.model_path, session_options, providers=[self.provider])
        self.input_name = self.session.get_inputs()[0].name

    def benchmark(self, batch_size: int = 1, seq_len: int = 32, embed_dim: int = 128, iterations: int = 200) -> dict:
        """Executes performance benchmark tracking latency, throughput, and memory footprint."""
        print(f"[*] Benchmarking model '{self.model_path}' using provider '{self.provider}'...")
        
        input_data = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        
        # Warmup iterations
        for _ in range(20):
            _ = self.session.run(None, {self.input_name: input_data})
            
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        latencies = []
        start_global = time.perf_counter()
        
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = self.session.run(None, {self.input_name: input_data})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            
        total_time = time.perf_counter() - start_global
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        latencies_arr = np.array(latencies)
        metrics = {
            "model_path": self.model_path,
            "provider": self.provider,
            "mean_latency_ms": float(np.mean(latencies_arr)),
            "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
            "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
            "throughput_qps": float(iterations / total_time),
            "memory_footprint_mb": float(mem_after - mem_before)
        }
        return metrics


if __name__ == "__main__":
    print("=== Initializing Low-Latency Edge Inference & Compilation Pipeline ===")
    
    # Configuration
    BATCH_SIZE = 1
    SEQ_LEN = 32
    EMBED_DIM = 128
    
    # 1. Instantiate PyTorch Model Block
    pytorch_model = EdgeTransformerBlock(embed_dim=EMBED_DIM, num_heads=4, ff_dim=256)
    dummy_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, dtype=torch.float32)
    
    # Paths for artifacts
    os.makedirs("build", exist_ok=True)
    base_onnx_path = "build/edge_model_fp32.onnx"
    opt_onnx_path = "build/edge_model_optimized.onnx"
    quant_onnx_path = "build/edge_model_int8.onnx"
    
    # 2. Programmatic Model Compilation Layer
    compiler = EdgeModelCompiler(pytorch_model, dummy_tensor)
    exported_path = compiler.export_to_onnx(base_onnx_path)
    
    # Try node-fusion optimization, fallback to base if transformer optimizer schema mismatch occurs
    try:
        optimized_path = compiler.optimize_graph(exported_path, opt_onnx_path)
    except Exception as e:
        print(f"[!] Warning: Transformer graph optimizer encountered structural exception ({e}). Using exported base model.")
        optimized_path = exported_path

    # 3. Post-Training Quantization (PTQ) Pipeline
    quantized_path = compiler.quantize_model(optimized_path, quant_onnx_path)
    
    # 4. Simulation Runner & Telemetry Profiler
    print("\n=== Executing Performance Benchmarks ===")
    runner_fp32 = EdgeInferenceRunner(optimized_path)
    metrics_fp32 = runner_fp32.benchmark(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, embed_dim=EMBED_DIM)
    
    runner_int8 = EdgeInferenceRunner(quantized_path)
    metrics_int8 = runner_int8.benchmark(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, embed_dim=EMBED_DIM)
    
    print("\n--- Benchmark Results Summary ---")
    print(f"FP32 Model ({os.path.basename(optimized_path)}):")
    print(f"  - Mean Latency: {metrics_fp32['mean_latency_ms']:.3f} ms")
    print(f"  - P95 Latency:  {metrics_fp32['p95_latency_ms']:.3f} ms")
    print(f"  - Throughput:   {metrics_fp32['throughput_qps']:.2f} QPS")
    
    print(f"\nINT8 Quantized Model ({os.path.basename(quantized_path)}):")
    print(f"  - Mean Latency: {metrics_int8['mean_latency_ms']:.3f} ms")
    print(f"  - P95 Latency:  {metrics_int8['p95_latency_ms']:.3f} ms")
    print(f"  - Throughput:   {metrics_int8['throughput_qps']:.2f} QPS")
    
    speedup = metrics_fp32['mean_latency_ms'] / metrics_int8['mean_latency_ms']
    print(f"\n[+] Quantization Speedup Factor: {speedup:.2f}x")
    print("=== Pipeline Execution Complete Successfully ===")