# PAST-QAFTROM-X

# The Phase-Structured Quantum Fourier Transform Architecture

A structured kernel learning framework whose feature map is a phase-modulated QFT,
with provable expressiveness guarantees, polynomial gradient scaling (no barren plateaus),
and applications to cryptographic primitive design.

**Paper:** *PS-QFT: Phase-Structured Quantum Fourier Features with Programmable Kernels*

---

## Overview

The standard Quantum Fourier Transform has a fixed phase kernel `Φ(k,j) = 2πkj/N`.
PS-QFT generalizes this to a *programmable* kernel:

```
Φ(k,j;θ) = 2πkj/N + G(k,j;θ)
```

The perturbation `G` is the design parameter through which task-relevant structure
(cryptographic keys, signal priors, learned representations) enters the transform
while remaining within the class of **unitary operators**.

The maximal unitary subfamily has separable structure:

```
PS-QFT_{α,β} = D_α · QFT · D_β
```

where `D_α = diag(e^{iα_k})` and `D_β = diag(e^{iβ_j})` are diagonal unitaries.
This gives **2N real degrees of freedom** — versus 0 for the standard QFT.

### Theoretical guarantees

| Theorem | Claim |
|---------|-------|
| **Theorem 1** (Unitarity) | `D_α·QFT·D_β` is unitary for any real α, β |
| **Theorem 2** (Expressiveness) | `K_QFF ⊊ K_PSQFT ⊆ closure(K_RFF)` |
| **Theorem 3** (Entropy) | Optimized β always achieves H(β*) ≥ H(0); improvement is strict for non-flat spectra |
| **Theorem 4** (Avalanche) | WHT post-processing achieves strictly greater diffusion after PS-QFT than standard QFT |
| **Proposition 2** (No barren plateaus) | `‖∇H(β)‖ = Ω(1/N)` — polynomial in n = log₂N |

---

## Installation

```bash
# Core (NumPy + SciPy only)
pip install -e .

# With PyTorch backend (GPU acceleration, autograd)
pip install -e ".[torch]"

# With development tools
pip install -e ".[dev]"

# With example dependencies
pip install -e ".[examples]"
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10

---

## Quick Start

```python
import numpy as np
from psqft import PSQFT, KeySchedule, AvalancheEvaluator
from psqft.utils.encoding import BitEncoder

# --- Basic usage ---
N = 256
beta = KeySchedule(N, mode="hash").derive(b"my_secret_key")
op = PSQFT(N=N, beta=beta)

# Encode an input and compute output probabilities
enc = BitEncoder(N)
c = enc.encode(42)          # |42⟩ computational basis state
p = op(c)                   # output probability distribution, shape (N,)
H = op.output_entropy(c)    # Shannon entropy in bits

# Batched inputs
x_batch = np.arange(16)
C = enc.encode(x_batch)     # shape (16, N)
P = op(C)                   # shape (16, N)

# --- Kernel computation ---
from psqft import PSQFTKernel
kernel = PSQFTKernel(op)
K = kernel.gram_matrix(C)   # (16, 16) Gram matrix, PSD guaranteed

# --- Avalanche evaluation ---
from psqft import AvalancheEvaluator
report = AvalancheEvaluator(op).evaluate(n_samples=1000)
print(report)
# AvalancheReport(
#   AC = 0.4998 ± 0.0031
#   bias = 0.0002
#   SAC score = 0.9375
#   H(output) = 7.981 ± 0.012 bits
#   KL(p||uniform) = 0.000312
#   n_samples = 1000
# )

# --- Entropy maximization ---
from psqft.optim import EntropyAscent
solver = EntropyAscent(N, lr=0.1, max_iter=200, lr_schedule="cosine")
result = solver.solve(C, beta_init=beta)
print(result)
# OptimResult(status=converged, n_iter=147, final_entropy=7.999 bits)

op_optimal = PSQFT(N=N, beta=result.beta)
```

---

## Architecture

```
psqft/
├── core/
│   ├── operator.py           ← PS-QFT operators (NumPy backend)
│   │   ├── BasePSQFT         ← Abstract base: forward, kernel, entropy
│   │   ├── ColumnPhasePSQFT  ← QFT · D_β  (workhorse — β affects feature map)
│   │   ├── RowPhasePSQFT     ← D_α · QFT  (α cancels in probabilities)
│   │   ├── SeparablePSQFT    ← D_α · QFT · D_β  (full unitary family)
│   │   ├── PSQFT             ← Alias for SeparablePSQFT
│   │   └── LayeredPSQFT      ← Multi-layer: ∏ D_{αl} · QFT · D_{βl}
│   └── operator_torch.py     ← PyTorch backend (GPU, autograd)
│       ├── TorchColumnPhasePSQFT   ← nn.Module, β is nn.Parameter
│       ├── TorchSeparablePSQFT
│       ├── TorchLayeredPSQFT
│       └── TorchEntropyMaximizer   ← Adam-based entropy maximization
│
├── kernels/
│   └── kernel.py
│       ├── PSQFTKernel         ← k(x,x') = φ(x)ᵀφ(x'), Gram matrices, CKA
│       ├── SpectralCorrelation ← Σ_U matrix, spectral gap Δ(U,V)
│       ├── RFFKernel           ← Random Fourier Features baseline
│       └── KernelComparison    ← Empirical Theorem 2 validation
│
├── optim/
│   └── solver.py
│       ├── PhaseMatchingSolver ← Fixed-point solver for Theorem 3 equation
│       ├── EntropyAscent       ← Projected gradient ascent on H(β)
│       ├── MultiStartOptim     ← Multi-restart wrapper
│       └── key_derived_init    ← Key → β initialization
│
├── crypto/
│   ├── schedule.py
│   │   ├── KeySchedule         ← 4 derivation modes: linear, sbox, hash, interleave
│   │   └── MultiKeyMixer       ← Multi-party key combination
│   └── avalanche.py
│       ├── AvalancheEvaluator  ← AC, SAC, KL divergence, Theorem 4 bound
│       ├── WHTPipeline         ← PS-QFT + Walsh-Hadamard post-processing
│       ├── AvalancheReport     ← Structured results dataclass
│       └── wht()               ← Walsh-Hadamard Transform (standalone)
│
└── utils/
    └── encoding.py
        ├── BitEncoder          ← Integer → |x⟩ one-hot (paper's crypto encoding)
        ├── AmplitudeEncoder    ← Float vector → normalized amplitudes (ML default)
        ├── PhaseEncoder        ← Real values → equal-magnitude phase encoding
        └── DensityEncoder      ← Amplitude → density matrix ρ = cc†
```

---

## Design Decisions

**Why only NumPy + SciPy for the core?**
The key operations are FFT (O(N log N)) and element-wise phase multiplication — both
are embarrassingly parallel and well-optimized in NumPy/SciPy. The library is fully
functional without PyTorch, making it easy to install and use in constrained environments.
The optional torch backend adds autograd and GPU acceleration.

**Why is `ColumnPhasePSQFT` the workhorse?**
Row phases α cancel in |·|², so the feature map `φ_θ(x) = {p_k(x;β)}` depends only
on β. This means: (a) only β needs to be optimized for downstream tasks, and (b) the
gradient `∂H/∂β` is the only non-trivial gradient. RowPhase and full Separable operators
are useful for multi-layer compositions and amplitude-level processing.

**Why is the gradient analytic rather than autodiff?**
The gradient `∂H/∂β_j = 2·Re[i·c_mod_j · (wca @ F)_j]` where `wca` is the
loss-weighted conjugate amplitude vector and `F` is the DFT matrix, can be computed
in O(N²) naively or O(N log N) via a single additional FFT. This is exact (no
approximation), numerically stable, and avoids PyTorch as a hard dependency for
the core library.

**Why does the gradient have magnitude O(1/N)?**
The DFT matrix F has entries `1/sqrt(N)`, so `d(amp_k)/d(β_j) = O(1/sqrt(N))`,
and `dp_k/d(β_j) = O(1/N)`. Summing over k gives `∂H/∂β_j = O(1)` in the worst
case, but the typical magnitude is `O(1/sqrt(N))` — polynomial in n = log₂N.
This is what Proposition 2 guarantees: no exponential vanishing (no barren plateau).

---

## Key Schedules

Four derivation modes, ordered by cryptographic strength:

```python
from psqft.crypto.schedule import KeySchedule

N = 256
key = b"my_128_bit_key_!!"

# Linear: beta_j = (key[j % L] / 255) * 2pi  [paper default, fast]
beta_linear = KeySchedule(N, mode="linear").derive(key)

# S-box: apply nonlinear substitution before linear mapping
beta_sbox   = KeySchedule(N, mode="sbox").derive(key)

# Hash: BLAKE2b key expansion [cryptographically strong, recommended]
beta_hash   = KeySchedule(N, mode="hash").derive(key)

# Interleave: XOR consecutive bytes before mapping
beta_interleave = KeySchedule(N, mode="interleave").derive(key)

# Multi-party: combine multiple keys
from psqft.crypto.schedule import MultiKeyMixer
beta_combined = MultiKeyMixer(N, mode="sum").mix([key_alice, key_bob])
```

---

## PyTorch Backend

```python
from psqft.core.operator_torch import (
    TorchColumnPhasePSQFT,
    TorchEntropyMaximizer,
    from_numpy_op,
    to_numpy_op,
)
import torch

N = 1024
op = TorchColumnPhasePSQFT(N, device="cuda")  # GPU-accelerated

# Gradient via autograd
C = torch.randn(32, N, dtype=torch.complex128)
H = op.entropy(C)
H.mean().backward()
print(op.beta.grad)   # dH/d{beta}, computed via autograd

# Adam-based entropy maximization
maximizer = TorchEntropyMaximizer(op, lr=0.01, n_epochs=500)
history = maximizer.fit(C.numpy(), verbose=True)

# Convert between backends
numpy_op = to_numpy_op(op)       # TorchColumnPhasePSQFT → ColumnPhasePSQFT
torch_op = from_numpy_op(numpy_op, device="cuda")
```

---

## Reproducing Paper Results

```bash
# Table 1: all configurations (fast version N=256)
python examples/table1_reproduction.py --N 256 --n_samples 1000

# Table 1: paper-exact (N=2^16, takes ~2 min)
python examples/table1_reproduction.py --N 65536 --n_samples 100000

# Entropy-AC correlation (paper §7, r=0.71)
python examples/entropy_ac_correlation.py --N 64 --n_keys 500

# Kernel comparison (Theorem 2 hierarchy)
python examples/kernel_comparison.py --N 32 --n_samples 50
```

---

## Running Tests

```bash
# Install pytest
pip install pytest

# Run full test suite
pytest tests/ -v

# Run specific theorem tests
pytest tests/ -v -k "Unitarity"
pytest tests/ -v -k "Entropy"
pytest tests/ -v -k "Gradient"
```

---

## Citation

```bibtex
@inproceedings{psqft2025,
  title     = {{PS-QFT}: Phase-Structured Quantum Fourier Features
               with Programmable Kernels},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
}
```

---

## License

MIT License. See `LICENSE` for details.
