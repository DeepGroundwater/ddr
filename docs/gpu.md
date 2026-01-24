---
icon: lucide/gpu
---

# GPU Routing

DDR is optimized for GPU acceleration using sparse matrix operations. This guide explains how GPU routing works and how to optimize performance.

## Overview

DDR's routing algorithm involves solving sparse triangular linear systems at each timestep. GPU acceleration provides significant speedups for large networks

## Requirements

### Hardware

- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- Recommended: 8GB+ VRAM for CONUS-scale routing

### Software

```bash
# Install with GPU support
uv sync --all-packages --extra cu124
```

This installs:

- **PyTorch** with CUDA 12.4 support
- **CuPy**: GPU-accelerated NumPy/SciPy
- **cupyx.scipy.sparse**: GPU sparse matrix operations

## Sparse Matrix Implementation

### Why Sparse Matrices?

River networks are naturally sparse - each segment connects to at most a few upstream segments. Using dense matrices would be computationally infeasible

### COO Format

DDR stores adjacency matrices in Coordinate (COO) format, which efficiently represents sparse data:

```python
# COO representation
row_indices = [1, 2, 3, 3]      # Downstream segment indices
col_indices = [0, 1, 1, 2]      # Upstream segment indices
values = [1, 1, 1, 1]           # Connection weights (always 1)
shape = (n_segments, n_segments)

# Matrix interpretation:
# Flow from segment 0 → segment 1
# Flow from segment 1 → segment 2
# Flow from segment 1 → segment 3
# Flow from segment 2 → segment 3
```

### Lower Triangular Structure

The adjacency matrix is always lower triangular after topological sorting:

```
     0  1  2  3
0 [  0  0  0  0 ]  ← Headwater
1 [  1  0  0  0 ]  ← Receives from 0
2 [  0  1  0  0 ]  ← Receives from 1
3 [  0  1  1  0 ]  ← Receives from 1 and 2
```

This structure enables efficient forward substitution during routing.

## GPU Solver Architecture

### Triangular Sparse Solve

The core routing step solves: `A * Q_{t+1} = b`

Where:
- `A = I - C₁ * N` (identity minus scaled adjacency)
- `b = C₂(N·Q_t) + C₃·Q_t + C₄·Q'` (right-hand side)
- `C₁, C₂, C₃, C₄` = Muskingum-Cunge coefficients

```python
# GPU solver path (from ddr/routing/utils.py)
def triangular_sparse_solve(A_values, crow_indices, col_indices, b, lower, unit_diagonal, device):
    """Custom autograd function for sparse triangular solve."""

    if device == "cpu":
        # Use SciPy
        A_scipy = sp.csr_matrix((data_np, col_np, crow_np), shape=(n, n))
        x = spsolve_triangular(A_scipy, b_np, lower=lower)
    else:
        # Use CuPy (GPU)
        A_cp = cp_csr_matrix((data_cp, indices_cp, indptr_cp), shape=(n, n))
        x = cp_spsolve_triangular(A_cp, b_cp, lower=lower)

    return x
```

### Gradient Computation

DDR implements custom backward passes for sparse operations to allow autograd to work:

```python
# Backward pass (simplified)
def backward(ctx, grad_output):
    # Solve transposed system: A^T * gradb = grad_output
    A_T = A.T
    gradb = spsolve_triangular(A_T, grad_output, lower=not lower)

    # Gradient for matrix values
    # gradA_values = -gradb[rows] * x[cols]
    return gradA_values, gradb
```

float32 precision is used to decrease computational memory requirements
