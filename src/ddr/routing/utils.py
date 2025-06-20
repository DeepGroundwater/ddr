import logging
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import spsolve_triangular

log = logging.getLogger(__name__)

# Try to import cupy - if not available, we'll handle gracefully during runtime
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cp_spsolve_triangular
except ImportError:
    log.warning("CuPy not available. GPU solver functionality will be disabled.")

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)


class PatternMapper:
    """Map data vectors to non-zero elements of a sparse matrix.

    This class handles mapping between data vectors and sparse matrix elements,
    utilizing a generic matrix filling operation to establish mapping relationships.
    """

    def __init__(
        self,
        fillOp,
        matrix_dim,
        constant_diags=None,
        constant_offsets=None,
        aux=None,
        indShift=0,
        device=None,
    ):
        """Initialize the PatternMapper.

        Parameters
        ----------
        fillOp : callable
            Function for filling the matrix.
        matrix_dim : int
            Dimension of the matrix.
        constant_diags : array_like, optional
            Constant diagonal values.
        constant_offsets : array_like, optional
            Offset values for diagonals.
        aux : any, optional
            Auxiliary data.
        indShift : int, optional
            Index shift value.
        device : str or torch.device, optional
            Computation device.
        """
        # indShift=1 if the sparse matrix of a library start with index 1
        # starting from offset rather than 0,
        # this is to avoid 0 getting removed from to_sparse_csr
        offset = 1
        indVec = torch.arange(
            start=offset,
            end=matrix_dim + offset,
            dtype=torch.float32,
            device=device,
        )
        A = fillOp(indVec)  # it can return either full or sparse

        if not A.is_sparse_csr:
            A = A.to_sparse_csr()

        self.crow_indices = A.crow_indices()
        self.col_indices = A.col_indices()

        I_matrix = A.values().int() - offset
        # this mapping is: datvec(I(i)) --> sp.values(i)
        # this is equivalent to datvec * M
        # where M is a zero matrix with 1 at (I(i),i)
        C = torch.arange(0, len(I_matrix), device=device) + indShift
        indices = torch.stack((I_matrix, C), 0)
        ones = torch.ones(len(I_matrix), device=device)
        M_coo = torch.sparse_coo_tensor(indices, ones, size=(matrix_dim, len(I_matrix)), device=device)
        self.M_csr = M_coo.to_sparse_csr()

    def map(self, datvec: torch.Tensor) -> torch.Tensor:
        """Map input vector to sparse matrix format.

        Parameters
        ----------
        datvec : torch.Tensor
            Input data vector.

        Returns
        -------
        torch.Tensor
            Mapped dense tensor.
        """
        return torch.matmul(datvec, self.M_csr).to_dense()

    def getSparseIndices(self):
        """Retrieve sparse matrix indices.

        Returns
        -------
        tuple
            Compressed row indices and column indices.
        """
        return self.crow_indices, self.col_indices

    @staticmethod
    def inverse_diag_fill(data_vector):
        """Fill matrix diagonally with inverse ordering of data vector.

        Parameters
        ----------
        data_vector : torch.Tensor
            Input vector for filling the matrix.

        Returns
        -------
        torch.Tensor
            Filled matrix.
        """
        # A = fillOp(vec) # A can be sparse in itself.
        n = data_vector.shape[0]
        # a slow matrix filling operation
        A = torch.zeros([n, n], dtype=data_vector.dtype)
        for i in range(n):
            A[i, i] = data_vector[n - 1 - i]
        return A

    @staticmethod
    def diag_aug(datvec, n, constant_diags, constant_offsets):
        """Augment data vector with constant diagonals.

        Parameters
        ----------
        datvec : torch.Tensor
            Input data vector.
        n : int
            Dimension size.
        constant_diags : array_like
            Constant diagonal values.
        constant_offsets : array_like
            Offset values for diagonals.

        Returns
        -------
        torch.Tensor
            Augmented data vector.
        """
        datvec_aug = datvec.clone()
        # constant_offsets = [0]
        for j in range(len(constant_diags)):
            d = torch.zeros([n]) + constant_diags[j]
            datvec_aug = torch.cat((datvec_aug, d), nsdim(datvec))
        return datvec_aug


def nsdim(datvec):
    """Find the first non-singleton dimension of the input tensor.

    Parameters
    ----------
    datvec : torch.Tensor
        Input tensor.

    Returns
    -------
    int or None
        Index of first non-singleton dimension.
    """
    for i in range(datvec.ndim):
        if datvec[i] > 1:
            ns = i
            return ns
    return None


def get_network_idx(mapper: PatternMapper) -> tuple[torch.Tensor, torch.Tensor]:
    """Gets the index of the csr saved in the PatternMapper

    Parameters
    ----------
    mapper: PatternMapper
        The pattern mapper object

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The sparse rows and columns contained in the pattern mapper
    """
    rows = []
    cols = []
    crow_indices_list = mapper.crow_indices.tolist()
    col_indices_list = mapper.col_indices.tolist()
    for i in range(len(crow_indices_list) - 1):
        start, end = crow_indices_list[i], crow_indices_list[i + 1]
        these_cols = col_indices_list[start:end]
        these_rows = [i] * len(these_cols)
        rows.extend(these_rows)
        cols.extend(these_cols)
    return torch.tensor(rows), torch.tensor(cols)


def denormalize(value: torch.Tensor, bounds: list[float]) -> torch.Tensor:
    """Denormalizing neural network outputs to be within the Physical Bounds

    Parameters
    ----------
    value: torch.Tensor
        The NN output
    bounds: list[float]
        The specified physical parameter bounds

    Returns
    -------
    torch.Tensor:
        The NN output in a physical parameter space
    """
    output = (value * (bounds[1] - bounds[0])) + bounds[0]
    return output


def torch_to_cupy(tensor):
    """Efficiently convert PyTorch tensor to CuPy array without going through CPU.

    This is much faster than tensor.cpu().numpy() for CUDA tensors.
    """
    pointer = tensor.data_ptr()
    size = tensor.shape
    dtype = tensor.dtype

    # Map PyTorch dtype to CuPy dtype
    dtype_map = {
        torch.float32: cp.float32,
        torch.float64: cp.float64,
        torch.int32: cp.int32,
        torch.int64: cp.int64,
    }
    if dtype not in dtype_map:
        # Fall back to numpy conversion for unsupported dtypes
        return cp.array(tensor.detach().cpu().numpy())
    cupy_dtype = dtype_map[dtype]

    # NOTE Create a CuPy array from the pointer which points
    # to the same GPU memory as the PyTorch tensor without any copying
    cupy_array = cp.ndarray(
        shape=size,
        dtype=cupy_dtype,
        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(pointer, 0, None), 0),
    )

    return cupy_array


def cupy_to_torch(cupy_array, device=None, dtype=torch.float32):
    """Efficiently convert CuPy array to PyTorch tensor without going through CPU."""
    # Determine the PyTorch dtype from CuPy dtype
    dtype_map = {
        cp.float32: torch.float32,
        cp.float64: torch.float64,
        cp.int32: torch.int32,
        cp.int64: torch.int64,
    }

    if cupy_array.dtype.type not in dtype_map:
        # Fall back to numpy conversion for unsupported dtypes
        return torch.from_numpy(cp.asnumpy(cupy_array)).to(device)

    # Get CuPy array pointer

    # NOTE Create a PyTorch tensor from the CuPy array pointer
    # This avoids any memory copying between GPU and CPU
    torch_dtype = dtype_map[cupy_array.dtype.type]
    torch_tensor = torch.empty(
        cupy_array.shape,
        dtype=dtype,
        device=device,
    )
    torch_tensor.data_ptr()

    result = torch_tensor.copy_(torch.as_tensor(np.zeros(cupy_array.shape).astype(np.float32), device=device))
    result.copy_(
        torch.frombuffer(cupy_array.tobytes(), dtype=torch_dtype).reshape(cupy_array.shape).to(device)
    )

    return result


class TriangularSparseSolver(torch.autograd.Function):
    """Custom autograd function for solving triangular sparse linear systems."""

    @staticmethod
    def forward(ctx, A_values, crow_indices, col_indices, b, lower, unit_diagonal, device):
        """Solve the sparse triangular linear system with optimized GPU transfers."""
        n = len(crow_indices) - 1

        if device == "cpu":
            crow_np = crow_indices.cpu().numpy().astype(np.int32)
            col_np = col_indices.cpu().numpy().astype(np.int32)
            data_np = A_values.cpu().numpy().astype(np.float64)
            b_np = b.cpu().numpy().astype(np.float64)

            A_scipy = sp.csr_matrix((data_np, col_np, crow_np), shape=(n, n))

            try:
                x_np = spsolve_triangular(A_scipy, b_np, lower=lower, unit_diagonal=unit_diagonal)
                x = torch.tensor(x_np, dtype=b.dtype, device=b.device)
            except ValueError as e:
                log.error(f"CPU triangular sparse solve failed: {e}")
                raise ValueError(f"SciPy triangular sparse solver failed: {e}") from e
        else:
            try:
                cuda_device = cp.cuda.Device(device)
                with cuda_device:
                    data_cp = torch_to_cupy(A_values)
                    indices_cp = torch_to_cupy(col_indices)
                    indptr_cp = torch_to_cupy(crow_indices)
                    b_cp = torch_to_cupy(b)
                    A_cp = cp_csr_matrix((data_cp, indices_cp, indptr_cp), shape=(n, n))

                    x_cp = cp_spsolve_triangular(A_cp, b_cp, lower=lower, unit_diagonal=unit_diagonal)

                    pytorch_device = A_values.device if A_values.is_cuda else b.device
                    x = cupy_to_torch(x_cp, device=pytorch_device)

            except ValueError as e:
                log.error(f"GPU triangular sparse solve failed: {e}")
                raise ValueError from e

        # Save all necessary context for backward
        ctx.save_for_backward(A_values, crow_indices, col_indices, x, b)
        ctx.lower = lower
        ctx.unit_diagonal = unit_diagonal
        ctx.device = device

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients with optimized memory transfers."""
        A_values, crow_indices, col_indices, x, b = ctx.saved_tensors
        lower = ctx.lower
        unit_diagonal = ctx.unit_diagonal
        device = ctx.device

        # NOTE For backward pass with triangular matrices: if A is lower triangular, A^T is upper triangular
        transposed_lower = not lower

        # Optimize the transposition process - do it in one go with vectorized operations
        n = len(crow_indices) - 1

        # Create CSR to COO conversion using optimized PyTorch operations
        # This is much faster than Python loops for large matrices
        rows = []
        cols = []
        crow_indices_cpu = crow_indices.cpu()
        col_indices_cpu = col_indices.cpu()

        for i in range(n):
            start, end = crow_indices_cpu[i].item(), crow_indices_cpu[i + 1].item()
            row_count = end - start
            if row_count > 0:  # Skip empty rows
                rows.extend([i] * row_count)
                cols.extend([col_indices_cpu[j].item() for j in range(start, end)])

        # Create COO indices for transposed matrix
        transposed_indices = torch.tensor([cols, rows], dtype=torch.int32, device=A_values.device)

        # Create transposed COO tensor
        A_T_values = A_values.clone()
        A_T_coo = torch.sparse_coo_tensor(transposed_indices, A_T_values, size=(n, n), device=A_values.device)

        # Convert to CSR for solving
        A_T_csr = A_T_coo.to_sparse_csr()
        A_T_crow = A_T_csr.crow_indices()
        A_T_col = A_T_csr.col_indices()
        A_T_values = A_T_csr.values()

        # Solve the transposed system
        gradb = TriangularSparseSolver.apply(
            A_T_values, A_T_crow, A_T_col, grad_output, transposed_lower, unit_diagonal, device
        )

        # Optimize gradient computation for A_values if needed
        if A_values.requires_grad:
            if len(rows) > 0:  # Make sure we have non-zero elements
                row_indices = torch.tensor(rows, device=A_values.device)
                col_indices_vector = torch.tensor(cols, device=A_values.device)

                # Vectorized gradient computation: -gradb[rows] * x[cols]
                gradA_values = -gradb[row_indices] * x[col_indices_vector]
            else:
                gradA_values = torch.zeros_like(A_values)

            return gradA_values, None, None, gradb, None, None, None
        else:
            return None, None, None, gradb, None, None, None


triangular_sparse_solve = TriangularSparseSolver.apply
