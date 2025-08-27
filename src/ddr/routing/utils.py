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


def _backward_cpu(
    A_values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    x: torch.Tensor,
    grad_output: torch.Tensor,
    transposed_lower: bool,
    unit_diagonal: bool,
) -> torch.Tensor:
    """
    CPU backward pass using SciPy for sparse triangular linear system.

    This function solves the transposed system A^T * gradb = grad_output using SciPy's
    efficient sparse matrix operations, avoiding the expensive manual transpose construction.

    Parameters
    ----------
    A_values : torch.Tensor
        Values of the sparse matrix A in CSR format.
    crow_indices : torch.Tensor
        Row pointers for the CSR format (crow_indices).
    col_indices : torch.Tensor
        Column indices for the CSR format.
    x : torch.Tensor
        Solution vector from the forward pass (Ax = b).
    grad_output : torch.Tensor
        Gradient from the next layer in the computational graph.
    transposed_lower : bool
        Whether the transposed matrix is lower triangular.
    unit_diagonal : bool
        Whether the matrix has unit diagonal elements.

    Returns
    -------
    torch.Tensor
        Gradient with respect to the right-hand side vector b.
    """
    n = len(crow_indices) - 1

    # Convert to SciPy CSR matrix
    crow_np = crow_indices.cpu().numpy().astype(np.int32)
    col_np = col_indices.cpu().numpy().astype(np.int32)
    data_np = A_values.cpu().numpy().astype(np.float64)

    A_scipy = sp.csr_matrix((data_np, col_np, crow_np), shape=(n, n))

    A_T_scipy = A_scipy.T

    # Solve transposed system
    grad_output_np = grad_output.cpu().numpy().astype(np.float64)
    gradb_np = spsolve_triangular(
        A_T_scipy, grad_output_np, lower=transposed_lower, unit_diagonal=unit_diagonal
    )

    return torch.tensor(gradb_np, dtype=grad_output.dtype, device=grad_output.device)


def _backward_gpu(
    A_values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    x: torch.Tensor,
    grad_output: torch.Tensor,
    transposed_lower: bool,
    unit_diagonal: bool,
    device: str | torch.device,
) -> torch.Tensor:
    """
    GPU backward pass using CuPy for sparse triangular linear system.

    This function performs the backward pass computation entirely on GPU using CuPy's
    efficient sparse matrix operations, avoiding CPU-GPU transfers during transpose.

    Parameters
    ----------
    A_values : torch.Tensor
        Values of the sparse matrix A in CSR format.
    crow_indices : torch.Tensor
        Row pointers for the CSR format (crow_indices).
    col_indices : torch.Tensor
        Column indices for the CSR format.
    x : torch.Tensor
        Solution vector from the forward pass (Ax = b).
    grad_output : torch.Tensor
        Gradient from the next layer in the computational graph.
    transposed_lower : bool
        Whether the transposed matrix is lower triangular.
    unit_diagonal : bool
        Whether the matrix has unit diagonal elements.
    device : str | torch.device
        CUDA device identifier (e.g., 'cuda:0').

    Returns
    -------
    torch.Tensor
        Gradient with respect to the right-hand side vector b.

    Raises
    ------
    Exception
        If GPU computation fails, falls back to CPU implementation.
    """
    n = len(crow_indices) - 1

    cuda_device = cp.cuda.Device(device)
    with cuda_device:
        # Convert to CuPy CSR matrix (reusing conversion functions)
        data_cp = torch_to_cupy(A_values)
        indices_cp = torch_to_cupy(col_indices)
        indptr_cp = torch_to_cupy(crow_indices)
        grad_output_cp = torch_to_cupy(grad_output)

        A_cp = cp_csr_matrix((data_cp, indices_cp, indptr_cp), shape=(n, n))

        # Use CuPy's efficient transpose (GPU-native operation)
        A_T_cp = A_cp.T

        # Solve transposed system on GPU
        gradb_cp = cp_spsolve_triangular(
            A_T_cp, grad_output_cp, lower=transposed_lower, unit_diagonal=unit_diagonal
        )

        # Convert back to PyTorch
        pytorch_device = A_values.device if A_values.is_cuda else grad_output.device
        return cupy_to_torch(gradb_cp, device=pytorch_device)

    # NOTE: Uncomment if we get to an exception one day
    # except Exception as e:
    #     log.warning(f"GPU backward failed, falling back to CPU: {e}")
    #     # Fallback to CPU implementation
    #     return _backward_cpu(
    #         A_values, crow_indices, col_indices, x, grad_output, transposed_lower, unit_diagonal
    #     )


def _compute_A_gradients(
    A_values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    x: torch.Tensor,
    gradb: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Gradient computation for sparse matrix values A_values.

    Computes gradients with respect to the non-zero values of matrix A using the formula:
    gradA_values = -gradb[rows] * x[cols], where rows and cols correspond to the
    non-zero positions in the sparse matrix.

    Parameters
    ----------
    A_values : torch.Tensor
        Values of the sparse matrix A in CSR format.
    crow_indices : torch.Tensor
        Row pointers for the CSR format (crow_indices).
    col_indices : torch.Tensor
        Column indices for the CSR format.
    x : torch.Tensor
        Solution vector from the forward pass (Ax = b).
    gradb : torch.Tensor
        Gradient with respect to the right-hand side vector b.
    device : str | torch.device
        Device where computations should be performed.

    Returns
    -------
    torch.Tensor
        Gradient with respect to the non-zero values of matrix A.

    Notes
    -----
    This function uses vectorized operations to compute gradients efficiently:
    - On CPU: Pre-allocates arrays and uses vectorized row index computation
    - On GPU: Uses torch.repeat_interleave for efficient row index generation

    The gradient computation is based on the matrix calculus identity for linear solves.
    """
    if device == "cpu":
        # CPU path - vectorized operations
        crow_indices_cpu = crow_indices.cpu()

        # Pre-allocate arrays for better memory access
        nnz = len(A_values)
        row_indices = torch.empty(nnz, dtype=torch.long, device="cpu")

        # Vectorized row index computation
        _fill_row_indices_vectorized(crow_indices_cpu, row_indices)

        # Move to target device and compute gradients
        row_indices = row_indices.to(device)
        col_indices_device = col_indices.to(device)

        # Vectorized gradient computation: -gradb[rows] * x[cols]
        gradA_values = -gradb[row_indices] * x[col_indices_device]

    else:
        # GPU path - keep everything on GPU
        row_indices = _compute_row_indices_gpu(crow_indices, len(A_values))

        # Vectorized gradient computation on GPU
        gradA_values = -gradb[row_indices] * x[col_indices]

    return gradA_values


def _fill_row_indices_vectorized(crow_indices: torch.Tensor, row_indices: torch.Tensor) -> None:
    """
    Vectorized computation of row indices from CSR crow_indices for CPU.

    Fills the row_indices tensor with the row index for each non-zero element
    in the sparse matrix, based on the CSR format's crow_indices.

    Parameters
    ----------
    crow_indices : torch.Tensor
        Row pointers for the CSR format. Should be on CPU.
    row_indices : torch.Tensor
        Pre-allocated tensor to store row indices. Will be modified in-place.

    Notes
    -----
    This function operates in-place on the row_indices tensor for memory efficiency.
    The crow_indices[i] gives the starting position in the values array for row i,
    and crow_indices[i+1] gives the ending position.

    Examples
    --------
    For a 3x3 matrix with crow_indices = [0, 2, 3, 5]:
    - Row 0 has 2 non-zeros at positions 0, 1
    - Row 1 has 1 non-zero at position 2
    - Row 2 has 2 non-zeros at positions 3, 4
    Result: row_indices = [0, 0, 1, 2, 2]
    """
    n = len(crow_indices) - 1
    idx = 0

    for i in range(n):
        start, end = crow_indices[i].item(), crow_indices[i + 1].item()
        count = end - start
        if count > 0:
            row_indices[idx : idx + count] = i
            idx += count


def _compute_row_indices_gpu(crow_indices: torch.Tensor, nnz: int) -> torch.Tensor:
    """
    GPU computation of row indices using PyTorch operations.

    Computes row indices for each non-zero element in a sparse matrix using
    efficient GPU operations, avoiding CPU-GPU transfers.

    Parameters
    ----------
    crow_indices : torch.Tensor
        Row pointers for the CSR format. Should be on GPU.
    nnz : int
        Number of non-zero elements in the sparse matrix.

    Returns
    -------
    torch.Tensor
        Row indices for each non-zero element, with shape (nnz,).

    Notes
    -----
    This function uses torch.repeat_interleave which is on GPU
    and avoids the Python loops required in the CPU version. The operation is
    equivalent to expanding each row index i by the number of non-zeros in that row.

    Examples
    --------
    For a matrix with crow_indices = [0, 2, 3, 5] (3 rows):
    - Row 0: 2 non-zeros
    - Row 1: 1 non-zero
    - Row 2: 2 non-zeros
    Result: [0, 0, 1, 2, 2]
    """
    # Use torch operations to avoid CPU-GPU transfers
    device = crow_indices.device
    n = len(crow_indices) - 1

    # Compute row counts
    row_counts = crow_indices[1:] - crow_indices[:-1]

    # Use repeat_interleave for efficient row index generation
    row_indices = torch.arange(n, device=device, dtype=torch.long)
    row_indices = torch.repeat_interleave(row_indices, row_counts)

    return row_indices


def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """
    Converts a torch tensor to the cupy using dlpack

    Parameters
    ----------
    t: torch.Tensor
        the input tensor

    Returns
    -------
    cp.ndarray
        The same input tensor moved to a CUPY array
    """
    assert t.is_cuda, "Expect a CUDA tensor"
    t = t.contiguous()  # ensure C-contiguous layout
    with cp.cuda.Device(t.device.index):
        return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t))

def cupy_to_torch(a: cp.ndarray) -> torch.Tensor:
    """
    Returns a CUDA tensor sharing memory with `a`

    Parameters
    ----------
    a: cp.ndarray
        the input cupy array

    Returns
    -------
    torch.Tensor
        The output pytorch tensor
    """
    return torch.utils.dlpack.from_dlpack(a)


class TriangularSparseSolver(torch.autograd.Function):
    """
    Custom autograd function for solving triangular sparse linear systems.

    This class implements forward and backward passes for solving sparse triangular
    linear systems of the form Ax = b, where A is a sparse triangular matrix.
    The implementation supports both CPU (via SciPy) and GPU (via CuPy) computation
    with memory transfers and transpose operations.

    Notes
    -----
    The backward pass solves the transposed system A^T * gradb = grad_output to
    compute gradients with respect to the right-hand side b, and uses matrix
    calculus to compute gradients with respect to the matrix values A.

    For the gradient computation:
    - ∇b = A^(-T) * grad_output (solved via transpose system)
    - ∇A = -A^(-T) * grad_output * x^T (computed efficiently for sparse entries)
    """

    @staticmethod
    def forward(
        ctx,
        A_values: torch.Tensor,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        b: torch.Tensor,
        lower: bool,
        unit_diagonal: bool,
        device: str | torch.device,
    ) -> torch.Tensor:
        """
        Solve the sparse triangular linear system Ax = b.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object for saving tensors for backward pass.
        A_values : torch.Tensor
            Values of the sparse triangular matrix A in CSR format.
        crow_indices : torch.Tensor
            Row pointers for the CSR format.
        col_indices : torch.Tensor
            Column indices for the CSR format.
        b : torch.Tensor
            Right-hand side vector.
        lower : bool
            Whether A is lower triangular (True) or upper triangular (False).
        unit_diagonal : bool
            Whether A has unit diagonal elements.
        device : str | torch.device
            Device for computation ('cpu' or CUDA device like 'cuda:0').

        Returns
        -------
        torch.Tensor
            Solution vector x such that Ax = b.

        Raises
        ------
        ValueError
            If the sparse solver fails to converge or encounters numerical issues.

        Notes
        -----
        The forward pass uses:
        - SciPy's spsolve_triangular for CPU computation
        - CuPy's spsolve_triangular for GPU computation

        """
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
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, None, None, None]:
        """
        Compute gradients for the sparse triangular linear system solve.

        This method implements the backward pass for the triangular solve operation,
        computing gradients with respect to the matrix values A and right-hand side b.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object containing saved tensors from forward pass.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output x.

        Returns
        -------
        Tuple[torch.Tensor, None, None, torch.Tensor, None, None, None]
            Gradients with respect to:
            - A_values: Gradient w.r.t. sparse matrix values
            - crow_indices: None (not differentiable)
            - col_indices: None (not differentiable)
            - b: Gradient w.r.t. right-hand side vector
            - lower: None (not differentiable)
            - unit_diagonal: None (not differentiable)
            - device: None (not differentiable)

        Notes
        -----
        The backward pass computes:
        1. ∇b by solving A^T * gradb = grad_output
        2. ∇A by computing -gradb[rows] * x[cols] for non-zero entries

        The implementation uses transpose operations:
        - SciPy's .T property for CPU
        - CuPy's .T property for GPU

        This avoids the expensive manual CSR ↔ COO conversions used in naive implementations.
        """
        A_values, crow_indices, col_indices, x, b = ctx.saved_tensors
        lower = ctx.lower
        unit_diagonal = ctx.unit_diagonal
        device = ctx.device

        # NOTE For backward pass with triangular matrices: if A is lower triangular, A^T is upper triangular
        transposed_lower = not lower

        if device == "cpu":
            gradb = _backward_cpu(
                A_values, crow_indices, col_indices, x, grad_output, transposed_lower, unit_diagonal
            )
        else:
            gradb = _backward_gpu(
                A_values, crow_indices, col_indices, x, grad_output, transposed_lower, unit_diagonal, device
            )

        # gradient computation for A_values if needed
        if A_values.requires_grad:
            gradA_values = _compute_A_gradients(A_values, crow_indices, col_indices, x, gradb, device)
            return gradA_values, None, None, gradb, None, None, None
        else:
            return None, None, None, gradb, None, None, None


triangular_sparse_solve = TriangularSparseSolver.apply
