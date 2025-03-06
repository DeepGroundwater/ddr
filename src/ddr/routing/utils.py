import logging
import warnings

import torch
import numpy as np
import scipy.sparse as sp
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


class TriangularSparseSolver(torch.autograd.Function):
    """Custom autograd function for solving triangular sparse linear systems.
    """
    
    @staticmethod
    def forward(ctx, A_values, crow_indices, col_indices, b, lower, unit_diagonal, device):
        """Solve the sparse triangular linear system.
        
        Parameters
        ----------
        ctx : Context
            Context object for storing information for backward computation.
        A_values : torch.Tensor
            Values for the sparse tensor.
        crow_indices : torch.Tensor
            Compressed row indices.
        col_indices : torch.Tensor
            Column indices.
        b : torch.Tensor
            Dense tensor representing the right-hand side.
        lower : bool, (True)
            Whether to solve a lower triangular system. Default is True.
        unit_diagonal : bool, (True)
            Whether the diagonal of the matrix consists of ones. Default is False.
            
        Returns
        -------
        torch.Tensor
            Solution to the system Ax = b.
        """
        # convert to Scipy csr
        crow_np = crow_indices.cpu().numpy().astype(np.int32)
        col_np = col_indices.cpu().numpy().astype(np.int32)
        data_np = A_values.cpu().numpy().astype(np.float64)
        b_np = b.cpu().numpy().astype(np.float64)
            
        n = len(crow_np) - 1
        
        if device == "cpu":
            A_scipy = sp.csr_matrix((data_np, col_np, crow_np), shape=(n, n))
            try:
                x_np = spsolve_triangular(
                    A_scipy, b_np, lower=lower, unit_diagonal=unit_diagonal
                )
                    
            except Exception as e:
                log.error(f"Triangular sparse solve failed: {e}")
                raise ValueError(f"SciPy triangular sparse solver failed: {e}")
        else:
            device = cp.cuda.Device(device)  # Device 1
            with device:
                data_cp = cp.array(data_np)
                indices_cp = cp.array(col_np)
                indptr_cp = cp.array(crow_np)
                b_cp = cp.array(b_np)
                
                # Create CuPy CSR matrix
                A_cp = cp_csr_matrix((data_cp, indices_cp, indptr_cp), shape=(n, n))
                
                # Solve on GPU
                x_cp = cp_spsolve_triangular(
                    A_cp, b_cp, lower=lower, unit_diagonal=unit_diagonal
                )
                
                # Transfer solution back to CPU
                x_np = cp.asnumpy(x_cp)
                log.debug("GPU solver completed successfully")
            
        # Convert solution back to PyTorch tensor and save gradients/states
        x = torch.tensor(x_np, dtype=b.dtype, device=b.device)
        ctx.save_for_backward(A_values, crow_indices, col_indices, x, b)
        ctx.lower = lower
        ctx.unit_diagonal = unit_diagonal
        ctx.device = device
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for the backward pass.
        
        Parameters
        ----------
        ctx : Context
            Context object containing saved tensors.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.
            
        Returns
        -------
        tuple
            Gradients with respect to inputs.
        """
        A_values, crow_indices, col_indices, x, b = ctx.saved_tensors
        lower = ctx.lower
        unit_diagonal = ctx.unit_diagonal
        device = ctx.device
            
        # NOTE For backward pass with triangular matrices, we need to be careful
        # Since A is lower triangular, A^T is upper triangular
        transposed_lower = not lower  # opposite 
        
        # Convert to COO format for easier transposition
        n = len(crow_indices) - 1
        rows = []
        cols = []
        for i in range(n):
            start, end = crow_indices[i].item(), crow_indices[i+1].item()
            for j in range(start, end):
                rows.append(i)
                cols.append(col_indices[j].item())
        
        transposed_indices = torch.tensor([cols, rows], dtype=torch.int64, device=A_values.device)
        A_T_values = A_values.clone()  # Values stay the same for transpose
        A_T_coo = torch.sparse_coo_tensor(
            transposed_indices, A_T_values, size=(n, n), device=A_values.device
        )
        
        # Convert to CSR for efficient solving
        A_T_csr = A_T_coo.to_sparse_csr()
        A_T_crow = A_T_csr.crow_indices()
        A_T_col = A_T_csr.col_indices()
        A_T_values = A_T_csr.values()
        
        # Solve the transposed system to get gradb
        gradb = TriangularSparseSolver.apply(
            A_T_values, A_T_crow, A_T_col, grad_output, transposed_lower, unit_diagonal, device   
        )
        
        # NOTE: For gradA, we need to compute -gradb * x^T
        # But since A is sparse, we only need the gradients at the non-zero locations
        if A_values.requires_grad:
            gradA_values = torch.zeros_like(A_values)
            
            for i in range(n):
                start, end = crow_indices[i].item(), crow_indices[i+1].item()
                for j_idx in range(start, end):
                    j = col_indices[j_idx].item()
                    # Compute gradient for this location: -gradb[i] * x[j]
                    gradA_values[j_idx] = -gradb[i] * x[j]
                        
            return gradA_values, None, None, gradb, None, None, None
        else:
            return None, None, None, gradb, None, None, None


triangular_sparse_solve = TriangularSparseSolver.apply
