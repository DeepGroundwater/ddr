import logging
import warnings

import torch

log = logging.getLogger(__name__)

# Disable prototype warnings and such
warnings.filterwarnings(action="ignore", category=UserWarning)


class SolverError(Exception):
    """Custom exception for solver-related errors"""

    pass


class RiverNetworkMatrix(torch.autograd.Function):
    """Custom autograd function for sparse tensor operations in river routing."""

    @staticmethod
    def forward(*args, **kwargs):
        """Create a sparse CSR tensor from input values and indices.

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

        Returns
        -------
        torch.sparse_csr_tensor
            Constructed sparse CSR tensor.
        """
        ctx, A_values, crow_indices, col_indices = args
        A_csr = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            A_values,
        )
        ctx.save_for_backward(
            A_values,
            A_csr,
            crow_indices,
            col_indices,
        )
        return A_csr

    @staticmethod
    def backward(*args):
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
            Gradients with respect to the inputs.
        """

        def extract_csr_values(
            col_indices: torch.Tensor,
            crow_indices: torch.Tensor,
            dense_matrix: torch.Tensor,
        ) -> torch.Tensor:
            """Extract values from dense matrix using CSR format indices.

            Parameters
            ----------
            col_indices : torch.Tensor
                Column indices for sparse matrix.
            crow_indices : torch.Tensor
                Compressed row indices.
            dense_matrix : torch.Tensor
                Dense matrix to extract values from.

            Returns
            -------
            torch.Tensor
                Extracted values in CSR format.
            """
            crow_indices_list = crow_indices.tolist()
            col_indices_list = col_indices.tolist()
            rows = []
            cols = []
            for i in range(len(crow_indices_list) - 1):
                start, end = crow_indices_list[i], crow_indices_list[i + 1]
                these_cols = col_indices_list[start:end]
                these_rows = [i] * len(these_cols)
                rows.extend(these_rows)
                cols.extend(these_cols)
            values = dense_matrix[rows, cols]
            return values

        ctx, grad_output = args
        with torch.no_grad():
            A_values, A_csr, crow_indices, col_indices = ctx.saved_tensors
            grad_A_values = extract_csr_values(col_indices, crow_indices, grad_output)
        return grad_A_values, None, None, None


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
