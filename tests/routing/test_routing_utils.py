"""Tests for ddr.routing.utils — denormalize, PatternMapper, get_network_idx,
TriangularSparseSolver, and _fill_row_indices_vectorized."""

import torch

from ddr.routing.utils import (
    PatternMapper,
    _fill_row_indices_vectorized,
    denormalize,
    get_network_idx,
    triangular_sparse_solve,
)


# ---------------------------------------------------------------------------
# denormalize()
# ---------------------------------------------------------------------------
class TestDenormalize:
    """Tests for denormalize()."""

    def test_denormalize_linear_midpoint(self) -> None:
        result = denormalize(torch.tensor(0.5), [0.0, 10.0])
        assert torch.isclose(result, torch.tensor(5.0))

    def test_denormalize_linear_bounds(self) -> None:
        low = denormalize(torch.tensor(0.0), [0.0, 10.0])
        high = denormalize(torch.tensor(1.0), [0.0, 10.0])
        assert torch.isclose(low, torch.tensor(0.0))
        assert torch.isclose(high, torch.tensor(10.0))

    def test_denormalize_log_space(self) -> None:
        result = denormalize(torch.tensor(0.5), [1.0, 100.0], log_space=True)
        # Geometric mean of 1 and 100 = 10
        assert torch.isclose(result, torch.tensor(10.0), atol=torch.tensor(0.5))

    def test_denormalize_log_space_bounds(self) -> None:
        low = denormalize(torch.tensor(0.0), [1.0, 100.0], log_space=True)
        high = denormalize(torch.tensor(1.0), [1.0, 100.0], log_space=True)
        assert torch.isclose(low, torch.tensor(1.0), atol=torch.tensor(0.01))
        assert torch.isclose(high, torch.tensor(100.0), atol=torch.tensor(0.1))

    def test_denormalize_preserves_gradient(self) -> None:
        x = torch.tensor(0.5, requires_grad=True)
        y = denormalize(x, [0.0, 10.0])
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad)

    def test_denormalize_vector_input(self) -> None:
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = denormalize(x, [0.0, 10.0])
        expected = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# PatternMapper
# ---------------------------------------------------------------------------
class TestPatternMapper:
    """Tests for PatternMapper."""

    def _make_identity_fill(self, n: int):
        """Create a PatternMapper using identity fill."""

        def fill_op(data_vector: torch.Tensor) -> torch.Tensor:
            return torch.diag(data_vector).to_sparse_csr()

        return PatternMapper(fill_op, n, device="cpu")

    def test_pattern_mapper_identity_fill(self) -> None:
        mapper = self._make_identity_fill(3)
        assert mapper.crow_indices is not None
        assert mapper.col_indices is not None

    def test_pattern_mapper_map_function(self) -> None:
        mapper = self._make_identity_fill(3)
        data = torch.tensor([10.0, 20.0, 30.0])
        mapped = mapper.map(data)
        # For identity fill, mapping should reconstruct the diagonal values
        assert mapped.shape[0] == 3

    def test_pattern_mapper_get_sparse_indices(self) -> None:
        mapper = self._make_identity_fill(4)
        crow, col = mapper.getSparseIndices()
        assert isinstance(crow, torch.Tensor)
        assert isinstance(col, torch.Tensor)

    def test_pattern_mapper_inverse_diag_fill(self) -> None:
        data = torch.tensor([1.0, 2.0, 3.0])
        result = PatternMapper.inverse_diag_fill(data)
        assert result.shape == (3, 3)
        assert result[0, 0].item() == 3.0
        assert result[2, 2].item() == 1.0


# ---------------------------------------------------------------------------
# get_network_idx()
# ---------------------------------------------------------------------------
class TestGetNetworkIdx:
    """Tests for get_network_idx()."""

    def test_get_network_idx_linear_chain(self) -> None:
        """Build a small 3-node linear chain and verify adjacency."""
        from ddr.routing.mmc import MuskingumCunge
        from tests.routing.test_utils import create_mock_config, create_mock_routing_dataclass

        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hf = create_mock_routing_dataclass(num_reaches=3)
        mc.network = hf.adjacency_matrix
        mapper, rows, cols = mc.create_pattern_mapper()

        rows_out, cols_out = get_network_idx(mapper)
        assert isinstance(rows_out, torch.Tensor)
        assert isinstance(cols_out, torch.Tensor)
        assert rows_out.shape == cols_out.shape


# ---------------------------------------------------------------------------
# TriangularSparseSolver (CPU integration)
# ---------------------------------------------------------------------------
class TestTriangularSparseSolver:
    """Tests for TriangularSparseSolver on CPU."""

    def _build_identity_system(self, n: int):
        """Build I*x=b as a CSR-based system."""
        crow_indices = torch.arange(n + 1, dtype=torch.int32)
        col_indices = torch.arange(n, dtype=torch.int32)
        A_values = torch.ones(n, dtype=torch.float32)
        return A_values, crow_indices, col_indices

    def test_solve_identity_system(self) -> None:
        A_values, crow, col = self._build_identity_system(5)
        b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        x = triangular_sparse_solve(A_values, crow, col, b, True, False, "cpu")
        assert torch.allclose(x, b, atol=1e-5)

    def test_solve_known_triangular_system(self) -> None:
        # L = [[2, 0, 0],
        #      [1, 3, 0],
        #      [0, 1, 4]]
        # b = [2, 7, 13] → x = [1, 2, 2.75]
        crow_indices = torch.tensor([0, 1, 3, 5], dtype=torch.int32)
        col_indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
        A_values = torch.tensor([2.0, 1.0, 3.0, 1.0, 4.0])
        b = torch.tensor([2.0, 7.0, 13.0])

        x = triangular_sparse_solve(A_values, crow_indices, col_indices, b, True, False, "cpu")

        expected = torch.tensor([1.0, 2.0, 2.75])
        assert torch.allclose(x, expected, atol=1e-5)

    def test_backward_produces_finite_grads(self) -> None:
        crow_indices = torch.tensor([0, 1, 3, 5], dtype=torch.int32)
        col_indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
        A_values = torch.tensor([2.0, 1.0, 3.0, 1.0, 4.0], requires_grad=True)
        b = torch.tensor([2.0, 7.0, 13.0], requires_grad=True)

        x = triangular_sparse_solve(A_values, crow_indices, col_indices, b, True, False, "cpu")
        loss = x.sum()
        loss.backward()

        assert A_values.grad is not None
        assert torch.isfinite(A_values.grad).all()
        assert b.grad is not None
        assert torch.isfinite(b.grad).all()


# ---------------------------------------------------------------------------
# _fill_row_indices_vectorized()
# ---------------------------------------------------------------------------
class TestFillRowIndicesVectorized:
    """Tests for _fill_row_indices_vectorized()."""

    def test_fill_row_indices_known_csr(self) -> None:
        crow_indices = torch.tensor([0, 2, 3, 5])
        row_indices = torch.empty(5, dtype=torch.long)
        _fill_row_indices_vectorized(crow_indices, row_indices)
        expected = torch.tensor([0, 0, 1, 2, 2])
        assert torch.equal(row_indices, expected)
