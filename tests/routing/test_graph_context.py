"""Tests for graph-context neighbor attribute aggregation."""

import torch

from ddr.routing.utils import aggregate_neighbor_attributes


class TestAggregateNeighborAttributes:
    """Tests for aggregate_neighbor_attributes()."""

    def _make_adjacency(self, indices: list[list[int]], values: list[float], n: int) -> torch.Tensor:
        """Create a sparse CSR adjacency matrix from COO-style inputs."""
        if len(indices[0]) == 0:
            return torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                size=(n, n),
            ).to_sparse_csr()
        idx = torch.tensor(indices, dtype=torch.long)
        vals = torch.tensor(values, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, vals, size=(n, n)).to_sparse_csr()

    def test_simple_chain(self) -> None:
        """3-node chain: 0 -> 1 -> 2.  N[1,0]=1, N[2,1]=1.

        Node 0: headwater (no upstream) -> returns own attrs
        Node 1: one upstream (node 0) -> returns attrs of node 0
        Node 2: one upstream (node 1) -> returns attrs of node 1
        """
        adj = self._make_adjacency([[1, 2], [0, 1]], [1.0, 1.0], 3)
        attrs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = aggregate_neighbor_attributes(attrs, adj)

        # Node 0: headwater -> own attrs [1, 2]
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
        # Node 1: upstream is node 0 -> [1, 2]
        assert torch.allclose(result[1], torch.tensor([1.0, 2.0]))
        # Node 2: upstream is node 1 -> [3, 4]
        assert torch.allclose(result[2], torch.tensor([3.0, 4.0]))

    def test_confluence(self) -> None:
        """Confluence: nodes 0 and 1 both flow into node 2.

        N[2,0]=1, N[2,1]=1. Node 2 should get mean of nodes 0 and 1.
        """
        adj = self._make_adjacency([[2, 2], [0, 1]], [1.0, 1.0], 3)
        attrs = torch.tensor([[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]])
        result = aggregate_neighbor_attributes(attrs, adj)

        # Nodes 0 and 1: headwaters -> own attrs
        assert torch.allclose(result[0], torch.tensor([2.0, 4.0]))
        assert torch.allclose(result[1], torch.tensor([6.0, 8.0]))
        # Node 2: mean of nodes 0 and 1 -> [4, 6]
        assert torch.allclose(result[2], torch.tensor([4.0, 6.0]))

    def test_all_headwaters(self) -> None:
        """All nodes are headwaters (no edges) -> all return own attrs."""
        adj = self._make_adjacency([[], []], [], 3)
        attrs = torch.tensor([[1.0], [2.0], [3.0]])
        result = aggregate_neighbor_attributes(attrs, adj)
        assert torch.allclose(result, attrs)

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        adj = self._make_adjacency([[1], [0]], [1.0], 4)
        attrs = torch.randn(4, 7)
        result = aggregate_neighbor_attributes(attrs, adj)
        assert result.shape == attrs.shape

    def test_gradients_flow(self) -> None:
        """Verify gradients flow through the aggregation."""
        adj = self._make_adjacency([[1, 2], [0, 1]], [1.0, 1.0], 3)
        attrs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        result = aggregate_neighbor_attributes(attrs, adj)
        loss = result.sum()
        loss.backward()
        assert attrs.grad is not None
        # All attributes should receive some gradient
        assert (attrs.grad != 0).any()
