"""Tests for ddr.nn.kan — kan neural network."""

import torch

from ddr.nn.kan import kan


class TestKan:
    """Tests for the KAN neural network."""

    def _make_kan(self, input_size: int = 5, hidden_size: int = 11, num_hidden_layers: int = 1):
        return kan(
            input_var_names=[f"attr_{i}" for i in range(input_size)],
            learnable_parameters=["n", "q_spatial"],
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            grid=3,
            k=3,
            seed=42,
            device="cpu",
        )

    def test_kan_output_shape(self) -> None:
        model = self._make_kan()
        inputs = torch.rand(10, 5)
        output = model(inputs=inputs)

        assert isinstance(output, dict)
        assert "n" in output
        assert "q_spatial" in output
        assert output["n"].shape == (10,)
        assert output["q_spatial"].shape == (10,)

    def test_kan_sigmoid_bounds(self) -> None:
        model = self._make_kan()
        inputs = torch.rand(20, 5)
        output = model(inputs=inputs)

        for key in ["n", "q_spatial"]:
            assert (output[key] >= 0).all(), f"{key} has values < 0"
            assert (output[key] <= 1).all(), f"{key} has values > 1"

    def test_kan_deterministic(self) -> None:
        model = self._make_kan()
        model.eval()
        inputs = torch.rand(5, 5)

        with torch.no_grad():
            out1 = model(inputs=inputs)
            out2 = model(inputs=inputs)

        assert torch.equal(out1["n"], out2["n"])
        assert torch.equal(out1["q_spatial"], out2["q_spatial"])

    def test_kan_gradient_flow(self) -> None:
        model = self._make_kan()
        inputs = torch.rand(5, 5)
        output = model(inputs=inputs)

        loss = output["n"].sum() + output["q_spatial"].sum()
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No parameter received gradients"

    def test_kan_multiple_hidden_layers(self) -> None:
        model = self._make_kan(num_hidden_layers=3)
        inputs = torch.rand(5, 5)
        output = model(inputs=inputs)

        assert output["n"].shape == (5,)
        assert output["q_spatial"].shape == (5,)
