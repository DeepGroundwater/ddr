"""Bundle contract: the real library readers open the Juniata bundle."""

from pathlib import Path

import numpy as np
import pytest
import torch

BUNDLE = Path(__file__).parents[2] / "examples" / "juniata" / "data"
needs_bundle = pytest.mark.skipif(
    not (BUNDLE / "juniata_gage.csv").exists(),
    reason="Juniata bundle not present — run examples/juniata/extract_bundle.py",
)


@needs_bundle
class TestBundleContract:
    def test_merit_builds_routing_dataclass(self) -> None:
        from examples.juniata.train_and_test import make_config

        from ddr.validation.enums import GeoDataset

        cfg = make_config(bundle_dir=BUNDLE)
        dataset = GeoDataset.get_dataset_class(cfg=cfg)
        batch = dataset.collate_fn(list(dataset.gage_ids))
        n = batch.adjacency_matrix.shape[0]
        assert n == 213
        assert batch.length.shape == (n,)
        assert batch.slope.shape == (n,)
        assert batch.normalized_spatial_attributes.shape == (n, 10)
        assert torch.isfinite(batch.normalized_spatial_attributes).all()
        # Gauge prediction extracts the gauge reach itself (post-#192 semantics)
        assert len(batch.outflow_idx) == 1 and batch.outflow_idx[0].shape == (1,)

    def test_streamflow_reader_returns_hourly_tensor(self) -> None:
        from examples.juniata.train_and_test import make_config

        from ddr import streamflow
        from ddr.validation.enums import GeoDataset

        cfg = make_config(bundle_dir=BUNDLE)
        dataset = GeoDataset.get_dataset_class(cfg=cfg)
        batch = dataset.collate_fn(list(dataset.gage_ids))
        flow = streamflow(cfg)
        q_prime = flow(routing_dataclass=batch)
        assert q_prime.shape[1] == 213
        assert q_prime.shape[0] == len(batch.dates.batch_hourly_time_range)
        assert (q_prime >= 0).all() and torch.isfinite(q_prime).all()

    def test_observations_aligned(self) -> None:
        from examples.juniata.train_and_test import make_config

        from ddr.validation.enums import GeoDataset

        cfg = make_config(bundle_dir=BUNDLE)
        dataset = GeoDataset.get_dataset_class(cfg=cfg)
        batch = dataset.collate_fn(list(dataset.gage_ids))
        obs = batch.observations.streamflow.values
        assert obs.shape[0] == 1
        assert np.isfinite(obs).mean() > 0.9  # Juniata record is nearly complete
