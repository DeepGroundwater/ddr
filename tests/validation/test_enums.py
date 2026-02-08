"""Tests for ddr.validation.enums."""

from ddr.validation.enums import GeoDataset, Mode


class TestEnums:
    """Test enum values."""

    def test_mode_values(self) -> None:
        assert Mode.TRAINING == "training"
        assert Mode.TESTING == "testing"
        assert Mode.ROUTING == "routing"

    def test_geodataset_values(self) -> None:
        assert GeoDataset.LYNKER_HYDROFABRIC == "lynker_hydrofabric"
        assert GeoDataset.MERIT == "merit"
