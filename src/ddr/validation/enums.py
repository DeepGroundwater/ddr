from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ddr.geodatazoo.base_geodataset import BaseGeoDataset
    from ddr.validation.configs import Config


class Mode(StrEnum):
    """The mode DDR is running"""

    TRAINING = "training"
    TESTING = "testing"
    ROUTING = "routing"


class GeoDataset(StrEnum):
    """The geospatial dataset used for predictions and routing"""

    LYNKER_HYDROFABRIC = "lynker_hydrofabric"
    MERIT = "merit"

    def get_dataset_class(self, cfg: "Config") -> "BaseGeoDataset":
        """A factory pattern for instantiating TorchDatasets through config settings"""
        from ddr.geodatazoo.lynker_hydrofabric import LynkerHydrofabric
        from ddr.geodatazoo.merit import Merit

        mapping = {
            GeoDataset.LYNKER_HYDROFABRIC: LynkerHydrofabric,
            GeoDataset.MERIT: Merit,
        }
        return mapping[self](cfg=cfg)
