import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict, PositiveFloat


class Gauge(BaseModel):
    """A pydantic object for managing properties for a Gauge and validating incoming CSV files"""

    model_config = ConfigDict(extra="ignore")
    STAID: str
    DRAIN_SQKM: PositiveFloat


class GaugeSet(BaseModel):
    """A pydantic object for storing a list of Gauges"""

    gauges: list[Gauge]


def validate_gages(file_path: Path) -> GaugeSet:
    """A function to read the training gauges file and validate based on a pydantic schema

    Parameters
    ----------
    file_path: Path
        The path to the gauges csv file

    Returns
    -------
    GaugeSet
        A set of pydantic-validated gauges
    """
    with file_path.open() as f:
        reader = csv.DictReader(f)
        gauges = [Gauge.model_validate(row) for row in reader]
        return GaugeSet(gauges=gauges)
