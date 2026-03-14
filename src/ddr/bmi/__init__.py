"""BMI wrapper for DDR differentiable Muskingum-Cunge routing.

Provides a BMI v2.0 (CSDMS) interface for integration with the NGWPC/ngen
NextGen Water Resources Modeling Framework as a drop-in replacement for t-route.
"""

from .ddr_bmi import DdrBmi

__all__ = ["DdrBmi"]
