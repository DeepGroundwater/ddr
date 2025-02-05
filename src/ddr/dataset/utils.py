import torch
import torch.nn.functional as F

def downsample(data: torch.Tensor, rho: int) -> torch.Tensor:
    """Downsamples from hourly to daily data using torch.nn.functional.interpolate

    Parameters
    ----------
    data : torch.Tensor
        The data to downsample
    rho : int
        The number of days to downsample to

    Returns
    -------
    torch.Tensor
        The downsampled daily data
    """
    downsampled_data = F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)
    return downsampled_data
