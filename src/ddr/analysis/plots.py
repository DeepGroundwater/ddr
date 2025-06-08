import matplotlib.pyplot as plt


def plot_time_series(prediction, observation, time_range, gage_id, name, metrics, path, warmup=3):
    """Plot time series for a single prediction

    Parameters
    ----------
    prediction : np.ndarray
        The predicted values
    observation : np.ndarray
        The observed values
    time_range : np.ndarray
        The time range
    gage_id : str
        The gage ID
    name : str
        The name of the gage
    mode : str
        The mode
    metrics : Dict
        The metrics
    """
    fig = plt.figure(figsize=(10, 5))
    prediction_to_plot = prediction[warmup:]
    observation_to_plot = observation[warmup:]
    plt.plot(time_range[warmup:], observation_to_plot, label="Observation")
    plt.plot(time_range[warmup:], prediction_to_plot, label="Routed Streamflow")
    nse = metrics["nse"]
    plt.title(f"Train time period Hydrograph - " f"GAGE ID: {gage_id} - Name: {name}")
    plt.xlabel("Time (hours)")
    plt.ylabel(r"Discharge $m^3/s$")
    plt.legend(title=f"NSE: {nse:.4f}")
    plt.savefig(path)
    plt.close(fig)
