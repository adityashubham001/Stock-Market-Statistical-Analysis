import matplotlib.pyplot as plt
import numpy as np


# =========================
# MONTE CARLO PLOTTING
# =========================

def plot_monte_carlo_paths(
    price_paths: np.ndarray,
    n_paths_to_plot: int = 100
):
    """
    Create a Monte Carlo price path plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """

    n_simulations = price_paths.shape[0]
    n_plot = min(n_paths_to_plot, n_simulations)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n_plot):
        ax.plot(price_paths[i], alpha=0.7)

    ax.set_title("Monte Carlo Simulation — 1-Year Price Paths")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Price")
    ax.grid(True)

    fig.tight_layout()

    return fig
