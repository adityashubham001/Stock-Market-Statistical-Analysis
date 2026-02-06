import matplotlib.pyplot as plt
import numpy as np


# =========================
# MONTE CARLO PLOTTING
# =========================

def plot_monte_carlo_paths(                                     # This function plots the simulated price paths
    price_paths: np.ndarray,                                    #     from the Monte Carlo simulation
    n_paths_to_plot: int = 100                                  #     and the number of paths to plot
):                                                              #     (default is 100)
  
    n_simulations = price_paths.shape[0]                        # Count number of simulations
    n_plot = min(n_paths_to_plot, n_simulations)                # Determine number of paths to plot

    fig, ax = plt.subplots(figsize=(10, 6))                     # Create figure & axis

    for i in range(n_plot):                                     # Plot each path
        ax.plot(price_paths[i], alpha=0.7)                      #     with transparency

    ax.set_title("Monte Carlo Simulation — 1-Year Price Paths") # Set title
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Price")
    ax.grid(True)

    fig.tight_layout()

    return fig                                                     # Return figure
