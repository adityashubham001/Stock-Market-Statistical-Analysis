import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



############### DISTRIBUTION PLOTTING ################


def plot_return_distributions(returns, bins=50):                # This function creates a plot comparing normal and Student-t distributions fitted to the returns
    r = returns.dropna()                                    # Clean the returns by dropping NaN values

    # Fit distributions
    mu, sigma = stats.norm.fit(r)                           # Fit normal distribution   
    df, loc, scale = stats.t.fit(r)                         # Fit Student-t distribution                    

    # Prepare data for plotting
    
    # Use percentile range to avoid extreme outliers dominating the plot
    
    q_low, q_high = np.percentile(r, [1, 99])                   #   
    x = np.linspace(q_low, q_high, 500)
                

    pdf_normal = stats.norm.pdf(x, mu, sigma)                # Calculate normal PDF
    pdf_t = stats.t.pdf(x, df, loc, scale)                  # Calculate Student-t PDF   

    # Create figure & axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram & PDFs
    ax.hist(r, bins=bins, density=True, alpha=0.5, label="Returns") # Histogram of returns                  
    ax.plot(                                                                # Plot PDFs 
        x, pdf_normal, linewidth=2,                                         # PDF of normal distribution
        label=f"Normal (μ={mu:.4f}, σ={sigma:.4f})"                         
    )

    ax.plot(                                                    #   PDF of Student-t distribution
        x, pdf_t, linewidth=2,                        
        label=f"Student-t (df={df:.1f})"                        
    )
        

    # Styling
    ax.set_title("Return Distribution: Normal vs Student-t")            
    ax.set_xlabel("Returns")
    ax.set_ylabel("Density")
    ax.legend()                             # Add legend 
    ax.grid(True)                           # Add grid

    fig.tight_layout()

    return fig
