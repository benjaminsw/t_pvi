
# Simple mmdfuse replacement for statistical testing
import jax
import jax.numpy as np
from jax.scipy.stats import chi2

def mmdfuse(x, y, key, alpha=0.05):
    """
    Simple two-sample test replacement for mmdfuse
    Returns rejection rate (1 for reject null, 0 for accept)
    
    This is a simplified Maximum Mean Discrepancy test
    """
    n_x, n_y = x.shape[0], y.shape[0]
    
    # Simple kernel MMD test using RBF kernel
    def rbf_kernel(x, y, sigma=1.0):
        diff = x[:, None, :] - y[None, :, :]
        return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))
    
    # Compute MMD statistic
    Kxx = rbf_kernel(x, x)
    Kyy = rbf_kernel(y, y)
    Kxy = rbf_kernel(x, y)
    
    mmd_stat = (np.sum(Kxx) / (n_x * n_x) + 
                np.sum(Kyy) / (n_y * n_y) - 
                2 * np.sum(Kxy) / (n_x * n_y))
    
    # Simple threshold-based test (approximate)
    # In practice, this should use permutation testing
    threshold = 0.1  # Simplified threshold
    
    return float(mmd_stat > threshold)
