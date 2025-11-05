import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

# Parameters
theta = 1.0
r = theta / 2
n_samples = 10000
l_sweep = np.linspace(theta, 1.5 * theta, 30)
dims_to_test = [2, 5, 10, 15, 20, 50]
# np.random.seed(42)

# Function: Uniform sampling in d-dimensional ball
def generate_uniform_ball(center, radius, n, d):
    u = np.random.normal(0, 1, size=(n, d))
    u /= np.linalg.norm(u, axis=1)[:, None]
    radii = np.random.rand(n) ** (1 / d)
    return center + radius * radii[:, None] * u

# Step 1: Estimate p_theta(ell) using Gaussian approximation
p_theta_sweep_results = {}
for d in dims_to_test:
    p_vals = []
    for ell in l_sweep:
        c1 = np.zeros(d)
        c2 = np.zeros(d)
        c2[0] = ell
        B1 = generate_uniform_ball(c1, r, n_samples, d)
        B2 = generate_uniform_ball(c2, r, n_samples, d)
        idx = np.random.choice(n_samples, size=1000, replace=False)
        D = cdist(B1[idx], B2[idx]).flatten()
        mu = np.mean(D)
        sigma = np.std(D)
        p = norm.cdf((theta - mu) / sigma)
        p_vals.append(p)
    p_theta_sweep_results[d] = p_vals


# Step 3: Fit logistic model: p ≈ L / (1 + exp(k * (ell - x0)))
def logistic_model(ell, L, k, x0):
    return L / (1 + np.exp(k * (ell - x0)))

fitted_logistic = {}
for d in dims_to_test:
    y_data = np.array(p_theta_sweep_results[d])
    try:
        popt, _ = curve_fit(logistic_model, l_sweep, y_data, p0=(1.0, 10.0, theta), bounds=(0, np.inf))
        fitted_logistic[d] = {"L": popt[0], "k": popt[1], "x0": popt[2]}
    except RuntimeError:
        fitted_logistic[d] = {"L": np.nan, "k": np.nan, "x0": np.nan}

# Step 4: Plot comparison
plt.figure(figsize=(12, 8))
for i, d in enumerate(dims_to_test):
    # Gaussian approximation (empirical)
    y_emp = np.array(p_theta_sweep_results[d])
    plt.plot(l_sweep, y_emp, linestyle='-', label=f'Gaussian approx (d={d})')

    runner = 0.
    # Logistic fit
    if not np.isnan(fitted_logistic[d]["L"]):
        L = fitted_logistic[d]["L"]
        k = fitted_logistic[d]["k"]
        x0 = fitted_logistic[d]["x0"]
        y_log = logistic_model(l_sweep, L, k, x0)
        plt.plot(l_sweep, y_log, linestyle='--', label=f'Logistic fit (d={d})')
        mid_idx = len(l_sweep) // 2
        runner -= 0.05 * i  # Adjust runner for spacing
        plt.text(l_sweep[mid_idx] + runner, y_log[mid_idx] + 0.01, f'd={d}', fontsize=10, ha='center', va='bottom')

plt.axvline(theta, color='gray', linestyle='--', label=r'$\ell = \theta$')
plt.xlabel(r'$\ell$ (Distance between centers)')
plt.ylabel(r'$p_\theta(\ell)$')
plt.title('Gaussian Approximation vs Logistic Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


