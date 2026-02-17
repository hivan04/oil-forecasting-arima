# This script will be run with access to notebook kernel variables
# Fit the ARIMA model with the optimal order found by auto-ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import probplot
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# These variables should already be loaded in the notebook
# best_order, m_train

# Fit the ARIMA model with the optimal order
arima_model = ARIMA(m_train['spread'], order=best_order)
arima_fit = arima_model.fit()

# Print model summary
print(f"\n{'='*60}")
print(f"ARIMA Model Summary - Order {best_order}")
print(f"{'='*60}")
print(arima_fit.summary())

# Extract residuals
residuals = arima_fit.resid

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals over time
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].set_xlabel('Observation')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 2. Histogram with KDE
axes[0, 1].hist(residuals, bins=20, density=True, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
axes[0, 1].legend()

# 3. Q-Q plot
probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# 4. ACF of residuals
plot_acf(residuals, lags=20, ax=axes[1, 1])
axes[1, 1].set_title('ACF of Residuals')

plt.tight_layout()
plt.savefig('/Users/ivanhung/Documents/GitHub/financial-econometrics-cw/residuals_plot.png', dpi=300, bbox_inches='tight')
print("\nResiduals plot saved as 'residuals_plot.png'")
plt.show()

print(f"\n{'='*60}")
print(f"RESIDUAL DIAGNOSTIC TESTS")
print(f"{'='*60}")

# 1. Ljung-Box Test
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
lb_pvalue = lb_test['lb_pvalue'].values[0]
print(f"\nLjung-Box Test (Lag 10):")
print(f"  p-value: {lb_pvalue:.6f}")
if lb_pvalue > 0.05:
    print(f"  ✓ Residuals are independently distributed (fail to reject H0)")
else:
    print(f"  ✗ Residuals show autocorrelation (reject H0)")

# 2. ARCH Test (Heteroscedasticity)
arch_test = het_arch(residuals, nlags=10)
arch_pvalue = arch_test[1]
print(f"\nARCH Test (Lag 10):")
print(f"  p-value: {arch_pvalue:.6f}")
if arch_pvalue > 0.05:
    print(f"  ✓ No heteroscedasticity detected (fail to reject H0)")
else:
    print(f"  ✗ Heteroscedasticity detected (reject H0)")

# 3. Jarque-Bera Test
jb_stat, jb_pvalue = jarque_bera(residuals)
print(f"\nJarque-Bera Test:")
print(f"  p-value: {jb_pvalue:.6f}")
if jb_pvalue > 0.05:
    print(f"  ✓ Residuals are normally distributed (fail to reject H0)")
else:
    print(f"  ✗ Residuals deviate from normality (reject H0)")

# Summary table
print(f"\n{'='*60}")
print(f"SUMMARY OF P-VALUES")
print(f"{'='*60}")
print(f"{'Test':<20} {'P-Value':>15} {'Result':>20}")
print(f"{'-'*55}")
print(f"{'Ljung-Box':<20} {lb_pvalue:>15.6f} {'Pass' if lb_pvalue > 0.05 else 'Fail':>20}")
print(f"{'ARCH':<20} {arch_pvalue:>15.6f} {'Pass' if arch_pvalue > 0.05 else 'Fail':>20}")
print(f"{'Jarque-Bera':<20} {jb_pvalue:>15.6f} {'Pass' if jb_pvalue > 0.05 else 'Fail':>20}")
print(f"{'='*60}")
