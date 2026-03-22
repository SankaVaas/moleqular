import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                                y_std: np.ndarray, n_bins: int = 10) -> dict:
    """
    Expected Calibration Error (ECE) for regression.
    
    Theory:
    A model is CALIBRATED if its predicted confidence intervals
    contain the true value the expected fraction of the time.
    
    For a perfectly calibrated model:
        P(|y - μ| ≤ z·σ) = Φ(z) - Φ(-z)
    
    where Φ is the standard normal CDF.
    
    ECE measures the gap between predicted and empirical coverage.
    Lower ECE = better calibrated uncertainty estimates.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted mean values  
        y_std:  Predicted standard deviations (uncertainty)
    
    Returns:
        ece:              scalar ECE value
        confidence_levels: expected coverage levels
        empirical_coverage: actual coverage levels
    """
    # Standardized residuals
    z = (y_true - y_pred) / (y_std + 1e-8)

    # Expected confidence levels
    confidence_levels = np.linspace(0.01, 0.99, n_bins)
    empirical_coverage = []

    for conf in confidence_levels:
        # z-score for this confidence level
        z_score = stats.norm.ppf((1 + conf) / 2)
        # Fraction of predictions within this interval
        coverage = np.mean(np.abs(z) <= z_score)
        empirical_coverage.append(coverage)

    empirical_coverage = np.array(empirical_coverage)
    ece = np.mean(np.abs(empirical_coverage - confidence_levels))

    return {
        "ece": ece,
        "confidence_levels": confidence_levels,
        "empirical_coverage": empirical_coverage,
        "is_overconfident": empirical_coverage.mean() < confidence_levels.mean()
    }


def sharpness(y_std: np.ndarray) -> float:
    """
    Sharpness: average predicted uncertainty.
    Lower = sharper (more confident) predictions.
    A calibrated model should be as sharp as possible while remaining calibrated.
    """
    return float(np.mean(y_std))


def uncertainty_quality_report(y_true: np.ndarray, y_pred: np.ndarray,
                                y_std: np.ndarray) -> dict:
    """
    Full uncertainty quality report combining calibration + sharpness.
    
    A good uncertainty estimate is:
    1. Calibrated: ECE close to 0
    2. Sharp: small average std (not just predicting wide intervals)
    3. Ordered: higher uncertainty for harder predictions
    """
    cal = expected_calibration_error(y_true, y_pred, y_std)
    sharp = sharpness(y_std)

    # Spearman correlation between uncertainty and absolute error
    # Good UQ: higher uncertainty should correlate with higher error
    abs_errors = np.abs(y_true - y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_std, abs_errors)

    # NLL (Negative Log Likelihood) under Gaussian assumption
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * y_std**2 + 1e-8) +
        (y_true - y_pred)**2 / (y_std**2 + 1e-8)
    )

    report = {
        "ece": cal["ece"],
        "sharpness_mean_std": sharp,
        "spearman_r_uncertainty_error": float(spearman_r),
        "spearman_p": float(spearman_p),
        "nll_gaussian": float(nll),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred)**2))),
        "calibration_data": cal,
        "verdict": "well-calibrated" if cal["ece"] < 0.1 else "poorly-calibrated"
    }

    return report


def plot_calibration(report: dict, save_path: str = None):
    """Plot calibration curve and uncertainty vs error."""
    cal = report["calibration_data"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#1e2a3a')

    # 1. Calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], '--', color='#444', label='Perfect calibration')
    ax.plot(cal["confidence_levels"], cal["empirical_coverage"],
            color='#4488ff', linewidth=2, label=f'Model (ECE={report["ece"]:.3f})')
    ax.fill_between(cal["confidence_levels"], cal["confidence_levels"],
                    cal["empirical_coverage"],
                    alpha=0.2, color='#4488ff')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Calibration Curve')
    ax.legend(facecolor='#1e2a3a', labelcolor='#c9d1d9')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2. Uncertainty vs Error scatter
    ax = axes[1]
    ax.set_xlabel('Predicted Std (meV)')
    ax.set_ylabel('Absolute Error (meV)')
    ax.set_title(f'Uncertainty vs Error\n(Spearman r={report["spearman_r_uncertainty_error"]:.3f})')
    ax.text(0.05, 0.95, f'r = {report["spearman_r_uncertainty_error"]:.3f}',
            transform=ax.transAxes, color='#00ff88', fontsize=12)

    # 3. Metrics summary
    ax = axes[2]
    ax.axis('off')
    metrics = [
        ('ECE', f'{report["ece"]:.4f}'),
        ('MAE', f'{report["mae"]:.2f} meV'),
        ('RMSE', f'{report["rmse"]:.2f} meV'),
        ('Sharpness', f'{report["sharpness_mean_std"]:.4f}'),
        ('NLL', f'{report["nll_gaussian"]:.4f}'),
        ('Spearman r', f'{report["spearman_r_uncertainty_error"]:.4f}'),
        ('Verdict', report["verdict"]),
    ]
    y_pos = 0.9
    for label, value in metrics:
        color = '#00ff88' if label == 'Verdict' and 'well' in value else '#c9d1d9'
        ax.text(0.1, y_pos, f'{label}:', color='#888', fontsize=11)
        ax.text(0.55, y_pos, value, color=color, fontsize=11, fontweight='bold')
        y_pos -= 0.12
    ax.set_title('Uncertainty Quality Metrics')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
        print(f"Saved calibration plot to {save_path}")
    plt.show()
    return fig