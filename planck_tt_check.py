from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "COM_PowerSpect_CMB-TT-binned_R3.01.txt"
)


def load_planck_tt(path: Path = DATA_FILE) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Planck TT data file not found: {path}")
    values = np.loadtxt(path, comments="#")
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Expected columns: ell, Dl, -dDl, +dDl, BestFit.")
    if not np.isfinite(values).all():
        raise ValueError("Planck TT table contains non-finite values.")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Multipole bins must be strictly increasing.")
    if np.any(values[:, 2:4] <= 0):
        raise ValueError("Planck TT uncertainties must be positive.")
    return values


def run_check(output_dir: Path | None = None):
    output_dir = output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    values = load_planck_tt()
    ell, dl, err_low, err_high, best_fit = values.T
    residual = dl - best_fit
    sigma_for_pull = np.where(residual >= 0, err_high, err_low)
    pulls = residual / sigma_for_pull
    diagonal_chi2 = float(np.sum(pulls**2))
    peaks = [
        (ell[index], dl[index])
        for index in range(1, len(ell) - 1)
        if dl[index] > dl[index - 1]
        and dl[index] >= dl[index + 1]
        and ell[index] < 1200
    ]

    residual_path = output_dir / "planck_tt_residuals.csv"
    with residual_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["ell", "Dl_uK2", "error_low_uK2", "error_high_uK2", "bestfit_uK2", "pull"]
        )
        writer.writerows(
            zip(ell, dl, err_low, err_high, best_fit, pulls, strict=True)
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        height_ratios=(3, 1),
        constrained_layout=True,
    )
    axes[0].errorbar(
        ell,
        dl,
        yerr=np.vstack((err_low, err_high)),
        fmt=".",
        markersize=4,
        color="#202020",
        ecolor="#858585",
        linewidth=0.7,
        label="Planck 2018 binned TT (R3.01)",
    )
    axes[0].plot(ell, best_fit, color="#c43c39", linewidth=1.5, label="Planck best-fit spectrum")
    axes[0].set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.2)

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(2.0, color="#777777", linewidth=0.7, linestyle="--")
    axes[1].axhline(-2.0, color="#777777", linewidth=0.7, linestyle="--")
    axes[1].plot(ell, pulls, ".", color="#315a84", markersize=4)
    axes[1].set_xlabel(r"Multipole $\ell$")
    axes[1].set_ylabel("Residual / error")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Planck 2018 temperature spectrum: data and official best fit")
    fig.savefig(output_dir / "planck_tt_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Bins: {len(ell)}; ell range: {ell.min():.1f}-{ell.max():.1f}")
    print(f"Diagonal descriptive chi2/bin (not a likelihood): {diagonal_chi2 / len(ell):.6f}")
    print("First local TT maxima below ell=1200:")
    for peak_ell, peak_dl in peaks[:5]:
        print(f"  ell={peak_ell:.1f}, Dl={peak_dl:.2f} uK^2")
    print("No Neutromatherion fit was computed: its TT perturbation prediction is not specified.")
    print(f"Saved {residual_path} and planck_tt_comparison.png")


if __name__ == "__main__":
    run_check()
