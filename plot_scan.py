# plot_scan.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_heatmap(input_csv=None, output_png=None):
        base_dir = Path(__file__).resolve().parent
        input_csv = Path(input_csv) if input_csv else base_dir / "scan_results.csv"
        output_png = Path(output_png) if output_png else base_dir / "heatmap_scan_corrected.png"
        df = pd.read_csv(input_csv)

        required = {"lambda", "m0_eV", "DeltaR_over_R", "epsilon_gamma_proxy"}
        missing = required.difference(df.columns)
        if missing:
                raise ValueError(f"Solar scan is missing columns: {sorted(missing)}")

        df["log_lambda"] = np.log10(df["lambda"])
        df["log_m0"] = np.log10(df["m0_eV"])
        df["log_delta"] = np.log10(df["DeltaR_over_R"].clip(lower=1e-300))
        df["log_gamma_proxy"] = np.log10(
                df["epsilon_gamma_proxy"].clip(lower=1e-300)
        )
        pivot_delta = df.pivot_table(
                index="log_m0", columns="log_lambda", values="log_delta"
        )
        pivot_gamma = df.pivot_table(
                index="log_m0", columns="log_lambda", values="log_gamma_proxy"
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, pivot, title, colorbar_label in (
                (axes[0], pivot_delta, "Thin-shell diagnostic", "log10(DeltaR/R_sun)"),
                (axes[1], pivot_gamma, "Long-range PPN proxy", "log10(epsilon_gamma_proxy)"),
        ):
                image = ax.imshow(
                        pivot.to_numpy(),
                        aspect="auto",
                        origin="lower",
                        extent=[
                                pivot.columns.min(),
                                pivot.columns.max(),
                                pivot.index.min(),
                                pivot.index.max(),
                        ],
                )
                ax.set_title(title)
                ax.set_xlabel("log10(lambda)")
                ax.set_ylabel("log10(m0 [eV])")
                fig.colorbar(image, ax=ax, label=colorbar_label)

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved to {output_png}")


if __name__ == "__main__":
        create_heatmap()
