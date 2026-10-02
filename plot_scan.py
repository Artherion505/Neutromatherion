# plot_scan.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar CSV
df = pd.read_csv("scan_results.csv")

# Convertir a log10 para graficar
df["log_lambda"] = np.log10(df["lambda"])
df["log_m0"] = np.log10(df["m0_eV"])
df["log_DeltaR"] = np.log10(np.abs(df["DeltaR/R"]))
df["log_gamma"] = np.log10(df["|gamma-1|"])

# Pivotear para mapas
pivot_DR = df.pivot_table(index="log_m0", columns="log_lambda", values="log_DeltaR")
pivot_gam = df.pivot_table(index="log_m0", columns="log_lambda", values="log_gamma")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Mapa de ΔR/R_sun (log10)")
plt.imshow(pivot_DR, aspect="auto", origin="lower",
           extent=[pivot_DR.columns.min(), pivot_DR.columns.max(),
                   pivot_DR.index.min(), pivot_DR.index.max()])
plt.colorbar(label="log10(ΔR/R)")
plt.xlabel("log10 λ")
plt.ylabel("log10 m0 [eV]")

plt.subplot(1,2,2)
plt.title("Mapa de |γ-1| (log10)")
plt.imshow(pivot_gam, aspect="auto", origin="lower",
           extent=[pivot_gam.columns.min(), pivot_gam.columns.max(),
                   pivot_gam.index.min(), pivot_gam.index.max()])
plt.colorbar(label="log10(|γ-1|)")
plt.xlabel("log10 λ")
plt.ylabel("log10 m0 [eV]")

plt.tight_layout()
plt.show()
