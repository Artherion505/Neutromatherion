from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


G = 4.300e-6  # kpc (km/s)^2 / M_sun
UPSILON_DISK = 0.5  # M_sun / L_sun at 3.6 um
UPSILON_BULGE = 0.7  # M_sun / L_sun at 3.6 um

# Exploratory Yukawa-like correction; these parameters are not fitted to SPARC.
g_coupling = 3.8e-3
m_N = 0.005  # kpc^-1

SPARC_FILE = Path(__file__).resolve().parent / "data" / "NGC3198_rotmod.dat"
if not SPARC_FILE.is_file():
        raise FileNotFoundError(f"No se encuentra la tabla SPARC: {SPARC_FILE}")

# Official SPARC Rotmod_LTG data for NGC 3198.
# Source: https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip
# Cite Lelli, McGaugh & Schombert (2016), AJ, 152, 157.
sparc_data = np.loadtxt(SPARC_FILE, comments="#")
if sparc_data.ndim != 2 or sparc_data.shape[1] != 8:
        raise ValueError("La tabla SPARC debe tener ocho columnas numericas.")

(r_sparc, v_obs, err_v, v_gas, v_disk, v_bulge,
 surface_brightness_disk, surface_brightness_bulge) = sparc_data.T

# SPARC convention: preserve the sign of Vgas when forming Vbar squared.
v_baryon_squared = (
        v_gas * np.abs(v_gas)
        + UPSILON_DISK * v_disk**2
        + UPSILON_BULGE * v_bulge**2
)
v_baryon_squared = np.maximum(v_baryon_squared, 0.0)
v_baryon = np.sqrt(v_baryon_squared)

# Approximate the baryonic distribution as spherical to evaluate the proposed
# extra force. This is an exploratory approximation, not a SPARC mass fit.
m_b_spherical_equivalent = r_sparc * v_baryon_squared / G
alpha = g_coupling**2 / (4.0 * np.pi)
v_reactio_squared = (
        G
        * m_b_spherical_equivalent
        * alpha
        * (1.0 / r_sparc + m_N)
        * np.exp(-m_N * r_sparc)
)
v_neutromatherion = np.sqrt(np.maximum(v_baryon_squared + v_reactio_squared, 0.0))

plt.style.use(
        "seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default"
)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.errorbar(
        r_sparc,
        v_obs,
        yerr=err_v,
        fmt="o",
        color="#d62728",
        ecolor="#ff9896",
        elinewidth=1.5,
        capsize=3,
        label="Observaciones SPARC: NGC 3198",
        zorder=5,
)
ax.plot(
        r_sparc,
        v_baryon,
        "--",
        color="#1f77b4",
        linewidth=2,
        label=r"Bariones (SPARC, $\Upsilon_{disk}=0.5$)",
)
ax.plot(
        r_sparc,
        v_neutromatherion,
        "-",
        color="#2ca02c",
        linewidth=2.5,
        label="Bariones + correccion Neutromatherion (exploratoria)",
)

ax.set_title("Curva de rotacion SPARC: NGC 3198")
ax.set_xlabel("Radio galactocentrico [kpc]")
ax.set_ylabel("Velocidad circular [km/s]")
ax.set_xlim(0, r_sparc.max() + 1)
ax.set_ylim(bottom=0)
ax.legend(loc="best", frameon=True)
fig.tight_layout()

output_path = Path(__file__).resolve().with_name("curva_rotacion_NGC3198.png")
fig.savefig(output_path, dpi=300)
print(f"Grafica guardada en {output_path}")
plt.show()