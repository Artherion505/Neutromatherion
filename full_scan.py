import numpy as np
import csv

# Constantes físicas
c = 2.99792458e8
J_per_GeV = 1.602176634e-10
GeV_per_J = 1.0 / J_per_GeV
m_to_GeVinv = 5.0677307e15
Mpl_GeV = 2.435e18
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.9634e8
Phi_sun = G * M_sun / R_sun

def rho_kgm3_to_GeV4(rho_kgm3):
    eps_Jm3 = rho_kgm3 * c**2
    eps_GeV_per_m3 = eps_Jm3 * GeV_per_J
    return eps_GeV_per_m3 * (m_to_GeVinv**3)

rho_in = 1e3
rho_out = 1e-21
rho_in_GeV4 = rho_kgm3_to_GeV4(rho_in)
rho_out_GeV4 = rho_kgm3_to_GeV4(rho_out)

# --- AQUÍ está la función que faltaba ---
def solve_N0(rho, g, lam, m0):
    # ecuación: lam*N^3 + m0^2*N + g*rho = 0
    coeffs = [lam, 0, m0**2, g*rho]
    roots = np.roots(coeffs)
    # elegimos raíz real negativa
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    if not real_roots:
        return None
    return min(real_roots)  # la más negativa

# Parámetro g fijo
g = 1e-20

# Rango de parámetros
lams = np.logspace(-12, 0, 20)   # λ hasta 1
m0_eVs = np.logspace(-12, 0, 20) # m0 hasta 1 eV

rows = []
for lam in lams:
    for m0_eV in m0_eVs:
        m0 = m0_eV*1e-9
        N_in = solve_N0(rho_in_GeV4, g, lam, m0)
        N_out = solve_N0(rho_out_GeV4, g, lam, m0)
        if N_in is None or N_out is None:
            continue
        DeltaR_over_R = (N_out - N_in)/(6*g*Mpl_GeV*Phi_sun)
        gamma_minus_1 = 2*(g*abs(N_out)/Mpl_GeV)*max(1.0,3*abs(DeltaR_over_R))
        rows.append([lam, m0_eV, DeltaR_over_R, gamma_minus_1])

with open("scan_results.csv","w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda","m0_eV","DeltaR/R","|gamma-1|"])
    writer.writerows(rows)

print(f"Guardados {len(rows)} resultados en scan_results.csv")
