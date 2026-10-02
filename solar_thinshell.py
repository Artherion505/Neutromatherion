# scan_thinshell.py
import numpy as np
import csv
from math import log10
from scipy.optimize import brentq

# --- conversions and constants (same as corrected script) ---
c = 299792458.0
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
    eps_GeV4 = eps_GeV_per_m3 * (m_to_GeVinv**3)
    return eps_GeV4

rho_in = 1.0 * 1000.0      # kg/m^3
rho_out = 1e-24 * 1000.0  # kg/m^3
rho_in_GeV4 = rho_kgm3_to_GeV4(rho_in)
rho_out_GeV4 = rho_kgm3_to_GeV4(rho_out)

def find_N0(rho_GeV4, g, lam, m0):
    def f(N):
        return m0**2 * N + lam * N**3 + g * rho_GeV4
    # try bracketed root search
    scales = [1e-40,1e-30,1e-20,1e-10,1e-6,1e-3,1,1e3,1e6,1e12,1e20,1e30]
    for s in scales:
        a, b = -s, s
        if f(a)*f(b) < 0:
            return brentq(f, a, b, maxiter=200)
    # fallback Newton
    if lam>0:
        N_guess = -abs((g * rho_GeV4 / lam)**(1.0/3.0))
    else:
        N_guess = -1.0
    N = N_guess
    for _ in range(500):
        fv = f(N)
        df = m0**2 + 3.0 * lam * N**2
        if abs(df) < 1e-40:
            break
        N_new = N - fv/df
        if abs(N_new - N) < 1e-16 * max(1.0, abs(N_new)):
            return N_new
        N = N_new
    return N

# parameter grids (log space)
g_vals = np.logspace(-30, -15, 8)        # GeV^-1
lambda_vals = np.logspace(-40, -10, 7)   # dimensionless
m0_vals_eV = np.logspace(-30, -10, 9)    # eV
m0_vals = m0_vals_eV * 1e-9              # convert eV -> GeV

out_rows = []
for g in g_vals:
    for lam in lambda_vals:
        for m0 in m0_vals:
            N_in = find_N0(rho_in_GeV4, g, lam, m0)
            N_out = find_N0(rho_out_GeV4, g, lam, m0)
            mN_in = max(0.0, m0**2 + 3.0*lam*N_in**2)**0.5
            mN_out = max(0.0, m0**2 + 3.0*lam*N_out**2)**0.5
            R_sun_GeVinv = R_sun * m_to_GeVinv
            DeltaR_over_R = (N_out - N_in) / (6.0 * g * Mpl_GeV * Phi_sun)
            gamma_est = 2.0 * (g * abs(N_out) / Mpl_GeV) * max(1.0, 3.0 * abs(DeltaR_over_R))
            out_rows.append([g, lam, m0*1e9, N_in, N_out, mN_in, mN_out, mN_in*R_sun_GeVinv, DeltaR_over_R, gamma_est])

# write CSV
with open('thinshell_scan.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['g_GeVinv','lambda','m0_eV','N_in_GeV','N_out_GeV','mN_in_GeV','mN_out_GeV','mN_inR','DeltaR_over_R','gamma_est'])
    writer.writerows(out_rows)

# print a few promising candidates (DeltaR small and gamma small)
print("Candidates with DeltaR/R < 1 and gamma < 2e-5:")
for row in out_rows:
    if abs(row[8]) < 1.0 and row[9] < 2e-5:
        print("g={:.1e}, lambda={:.1e}, m0_eV={:.1e}, DeltaR={:.1e}, gamma={:.1e}".format(row[0], row[1], row[2], row[8], row[9]))
print("Scan complete. Results saved to thinshell_scan.csv")
