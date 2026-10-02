# quick_scan.py
import numpy as np

# ... (usa las mismas constantes y funciones de solar_fast.py)

g = 1e-20
lams = [1e-20, 1e-18, 1e-15, 1e-12]
m0_eVs = [1e-12, 1e-9, 1e-6]

for lam in lams:
    for m0_eV in m0_eVs:
        m0 = m0_eV*1e-9
        N_in = N0_approx(rho_in_GeV4, g, lam, m0)
        N_out = N0_approx(rho_out_GeV4, g, lam, m0)
        mN_in = np.sqrt(m0**2 + 3*lam*N_in**2)
        R_sun_GeVinv = R_sun*m_to_GeVinv
        DeltaR_over_R = (N_out - N_in)/(6*g*Mpl_GeV*Phi_sun)
        gamma_minus_1 = 2*(g*abs(N_out)/Mpl_GeV)*max(1.0,3*abs(DeltaR_over_R))
        print(f"λ={lam:.1e}, m0={m0_eV:.1e} eV -> ΔR/R={DeltaR_over_R:.1e}, |γ-1|={gamma_minus_1:.1e}")
