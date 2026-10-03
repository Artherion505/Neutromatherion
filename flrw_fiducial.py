"""Present-day FLRW normalization check using a Planck 2018 reference.

The reference uses the published central H0 and Omega_m values, derives the
photon density from a 2.7255 K blackbody, and omits relic neutrinos. It is a
normalization diagnostic, not a cosmological likelihood or data fit.
"""

from dataclasses import replace
from math import pi, sqrt

from scipy.optimize import brentq

from flrw_coupled import (
    FLRWInitialConditions,
    FLRWParameters,
    background_quantities,
    photon_speed_ratio,
    potential,
    potential_derivative,
)


PLANCK_H0_KM_S_MPC = 67.4
PLANCK_OMEGA_M = 0.315
TCMB_K = 2.7255
OMEGA_THETA_DIAGNOSTIC = 1e-30
MPC_KM = 3.0856775814913673e19
HBAR_GEV_S = 6.582119569e-25
BOLTZMANN_GEV_K = 8.617333262e-14
MPL_GEV = 2.4353234593382e18


def reference_densities():
    H0_s = PLANCK_H0_KM_S_MPC / MPC_KM
    H0_GeV = H0_s * HBAR_GEV_S
    rho_critical = 3.0 * MPL_GEV**2 * H0_GeV**2
    temperature_GeV = BOLTZMANN_GEV_K * TCMB_K
    rho_gamma = (pi**2 / 15.0) * temperature_GeV**4
    return {
        "H0_s": H0_s,
        "H0_GeV": H0_GeV,
        "rho_critical_GeV4": rho_critical,
        "rho_matter_GeV4": PLANCK_OMEGA_M * rho_critical,
        "rho_gamma_GeV4": rho_gamma,
        "omega_gamma": rho_gamma / rho_critical,
    }


def scalar_minimum(parameters, rho_matter_total, rho_gamma):
    eta = parameters.eta_GeV_inv

    def residual(N_GeV):
        xi = parameters.h_GeV_inv * N_GeV
        matter_factor = 1.0 + eta * N_GeV
        if abs(xi) >= 1.0 or matter_factor <= 0.0:
            raise ValueError("The trial minimum left the positive-energy domain.")
        rho_matter_bare = rho_matter_total / matter_factor
        rho_gamma_bare = rho_gamma / (1.0 - xi**2)
        return (
            potential_derivative(N_GeV, parameters)
            + eta * rho_matter_bare
            - parameters.h_GeV_inv * rho_gamma_bare
        )

    return brentq(residual, -1e12, 1e12, xtol=1e-12, rtol=1e-12)


def run_fiducial_check():
    reference = reference_densities()
    rho_critical = reference["rho_critical_GeV4"]
    rho_matter = reference["rho_matter_GeV4"]
    rho_gamma = reference["rho_gamma_GeV4"]

    parameters = FLRWParameters(
        m_phi_GeV=1e-39,
        lambda_N=1e-69,
        g_GeV_inv=1e-20,
        h_GeV_inv=0.5e-20,
        Z_theta=1.0,
        Mpl_GeV=MPL_GEV,
    )
    rho_theta = OMEGA_THETA_DIAGNOSTIC * rho_critical
    theta_dot = sqrt(2.0 * rho_theta / parameters.Z_theta)

    N_minimum = scalar_minimum(parameters, rho_matter, rho_gamma)
    eta = parameters.eta_GeV_inv
    rho_matter_bare_at_minimum = rho_matter / (1.0 + eta * N_minimum)
    minimum_initial = FLRWInitialConditions(
        N_GeV=N_minimum,
        Ndot_GeV2=0.0,
        rho_m_bare_i_GeV4=rho_matter_bare_at_minimum,
        rho_gamma_i_GeV4=rho_gamma,
        Theta_dot_i_GeV2=theta_dot,
    )
    minimum_state = (N_minimum, 0.0, 0.0, 0.0)
    minimum_background = background_quantities(
        0.0, minimum_state, parameters, minimum_initial
    )
    rho_lambda = rho_critical - minimum_background["rho_total_GeV4"]
    lambda_parameters = replace(parameters, rho_lambda_GeV4=rho_lambda)
    lambda_background = background_quantities(
        0.0, minimum_state, lambda_parameters, minimum_initial
    )

    rho_scalar_target = rho_critical - rho_matter - rho_gamma - rho_theta

    def scalar_energy_residual(N_GeV):
        return potential(N_GeV, parameters) - rho_scalar_target

    N_dark_energy = brentq(scalar_energy_residual, 0.0, 1e16, xtol=1e-10, rtol=1e-12)
    rho_matter_bare_no_lambda = rho_matter / (1.0 + eta * N_dark_energy)
    no_lambda_initial = FLRWInitialConditions(
        N_GeV=N_dark_energy,
        Ndot_GeV2=0.0,
        rho_m_bare_i_GeV4=rho_matter_bare_no_lambda,
        rho_gamma_i_GeV4=rho_gamma,
        Theta_dot_i_GeV2=theta_dot,
    )
    no_lambda_state = (N_dark_energy, 0.0, 0.0, 0.0)
    scalar_only_background = background_quantities(
        0.0, no_lambda_state, parameters, no_lambda_initial
    )
    potential_curvature_mass = sqrt(
        parameters.m_phi_GeV**2
        + 3.0 * parameters.lambda_N * N_dark_energy**2
    )
    scalar_acceleration = (
        -potential_derivative(N_dark_energy, parameters)
        - eta * scalar_only_background["rho_m_bare_GeV4"]
        + parameters.h_GeV_inv * scalar_only_background["rho_gamma_bare_GeV4"]
    )
    photon_speed_shift = photon_speed_ratio(N_dark_energy, parameters) - 1.0
    diagnostics = {
        "N_sourced_minimum_GeV": N_minimum,
        "omega_N_sourced_minimum": (
            minimum_background["rho_N_GeV4"] / rho_critical
        ),
        "omega_lambda": rho_lambda / rho_critical,
        "h_ratio_lambda": lambda_background["H_GeV"] / reference["H0_GeV"],
        "N_no_lambda_GeV": N_dark_energy,
        "omega_N_no_lambda": (
            scalar_only_background["rho_N_GeV4"] / rho_critical
        ),
        "xi_no_lambda": scalar_only_background["xi"],
        "delta_c_gamma_no_lambda": photon_speed_shift,
        "h_ratio_no_lambda": (
            scalar_only_background["H_GeV"] / reference["H0_GeV"]
        ),
        "curvature_mass_over_H0": (
            potential_curvature_mass / reference["H0_GeV"]
        ),
    }

    print("Planck 2018 present-day normalization; relic neutrinos omitted")
    print(
        f"H0={PLANCK_H0_KM_S_MPC:.1f} km/s/Mpc, "
        f"Omega_m={PLANCK_OMEGA_M:.3f}, "
        f"derived Omega_gamma={reference['omega_gamma']:.6g}"
    )
    print(
        "Solar-scan trial point: "
        f"g={parameters.g_GeV_inv:.3g} GeV^-1, "
        f"h={parameters.h_GeV_inv:.3g} GeV^-1, "
        f"m_phi={parameters.m_phi_GeV:.3g} GeV, "
        f"lambda={parameters.lambda_N:.3g}"
    )
    print(
        "With Lambda and N at the sourced minimum: "
        f"N={diagnostics['N_sourced_minimum_GeV']:.6g} GeV, "
        f"Omega_N={diagnostics['omega_N_sourced_minimum']:.6g}, "
        f"Omega_Lambda={diagnostics['omega_lambda']:.6g}, "
        f"H/H0={diagnostics['h_ratio_lambda']:.12g}"
    )
    print(
        "With Lambda=0 and V(N) normalized to today's missing density: "
        f"N={diagnostics['N_no_lambda_GeV']:.6g} GeV, "
        f"xi=hN={diagnostics['xi_no_lambda']:.6g}, "
        f"delta_c_gamma={diagnostics['delta_c_gamma_no_lambda']:.6g}, "
        f"H/H0={diagnostics['h_ratio_no_lambda']:.12g}"
    )
    print(
        "No-Lambda potential-curvature diagnostic (not the full coupled "
        "perturbation mass): "
        f"sqrt(V'')/H0={diagnostics['curvature_mass_over_H0']:.6g}, "
        f"Nddot={scalar_acceleration:.6g} GeV^3"
    )
    print(
        f"Clock diagnostic: Omega_Theta(today)={OMEGA_THETA_DIAGNOSTIC:.1e}; "
        "this is a chosen negligible-energy normalization, not a fitted value."
    )
    return diagnostics


if __name__ == "__main__":
    run_fiducial_check()