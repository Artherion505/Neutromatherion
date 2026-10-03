"""Quadratic no-Lambda scan and early-to-present FLRW history."""

from math import sqrt

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from flrw_coupled import (
    FLRWInitialConditions,
    FLRWParameters,
    background_quantities,
    photon_speed_ratio,
    relic_neutrino_components,
    relic_neutrino_energy_pressure,
    solve_background,
)
from flrw_fiducial import (
    MPL_GEV,
    OMEGA_THETA_DIAGNOSTIC,
    PLANCK_H0_KM_S_MPC,
    PLANCK_OMEGA_M,
    reference_densities,
)


EARLY_SCALE_FACTOR = 1e-8
COUPLING_G_GEV_INV = 1e-40
COUPLING_H_GEV_INV = 1e-40
SUM_NEUTRINO_MASSES_EV = 0.06
NEUTRINO_SPECIES_COUNT = 3
NEUTRINO_MASSES_GEV = (SUM_NEUTRINO_MASSES_EV * 1e-9,) + (0.0,) * (
    NEUTRINO_SPECIES_COUNT - 1
)
MASS_RATIOS_TO_SCAN = (0.1, 0.2, 0.5, 1.0)
HISTORY_CHECKPOINTS = (
    1e-8,
    1e-4,
    1.0 / 1101.0,
    1e-2,
    0.1,
    0.5,
    1.0,
)
DISTANCE_REDSHIFTS = (0.5, 1.0, 2.0)
DISTANCE_SCALE_FACTORS = tuple(1.0 / (1.0 + z) for z in DISTANCE_REDSHIFTS)


def _photon_speed_shift(N_GeV, parameters):
    xi = parameters.h_GeV_inv * N_GeV
    speed = photon_speed_ratio(N_GeV, parameters)
    return -2.0 * xi / ((1.0 + xi) * (1.0 + speed))


def _parameters_for_mass_ratio(mass_ratio, reference):
    if not np.isfinite(mass_ratio) or mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be finite and positive.")
    return FLRWParameters(
        m_phi_GeV=mass_ratio * reference["H0_GeV"],
        lambda_N=0.0,
        g_GeV_inv=COUPLING_G_GEV_INV,
        h_GeV_inv=COUPLING_H_GEV_INV,
        Z_theta=1.0,
        Mpl_GeV=MPL_GEV,
        neutrino_masses_GeV=NEUTRINO_MASSES_GEV,
        N_eff=3.046,
    )


def _integrate_trial(parameters, reference, N_initial_GeV, sample_count):
    if sample_count < 2:
        raise ValueError("sample_count must be at least two.")

    final_log_a = np.log(1.0 / EARLY_SCALE_FACTOR)
    rho_critical = reference["rho_critical_GeV4"]
    rho_theta_today = OMEGA_THETA_DIAGNOSTIC * rho_critical
    neutrino_today = relic_neutrino_components(1.0, parameters)
    rho_cb_today = (
        PLANCK_OMEGA_M * rho_critical
        - neutrino_today["rho_massive_GeV4"]
    )
    if rho_cb_today <= 0.0:
        raise ValueError("The relic-neutrino density exceeds the Planck matter density.")
    eta = parameters.eta_GeV_inv
    matter_factor = 1.0 + eta * N_initial_GeV
    if matter_factor <= 0.0:
        raise ValueError("The initial scalar value makes the dust mass nonpositive.")

    initial = FLRWInitialConditions(
        N_GeV=N_initial_GeV,
        Ndot_GeV2=0.0,
        rho_m_bare_i_GeV4=(
            rho_cb_today
            * np.exp(3.0 * final_log_a)
            / matter_factor
        ),
        rho_gamma_i_GeV4=(
            reference["rho_gamma_GeV4"] * np.exp(4.0 * final_log_a)
        ),
        Theta_dot_i_GeV2=(
            sqrt(2.0 * rho_theta_today / parameters.Z_theta)
            * np.exp(3.0 * final_log_a)
        ),
        scale_factor_i=EARLY_SCALE_FACTOR,
    )
    log_a_samples = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, final_log_a, sample_count),
                np.log(np.asarray(HISTORY_CHECKPOINTS) / EARLY_SCALE_FACTOR),
                np.log(np.asarray(DISTANCE_SCALE_FACTORS) / EARLY_SCALE_FACTOR),
            )
        )
    )
    solution = solve_background(
        parameters,
        initial,
        (0.0, final_log_a),
        rtol=2e-9,
        atol=(1e7, 1e-48, 1e-3, 1e28),
        max_step=0.1,
        t_eval=log_a_samples,
    )
    if not solution.success:
        raise RuntimeError(f"Background integration failed: {solution.message}")

    backgrounds = [
        background_quantities(log_a, state, parameters, initial)
        for log_a, state in zip(solution.t, solution.y.T)
    ]
    return solution, initial, backgrounds


def solve_slow_roll_history(mass_ratio, sample_count=1001):
    """Shoot the initial field value to H0 and return its full history."""
    reference = reference_densities()
    parameters = _parameters_for_mass_ratio(mass_ratio, reference)
    rho_theta = OMEGA_THETA_DIAGNOSTIC * reference["rho_critical_GeV4"]
    neutrino_today = relic_neutrino_components(1.0, parameters)
    rho_cb_today = (
        PLANCK_OMEGA_M * reference["rho_critical_GeV4"]
        - neutrino_today["rho_massive_GeV4"]
    )
    rho_scalar_target = (
        reference["rho_critical_GeV4"]
        - rho_cb_today
        - neutrino_today["rho_GeV4"]
        - reference["rho_gamma_GeV4"]
        - rho_theta
    )
    if rho_scalar_target <= 0.0:
        raise ValueError("The reference densities leave no positive scalar density.")

    N_guess = sqrt(2.0 * rho_scalar_target) / parameters.m_phi_GeV
    final_log_a = np.log(1.0 / EARLY_SCALE_FACTOR)

    def hubble_residual(N_initial_GeV):
        _, _, backgrounds = _integrate_trial(
            parameters, reference, N_initial_GeV, sample_count=2
        )
        return backgrounds[-1]["H_GeV"] / reference["H0_GeV"] - 1.0

    N_initial = brentq(
        hubble_residual,
        0.5 * N_guess,
        1.5 * N_guess,
        xtol=1.0,
        rtol=1e-10,
        maxiter=60,
    )
    solution, initial, backgrounds = _integrate_trial(
        parameters, reference, N_initial, sample_count
    )

    scale_factors = np.exp(solution.t - final_log_a)
    H_over_H0 = np.array(
        [item["H_GeV"] / reference["H0_GeV"] for item in backgrounds]
    )
    critical_densities = np.array(
        [3.0 * parameters.Mpl_GeV**2 * item["H_GeV"] ** 2 for item in backgrounds]
    )
    history = {
        "a": scale_factors,
        "H_over_H0": H_over_H0,
        "omega_N": np.array(
            [item["rho_N_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "omega_cb": np.array(
            [item["rho_m_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "omega_nu": np.array(
            [item["rho_neutrino_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "omega_nu_massive": np.array(
            [item["rho_neutrino_massive_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "omega_nu_massless": np.array(
            [item["rho_neutrino_massless_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "omega_gamma": np.array(
            [item["rho_gamma_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "omega_theta": np.array(
            [item["rho_theta_GeV4"] for item in backgrounds]
        )
        / critical_densities,
        "w_N": np.array(
            [item["pressure_N_GeV4"] / item["rho_N_GeV4"] for item in backgrounds]
        ),
        "xi": np.array([item["xi"] for item in backgrounds]),
    }
    today_background = backgrounds[-1]
    today_state = solution.y[:, -1]
    today = {
        "N_GeV": float(today_state[0]),
        "N_over_Mpl": float(today_state[0] / parameters.Mpl_GeV),
        "H_over_H0": float(H_over_H0[-1]),
        "omega_N": float(history["omega_N"][-1]),
        "omega_cb": float(history["omega_cb"][-1]),
        "omega_nu": float(history["omega_nu"][-1]),
        "omega_nu_massive": float(history["omega_nu_massive"][-1]),
        "omega_nu_massless": float(history["omega_nu_massless"][-1]),
        "omega_m_total": float(
            history["omega_cb"][-1] + history["omega_nu_massive"][-1]
        ),
        "omega_gamma": float(history["omega_gamma"][-1]),
        "omega_theta": float(history["omega_theta"][-1]),
        "w_N": float(history["w_N"][-1]),
        "xi": float(today_background["xi"]),
        "delta_c_gamma": float(_photon_speed_shift(today_state[0], parameters)),
        "m_eff_over_H0": float(parameters.m_phi_GeV / reference["H0_GeV"]),
        "max_abs_xi": float(np.max(np.abs(history["xi"]))),
    }

    omega_theta0 = OMEGA_THETA_DIAGNOSTIC
    neutrino_reference = [
        relic_neutrino_components(scale_factor, parameters)
        for scale_factor in scale_factors
    ]
    omega_nu_reference = np.array(
        [item["rho_GeV4"] for item in neutrino_reference]
    ) / reference["rho_critical_GeV4"]
    omega_nu_massive0 = (
        neutrino_reference[-1]["rho_massive_GeV4"]
        / reference["rho_critical_GeV4"]
    )
    omega_nu_massless0 = (
        neutrino_reference[-1]["rho_massless_GeV4"]
        / reference["rho_critical_GeV4"]
    )
    omega_nu0 = float(omega_nu_reference[-1])
    omega_cb0 = PLANCK_OMEGA_M - omega_nu_massive0
    omega_de_reference = (
        1.0
        - PLANCK_OMEGA_M
        - omega_nu_massless0
        - reference["omega_gamma"]
        - omega_theta0
    )
    reference_H_over_H0 = np.sqrt(
        omega_cb0 / scale_factors**3
        + reference["omega_gamma"] / scale_factors**4
        + omega_nu_reference
        + omega_theta0 / scale_factors**6
        + omega_de_reference
    )
    max_fractional_H_difference = float(
        np.max(np.abs(H_over_H0 / reference_H_over_H0 - 1.0))
    )
    history["omega_nu_reference"] = omega_nu_reference
    history["H_reference_over_H0"] = reference_H_over_H0

    log_scale_factors = np.log(scale_factors)
    comoving_distance_integrand = 1.0 / (scale_factors * H_over_H0)
    reference_distance_integrand = 1.0 / (
        scale_factors * reference_H_over_H0
    )
    distance_integral = cumulative_trapezoid(
        comoving_distance_integrand, log_scale_factors, initial=0.0
    )
    reference_distance_integral = cumulative_trapezoid(
        reference_distance_integrand, log_scale_factors, initial=0.0
    )
    distance_unit_Mpc = 299792.458 / PLANCK_H0_KM_S_MPC
    distance_comparison = {}
    for redshift, scale_factor in zip(
        DISTANCE_REDSHIFTS, DISTANCE_SCALE_FACTORS
    ):
        index = int(np.argmin(np.abs(scale_factors - scale_factor)))
        model_distance = (
            distance_integral[-1] - distance_integral[index]
        ) * distance_unit_Mpc
        reference_distance = (
            reference_distance_integral[-1]
            - reference_distance_integral[index]
        ) * distance_unit_Mpc
        distance_comparison[redshift] = {
            "model_Mpc": float(model_distance),
            "reference_Mpc": float(reference_distance),
            "fractional_difference": float(
                model_distance / reference_distance - 1.0
            ),
        }
    max_fractional_distance_difference = max(
        abs(value["fractional_difference"])
        for value in distance_comparison.values()
    )

    return {
        "mass_ratio": mass_ratio,
        "parameters": parameters,
        "initial": initial,
        "solution": solution,
        "history": history,
        "today": today,
        "omega_cb0": omega_cb0,
        "omega_nu0": omega_nu0,
        "omega_nu_massive0": omega_nu_massive0,
        "omega_nu_massless0": omega_nu_massless0,
        "max_fractional_H_difference": max_fractional_H_difference,
        "distance_comparison": distance_comparison,
        "max_fractional_distance_difference": max_fractional_distance_difference,
    }


def run_scan():
    print(
        "No-Lambda quadratic scan with thermal relic neutrinos; "
        "three unit-weight states (one 0.06 eV, two massless) plus "
        "Delta N_eff=0.046 massless radiation; not a likelihood fit."
    )
    print(
        "m_phi/H0  Ni/Mpl  N0/Mpl  Omega_cb0  Omega_nu,m0  Omega_nu,r0  "
        "Omega_m0  w_N0  delta_c_gamma  max|Delta H/H_ref|"
    )
    results = {
        ratio: solve_slow_roll_history(ratio) for ratio in MASS_RATIOS_TO_SCAN
    }
    for ratio, result in results.items():
        today = result["today"]
        print(
            f"{ratio:8.3g}  "
            f"{result['initial'].N_GeV / result['parameters'].Mpl_GeV:6.3f}  "
            f"{today['N_over_Mpl']:6.3f}  "
            f"{today['omega_cb']:.6f}  "
            f"{today['omega_nu_massive']:.6f}  "
            f"{today['omega_nu_massless']:.6f}  "
            f"{today['omega_m_total']:.6f}  {today['w_N']:.6f}  "
            f"{today['delta_c_gamma']:.3e}  "
            f"{result['max_fractional_H_difference']:.3e}"
        )

    selected = results[0.2]
    print("H(z)/H0, reference, and fractional difference for m_phi/H0=0.2:")
    for target_a in HISTORY_CHECKPOINTS:
        index = int(np.argmin(np.abs(selected["history"]["a"] - target_a)))
        H_over_H0 = selected["history"]["H_over_H0"][index]
        H_reference_over_H0 = selected["history"]["H_reference_over_H0"][index]
        print(
            f"z={1.0 / target_a - 1.0:.6g}, H/H0={H_over_H0:.8g}, "
            f"reference={H_reference_over_H0:.8g}, "
            f"Delta={H_over_H0 / H_reference_over_H0 - 1.0:.3e}"
        )
    print("Comoving-distance comparison:")
    for redshift, distance in selected["distance_comparison"].items():
        print(
            f"z={redshift:g}, D_C={distance['model_Mpc']:.4f} Mpc, "
            f"reference={distance['reference_Mpc']:.4f} Mpc, "
            f"Delta={distance['fractional_difference']:.3e}"
        )
    return results


if __name__ == "__main__":
    run_scan()