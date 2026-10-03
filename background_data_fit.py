"""Background-only Pantheon+ Only and DESI DR2 likelihood fits.

The uncalibrated supernova sample absorbs H0 into its profiled magnitude
intercept, while BAO with a free sound horizon measures H0*r_d, not H0 and
r_d separately. H0 below is therefore an explicit conversion pivot, not a
parameter constraint or a Planck prior. This script does not fit CMB spectra
or lensing, which require perturbation predictions.
"""

from dataclasses import dataclass
from math import log, log10, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import brentq, least_squares

from flrw_coupled import (
    FLRWInitialConditions,
    FLRWParameters,
    background_quantities,
    relic_neutrino_components,
    solve_background,
)
from flrw_fiducial import (
    BOLTZMANN_GEV_K,
    HBAR_GEV_S,
    MPL_GEV,
    MPC_KM,
    TCMB_K,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
H0_PIVOT_KM_S_MPC = 70.0
EARLY_SCALE_FACTOR = 1e-8
NEUTRINO_MASSES_GEV = (0.06e-9, 0.0, 0.0)
NEUTRINO_N_EFF = 3.046
COUPLING_G_GEV_INV = 1e-40
COUPLING_H_GEV_INV = 1e-40
OMEGA_THETA_DIAGNOSTIC = 1e-30
SPEED_OF_LIGHT_KM_S = 299792.458
SOLVER_RTOL = 2e-8
SOLVER_ATOL = (1e7, 1e-48, 1e-3, 1e28)
SOLVER_MAX_STEP = 0.2

DESI_EXPECTED_ORDER = (
    (0.295, "DV_over_rs"),
    (0.510, "DM_over_rs"),
    (0.510, "DH_over_rs"),
    (0.706, "DM_over_rs"),
    (0.706, "DH_over_rs"),
    (0.934, "DM_over_rs"),
    (0.934, "DH_over_rs"),
    (1.321, "DM_over_rs"),
    (1.321, "DH_over_rs"),
    (1.484, "DM_over_rs"),
    (1.484, "DH_over_rs"),
    (2.330, "DH_over_rs"),
    (2.330, "DM_over_rs"),
)


@dataclass(frozen=True)
class BackgroundData:
    sn_z_hd: np.ndarray
    sn_z_hel: np.ndarray
    sn_magnitude: np.ndarray
    sn_cholesky: np.ndarray
    sn_whitened_one: np.ndarray
    sn_one_norm: float
    bao_z: np.ndarray
    bao_values: np.ndarray
    bao_kinds: tuple
    bao_cholesky: np.ndarray

    @property
    def point_count(self):
        return len(self.sn_z_hd) + len(self.bao_z)


def _load_pantheon_covariance(path, expected_size):
    with path.open("r", encoding="ascii") as stream:
        declared_size = int(stream.readline().strip())
        flat_covariance = np.loadtxt(stream)
    if declared_size != expected_size or flat_covariance.size != declared_size**2:
        raise ValueError(
            "Pantheon+ covariance size does not match its data table: "
            f"header={declared_size}, rows={expected_size}, "
            f"elements={flat_covariance.size}."
        )
    return flat_covariance.reshape(declared_size, declared_size)


def load_background_data(data_dir=DEFAULT_DATA_DIR):
    """Load Pantheon+ Only and the combined DESI DR2 BAO vector."""
    data_dir = Path(data_dir)
    pantheon_path = data_dir / "Pantheon+SH0ES.dat"
    pantheon_covariance_path = data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
    desi_mean_path = data_dir / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    desi_covariance_path = data_dir / "desi_gaussian_bao_ALL_GCcomb_cov.txt"
    required_paths = (
        pantheon_path,
        pantheon_covariance_path,
        desi_mean_path,
        desi_covariance_path,
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing likelihood data files: " + ", ".join(missing_paths)
        )

    pantheon = pd.read_csv(pantheon_path, sep=r"\s+")
    pantheon_mask = pantheon["zHD"].to_numpy(dtype=float) > 0.01
    if len(pantheon) != 1701:
        raise ValueError(f"Expected 1701 Pantheon+ rows, found {len(pantheon)}.")
    pantheon_covariance = _load_pantheon_covariance(
        pantheon_covariance_path, len(pantheon)
    )
    sn_covariance = pantheon_covariance[np.ix_(pantheon_mask, pantheon_mask)]
    if not np.allclose(sn_covariance, sn_covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The selected Pantheon+ covariance is not symmetric.")
    sn_cholesky = cholesky(sn_covariance, lower=True, check_finite=True)
    sn_whitened_one = solve_triangular(
        sn_cholesky,
        np.ones(int(pantheon_mask.sum())),
        lower=True,
        check_finite=False,
    )

    desi_mean = np.loadtxt(desi_mean_path, dtype=str)
    desi_covariance = np.loadtxt(desi_covariance_path)
    expected_z = np.asarray([entry[0] for entry in DESI_EXPECTED_ORDER])
    expected_kinds = tuple(entry[1] for entry in DESI_EXPECTED_ORDER)
    if desi_mean.shape != (len(DESI_EXPECTED_ORDER), 3):
        raise ValueError(f"Unexpected DESI mean-vector shape: {desi_mean.shape}.")
    if desi_covariance.shape != (len(DESI_EXPECTED_ORDER),) * 2:
        raise ValueError(
            f"Unexpected DESI covariance shape: {desi_covariance.shape}."
        )
    if not np.allclose(desi_mean[:, 0].astype(float), expected_z, atol=1e-8):
        raise ValueError("DESI redshifts do not match the published vector order.")
    if tuple(desi_mean[:, 2]) != expected_kinds:
        raise ValueError("DESI observables do not match the published vector order.")
    if not np.allclose(desi_covariance, desi_covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("The DESI covariance is not symmetric.")
    bao_cholesky = cholesky(desi_covariance, lower=True, check_finite=True)

    return BackgroundData(
        sn_z_hd=pantheon.loc[pantheon_mask, "zHD"].to_numpy(dtype=float),
        sn_z_hel=pantheon.loc[pantheon_mask, "zHEL"].to_numpy(dtype=float),
        sn_magnitude=pantheon.loc[pantheon_mask, "m_b_corr"].to_numpy(
            dtype=float
        ),
        sn_cholesky=sn_cholesky,
        sn_whitened_one=sn_whitened_one,
        sn_one_norm=float(np.dot(sn_whitened_one, sn_whitened_one)),
        bao_z=desi_mean[:, 0].astype(float),
        bao_values=desi_mean[:, 1].astype(float),
        bao_kinds=expected_kinds,
        bao_cholesky=bao_cholesky,
    )


def _h0_in_gev(h0_km_s_mpc):
    return (h0_km_s_mpc / MPC_KM) * HBAR_GEV_S


def _neutrino_parameters(m_phi_GeV=0.0, g_GeV_inv=0.0, h_GeV_inv=0.0):
    return FLRWParameters(
        m_phi_GeV=m_phi_GeV,
        lambda_N=0.0,
        g_GeV_inv=g_GeV_inv,
        h_GeV_inv=h_GeV_inv,
        Z_theta=1.0,
        Mpl_GeV=MPL_GEV,
        neutrino_masses_GeV=NEUTRINO_MASSES_GEV,
        N_eff=NEUTRINO_N_EFF,
    )


def _present_densities(h0_km_s_mpc, parameters):
    h0_gev = _h0_in_gev(h0_km_s_mpc)
    rho_critical = 3.0 * MPL_GEV**2 * h0_gev**2
    temperature_gamma = BOLTZMANN_GEV_K * TCMB_K
    rho_gamma = (pi**2 / 15.0) * temperature_gamma**4
    neutrinos = relic_neutrino_components(1.0, parameters)
    return {
        "H0_GeV": h0_gev,
        "rho_critical_GeV4": rho_critical,
        "rho_gamma_GeV4": rho_gamma,
        "neutrinos": neutrinos,
    }


def lcdm_expansion(redshifts, h0_km_s_mpc, omega_m):
    """Flat LCDM expansion with the same thermal relic-neutrino treatment."""
    redshifts = np.asarray(redshifts, dtype=float)
    if np.any(~np.isfinite(redshifts)) or np.any(redshifts < 0.0):
        raise ValueError("Redshifts must be finite and nonnegative.")
    parameters = _neutrino_parameters()
    present = _present_densities(h0_km_s_mpc, parameters)
    rho_critical = present["rho_critical_GeV4"]
    neutrinos_today = present["neutrinos"]
    omega_nu_massive = neutrinos_today["rho_massive_GeV4"] / rho_critical
    omega_nu_massless = neutrinos_today["rho_massless_GeV4"] / rho_critical
    omega_gamma = present["rho_gamma_GeV4"] / rho_critical
    omega_cb = omega_m - omega_nu_massive
    omega_lambda = 1.0 - omega_m - omega_nu_massless - omega_gamma
    if omega_cb <= 0.0 or omega_lambda <= 0.0:
        raise ValueError("LCDM parameters leave no positive matter or Lambda density.")

    scale_factors = 1.0 / (1.0 + redshifts)
    rho_neutrinos = np.asarray(
        [
            relic_neutrino_components(scale_factor, parameters)["rho_GeV4"]
            for scale_factor in scale_factors
        ]
    )
    e_squared = (
        omega_cb / scale_factors**3
        + omega_gamma / scale_factors**4
        + rho_neutrinos / rho_critical
        + omega_lambda
    )
    if np.any(~np.isfinite(e_squared)) or np.any(e_squared <= 0.0):
        raise ValueError("LCDM expansion rate left the positive finite domain.")
    return np.sqrt(e_squared)


def scalar_clock_expansion(
    redshift_grid,
    h0_km_s_mpc,
    omega_m,
    mass_ratio,
    *,
    g_GeV_inv=COUPLING_G_GEV_INV,
    h_GeV_inv=COUPLING_H_GEV_INV,
    return_history=False,
):
    """Return E(z), optionally with the sampled scalar-clock history."""
    redshift_grid = np.asarray(redshift_grid, dtype=float)
    if (
        redshift_grid.ndim != 1
        or len(redshift_grid) < 2
        or np.any(~np.isfinite(redshift_grid))
        or np.any(np.diff(redshift_grid) <= 0.0)
        or redshift_grid[0] != 0.0
        or mass_ratio <= 0.0
    ):
        raise ValueError("Require an increasing redshift grid from zero and m/H0 > 0.")

    h0_gev = _h0_in_gev(h0_km_s_mpc)
    parameters = _neutrino_parameters(
        m_phi_GeV=mass_ratio * h0_gev,
        g_GeV_inv=g_GeV_inv,
        h_GeV_inv=h_GeV_inv,
    )
    present = _present_densities(h0_km_s_mpc, parameters)
    rho_critical = present["rho_critical_GeV4"]
    neutrinos_today = present["neutrinos"]
    rho_matter_today = (
        omega_m * rho_critical - neutrinos_today["rho_massive_GeV4"]
    )
    rho_theta_today = OMEGA_THETA_DIAGNOSTIC * rho_critical
    scalar_energy_target = (
        rho_critical
        - rho_matter_today
        - neutrinos_today["rho_GeV4"]
        - present["rho_gamma_GeV4"]
        - rho_theta_today
    )
    if rho_matter_today <= 0.0 or scalar_energy_target <= 0.0:
        raise ValueError("The selected matter density leaves no positive scalar branch.")

    final_log_a = log(1.0 / EARLY_SCALE_FACTOR)
    rho_theta_dot_today = sqrt(2.0 * rho_theta_today / parameters.Z_theta)

    def make_initial(N_initial_GeV):
        matter_factor = 1.0 + parameters.eta_GeV_inv * N_initial_GeV
        if matter_factor <= 0.0:
            raise ValueError("The initial scalar makes the dust mass nonpositive.")
        return FLRWInitialConditions(
            N_GeV=N_initial_GeV,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=(
                rho_matter_today * np.exp(3.0 * final_log_a) / matter_factor
            ),
            rho_gamma_i_GeV4=(
                present["rho_gamma_GeV4"] * np.exp(4.0 * final_log_a)
            ),
            Theta_dot_i_GeV2=(
                rho_theta_dot_today * np.exp(3.0 * final_log_a)
            ),
            scale_factor_i=EARLY_SCALE_FACTOR,
        )

    def integrate(initial, t_eval):
        solution = solve_background(
            parameters,
            initial,
            (0.0, final_log_a),
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_step=SOLVER_MAX_STEP,
            t_eval=t_eval,
        )
        if not solution.success:
            raise RuntimeError(f"Scalar background integration failed: {solution.message}")
        return solution

    N_guess = sqrt(2.0 * scalar_energy_target) / parameters.m_phi_GeV

    def hubble_residual(N_initial_GeV):
        initial = make_initial(N_initial_GeV)
        solution = integrate(initial, (0.0, final_log_a))
        today = background_quantities(
            final_log_a, solution.y[:, -1], parameters, initial
        )
        return today["H_GeV"] / h0_gev - 1.0

    N_initial = brentq(
        hubble_residual,
        0.5 * N_guess,
        1.5 * N_guess,
        xtol=1.0,
        rtol=1e-10,
        maxiter=60,
    )
    initial = make_initial(N_initial)
    redshifts_descending = redshift_grid[::-1]
    log_a_eval = np.log(
        1.0 / (1.0 + redshifts_descending) / EARLY_SCALE_FACTOR
    )
    solution = integrate(initial, log_a_eval)
    background_history = [
        background_quantities(float(log_a), state, parameters, initial)
        for log_a, state in zip(solution.t, solution.y.T)
    ]
    expansion_descending = np.asarray(
        [background["H_GeV"] / h0_gev for background in background_history]
    )
    expansion = expansion_descending[::-1]
    if np.any(~np.isfinite(expansion)) or np.any(expansion <= 0.0):
        raise ValueError("Scalar-clock expansion rate left the positive finite domain.")
    if return_history:
        rho_N = np.asarray(
            [background["rho_N_GeV4"] for background in background_history]
        )
        pressure_N = np.asarray(
            [background["pressure_N_GeV4"] for background in background_history]
        )
        if np.any(~np.isfinite(rho_N)) or np.any(rho_N <= 0.0):
            raise ValueError("The scalar energy density must stay positive and finite.")
        history = {
            "scale_factor": np.asarray(
                [background["scale_factor"] for background in background_history]
            ),
            "H_over_H0": expansion_descending,
            "N_GeV": solution.y[0].copy(),
            "Ndot_GeV2": solution.y[1].copy(),
            "rho_N_GeV4": rho_N,
            "pressure_N_GeV4": pressure_N,
            "w_N": pressure_N / rho_N,
            "N_initial_GeV": float(N_initial),
        }
        return expansion, float(N_initial), history
    return expansion, float(N_initial)


def scalar_clock_history(
    scale_factors,
    h0_km_s_mpc,
    omega_m,
    mass_ratio,
    *,
    g_GeV_inv=0.0,
    h_GeV_inv=0.0,
):
    """Return the canonical scalar history on an increasing scale-factor grid."""
    scale_factors = np.asarray(scale_factors, dtype=float)
    if (
        scale_factors.ndim != 1
        or len(scale_factors) < 2
        or np.any(~np.isfinite(scale_factors))
        or np.any(scale_factors <= 0.0)
        or np.any(np.diff(scale_factors) <= 0.0)
        or scale_factors[0] < EARLY_SCALE_FACTOR
        or scale_factors[-1] > 1.0
        or not np.isclose(scale_factors[-1], 1.0, rtol=0.0, atol=1e-14)
    ):
        raise ValueError(
            "Require an increasing scale-factor grid from at least the initial "
            "scale factor through a=1."
        )
    scale_factors[-1] = 1.0
    redshift_grid = 1.0 / scale_factors[::-1] - 1.0
    _expansion, _N_initial, history = scalar_clock_expansion(
        redshift_grid,
        h0_km_s_mpc,
        omega_m,
        mass_ratio,
        g_GeV_inv=g_GeV_inv,
        h_GeV_inv=h_GeV_inv,
        return_history=True,
    )
    return history


class BackgroundLikelihood:
    def __init__(
        self,
        data,
        h0_pivot_km_s_mpc=H0_PIVOT_KM_S_MPC,
        grid_size=1201,
    ):
        if not np.isfinite(h0_pivot_km_s_mpc) or h0_pivot_km_s_mpc <= 0.0:
            raise ValueError("The H0 conversion pivot must be finite and positive.")
        if grid_size < 101:
            raise ValueError("grid_size must be at least 101 points.")
        self.data = data
        self.h0_pivot_km_s_mpc = float(h0_pivot_km_s_mpc)
        self.redshift_max = float(max(np.max(data.sn_z_hd), np.max(data.bao_z)))
        self.redshift_grid = np.linspace(0.0, self.redshift_max, grid_size)

    def _expansion(self, model, parameters):
        if model == "lcdm":
            omega_m, _alpha_bao = parameters
            expansion = lcdm_expansion(
                self.redshift_grid, self.h0_pivot_km_s_mpc, omega_m
            )
            return expansion, None
        if model == "neutromatherion":
            omega_m, log10_mass_ratio, _alpha_bao = parameters
            expansion, N_initial = scalar_clock_expansion(
                self.redshift_grid,
                self.h0_pivot_km_s_mpc,
                omega_m,
                10.0**log10_mass_ratio,
            )
            return expansion, N_initial
        raise ValueError(f"Unknown background model: {model}")

    def residual_parts(self, model, parameters):
        expansion, N_initial = self._expansion(model, parameters)
        integral = cumulative_trapezoid(
            1.0 / expansion, self.redshift_grid, initial=0.0
        )
        sn_integral = np.interp(self.data.sn_z_hd, self.redshift_grid, integral)
        luminosity_distance_mpc = (
            (1.0 + self.data.sn_z_hel)
            * SPEED_OF_LIGHT_KM_S
            / self.h0_pivot_km_s_mpc
            * sn_integral
        )
        if np.any(luminosity_distance_mpc <= 0.0):
            raise ValueError("The supernova luminosity distances must be positive.")
        distance_modulus = 5.0 * np.log10(luminosity_distance_mpc) + 25.0
        sn_raw = self.data.sn_magnitude - distance_modulus
        sn_whitened_raw = solve_triangular(
            self.data.sn_cholesky,
            sn_raw,
            lower=True,
            check_finite=False,
        )
        magnitude_intercept = float(
            np.dot(self.data.sn_whitened_one, sn_whitened_raw)
            / self.data.sn_one_norm
        )
        sn_residual = (
            sn_whitened_raw
            - magnitude_intercept * self.data.sn_whitened_one
        )

        bao_integral = np.interp(self.data.bao_z, self.redshift_grid, integral)
        bao_expansion = np.interp(
            self.data.bao_z, self.redshift_grid, expansion
        )
        alpha_bao = float(parameters[-1])
        bao_dm = alpha_bao * bao_integral
        bao_dh = alpha_bao / bao_expansion
        bao_dv = alpha_bao * np.cbrt(
            self.data.bao_z * bao_integral**2 / bao_expansion
        )
        bao_predictions = np.asarray(
            [
                {
                    "DV_over_rs": bao_dv[index],
                    "DM_over_rs": bao_dm[index],
                    "DH_over_rs": bao_dh[index],
                }[kind]
                for index, kind in enumerate(self.data.bao_kinds)
            ]
        )
        bao_residual = solve_triangular(
            self.data.bao_cholesky,
            bao_predictions - self.data.bao_values,
            lower=True,
            check_finite=False,
        )
        return sn_residual, bao_residual, magnitude_intercept, N_initial

    def residual_vector(self, model, parameters):
        sn_residual, bao_residual, _intercept, _N_initial = self.residual_parts(
            model, parameters
        )
        return np.concatenate((sn_residual, bao_residual))

    def fit_model(self, model):
        if model == "lcdm":
            lower = np.asarray((0.10, 20.0))
            upper = np.asarray((0.60, 40.0))
            starts = (np.asarray((0.30, 30.0)), np.asarray((0.25, 31.0)))
        elif model == "neutromatherion":
            lower = np.asarray((0.10, -2.0, 20.0))
            upper = np.asarray((0.60, log10(2.0), 40.0))
            starts = (
                np.asarray((0.30, np.log10(0.2), 30.0)),
                np.asarray((0.25, -1.0, 31.0)),
                np.asarray((0.35, -0.3, 29.0)),
            )
        else:
            raise ValueError(f"Unknown background model: {model}")

        def safe_residuals(values):
            try:
                return self.residual_vector(model, values)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                return np.full(self.data.point_count, 1e6)

        attempts = [
            least_squares(
                safe_residuals,
                start,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=120,
            )
            for start in starts
        ]
        best = min(attempts, key=lambda result: 2.0 * result.cost)
        sn_residual, bao_residual, intercept, N_initial = self.residual_parts(
            model, best.x
        )
        chi2_sn = float(np.dot(sn_residual, sn_residual))
        chi2_bao = float(np.dot(bao_residual, bao_residual))
        chi2 = chi2_sn + chi2_bao
        information = best.jac.T @ best.jac
        if np.linalg.matrix_rank(best.jac) == len(best.x):
            parameter_covariance = np.linalg.solve(
                information, np.eye(len(best.x))
            )
            parameter_errors = np.sqrt(
                np.maximum(0.0, np.diag(parameter_covariance))
            )
        else:
            parameter_errors = np.full(len(best.x), np.inf)
        parameter_count = len(best.x) + 1
        point_count = self.data.point_count
        return {
            "model": model,
            "parameters": best.x,
            "parameter_errors": parameter_errors,
            "chi2_sn": chi2_sn,
            "chi2_bao": chi2_bao,
            "chi2": chi2,
            "magnitude_intercept": intercept,
            "N_initial_GeV": N_initial,
            "parameter_count": parameter_count,
            "degrees_of_freedom": point_count - parameter_count,
            "reduced_chi2": chi2 / (point_count - parameter_count),
            "aic_score": chi2 + 2.0 * parameter_count,
            "bic_score": chi2 + parameter_count * log(point_count),
            "point_count": point_count,
            "success": bool(best.success),
            "message": best.message,
            "evaluations": int(best.nfev),
            "bounds": (lower, upper),
        }

    def profile_scalar_mass(
        self,
        fit_result,
        mass_ratio_min=0.2,
        mass_ratio_max=2.0,
        profile_points=19,
    ):
        """Profile chi-square over m_phi/H0, refitting Omega_m and BAO scale."""
        if fit_result["model"] != "neutromatherion":
            raise ValueError("The scalar mass profile requires a scalar-clock fit.")
        if (
            not np.isfinite(mass_ratio_min)
            or not np.isfinite(mass_ratio_max)
            or mass_ratio_min <= 0.0
            or mass_ratio_max <= mass_ratio_min
            or profile_points < 5
        ):
            raise ValueError("Require positive ordered mass bounds and at least 5 profile points.")

        best_log_mass = float(fit_result["parameters"][1])
        if not log10(mass_ratio_min) < best_log_mass < log10(mass_ratio_max):
            raise ValueError("The fitted mass must lie inside the profile range.")
        cache = {
            best_log_mass: {
                "chi2": float(fit_result["chi2"]),
                "nuisance": np.asarray(
                    (fit_result["parameters"][0], fit_result["parameters"][2])
                ),
                "success": bool(fit_result["success"]),
            }
        }

        def profile_at(log_mass):
            log_mass = float(log_mass)
            if log_mass in cache:
                return cache[log_mass]
            nearest = min(cache, key=lambda value: abs(value - log_mass))

            def residuals(nuisance):
                try:
                    return self.residual_vector(
                        "neutromatherion",
                        (nuisance[0], log_mass, nuisance[1]),
                    )
                except (
                    ValueError,
                    RuntimeError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ):
                    return np.full(self.data.point_count, 1e6)

            fit = least_squares(
                residuals,
                cache[nearest]["nuisance"],
                bounds=((0.10, 20.0), (0.60, 40.0)),
                method="trf",
                x_scale="jac",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=120,
            )
            chi2 = float(2.0 * fit.cost)
            if not np.isfinite(chi2) or chi2 > 1e10:
                raise RuntimeError(
                    "Scalar mass profile failed at "
                    f"m_phi/H0={10.0**log_mass:.6g}: {fit.message}"
                )
            cache[log_mass] = {
                "chi2": chi2,
                "nuisance": fit.x,
                "success": bool(fit.success),
            }
            return cache[log_mass]

        scan_logs = np.log10(
            np.geomspace(mass_ratio_min, mass_ratio_max, profile_points)
        )
        if mass_ratio_min <= 1.0 <= mass_ratio_max:
            scan_logs = np.append(scan_logs, 0.0)
        scan_logs = np.unique(scan_logs)
        for log_mass in sorted(scan_logs, key=lambda value: abs(value - best_log_mass)):
            profile_at(log_mass)

        minimum_log_mass = min(cache, key=lambda value: cache[value]["chi2"])
        minimum_chi2 = cache[minimum_log_mass]["chi2"]

        def find_endpoint(delta_chi2, side):
            ordered_logs = sorted(cache)
            brackets = []
            for left, right in zip(ordered_logs[:-1], ordered_logs[1:]):
                left_delta = cache[left]["chi2"] - minimum_chi2 - delta_chi2
                right_delta = cache[right]["chi2"] - minimum_chi2 - delta_chi2
                if left_delta * right_delta > 0.0:
                    continue
                if side == "lower" and right <= minimum_log_mass:
                    brackets.append((left, right))
                if side == "upper" and left >= minimum_log_mass:
                    brackets.append((left, right))
            if not brackets:
                return None
            bracket = (
                max(brackets, key=lambda pair: pair[1])
                if side == "lower"
                else min(brackets, key=lambda pair: pair[0])
            )
            left, right = bracket
            left_delta = cache[left]["chi2"] - minimum_chi2 - delta_chi2
            right_delta = cache[right]["chi2"] - minimum_chi2 - delta_chi2
            if left_delta == 0.0:
                return 10.0**left
            if right_delta == 0.0:
                return 10.0**right
            fraction = left_delta / (left_delta - right_delta)
            crossing = left + fraction * (right - left)
            return 10.0**crossing

        thresholds = {"68.3%": 1.0, "95%": 3.84}
        intervals = {
            label: (
                find_endpoint(threshold, "lower"),
                find_endpoint(threshold, "upper"),
            )
            for label, threshold in thresholds.items()
        }
        return {
            "minimum_mass_ratio": 10.0**minimum_log_mass,
            "minimum_chi2": minimum_chi2,
            "intervals": intervals,
            "mass_ratio_range": (mass_ratio_min, mass_ratio_max),
            "optimizer_success": all(item["success"] for item in cache.values()),
        }


def run_background_fit(
    data_dir=DEFAULT_DATA_DIR,
    h0_pivot_km_s_mpc=H0_PIVOT_KM_S_MPC,
    grid_size=1201,
):
    data = load_background_data(data_dir)
    likelihood = BackgroundLikelihood(
        data,
        h0_pivot_km_s_mpc=h0_pivot_km_s_mpc,
        grid_size=grid_size,
    )
    lcdm = likelihood.fit_model("lcdm")
    scalar_clock = likelihood.fit_model("neutromatherion")
    scalar_mass_profile = likelihood.profile_scalar_mass(scalar_clock)
    delta_chi2 = scalar_clock["chi2"] - lcdm["chi2"]
    delta_aic = scalar_clock["aic_score"] - lcdm["aic_score"]
    delta_bic = scalar_clock["bic_score"] - lcdm["bic_score"]

    print("Background-only fit: Pantheon+ Only + DESI DR2 combined BAO")
    print(
        f"Pantheon+: {len(data.sn_z_hd)} SNe (zHD > 0.01), full STAT+SYS covariance; "
        f"DESI: {len(data.bao_z)} correlated measurements."
    )
    print(
        f"H0 conversion pivot: {h0_pivot_km_s_mpc:.1f} km/s/Mpc (not fitted); "
        "the free BAO scale is H0*r_d."
    )
    print(
        "The scalar branch fixes lambda_N=0, rho_Lambda=0, g=h=1e-40 GeV^-1, "
        "and Omega_Theta=1e-30; it fits Omega_m and m_phi/H0."
    )
    print("Planck TT and lensing are not included in this background likelihood.")
    print(
        "Parameter errors are local Gaussian 1-sigma estimates from the "
        "profiled residual Jacobian; covariance normalization is treated as known."
    )
    for result in (lcdm, scalar_clock):
        print(f"\n{result['model']}:")
        print(
            f"  chi2_SN={result['chi2_sn']:.3f}, "
            f"chi2_BAO={result['chi2_bao']:.3f}, "
            f"chi2_total={result['chi2']:.3f}, "
            f"chi2/dof={result['reduced_chi2']:.5f}"
        )
        omega_m_error = result["parameter_errors"][0]
        alpha_bao = result["parameters"][-1]
        alpha_bao_error = result["parameter_errors"][-1]
        h0_rd = SPEED_OF_LIGHT_KM_S / alpha_bao
        h0_rd_error = SPEED_OF_LIGHT_KM_S * alpha_bao_error / alpha_bao**2
        print(
            f"  Omega_m={result['parameters'][0]:.6f} +/- {omega_m_error:.6f}, "
            f"H0*r_d={h0_rd:.2f} +/- {h0_rd_error:.2f} km/s, "
            f"profiled M={result['magnitude_intercept']:.5f}"
        )
        if result["model"] == "neutromatherion":
            mass_ratio = 10.0 ** result["parameters"][1]
            log_mass_error = result["parameter_errors"][1]
            mass_interval = (
                10.0 ** (result["parameters"][1] - log_mass_error),
                10.0 ** (result["parameters"][1] + log_mass_error),
            )
            print(
                f"  m_phi/H0={mass_ratio:.6g} "
                f"(local 1-sigma interval {mass_interval[0]:.6g} to "
                f"{mass_interval[1]:.6g})"
            )
            lower, upper = result["bounds"]
            if mass_interval[0] < 10.0 ** lower[1] or mass_interval[1] > 10.0 ** upper[1]:
                print("  Note: the local Gaussian interval extends beyond a search bound.")
            if np.isclose(result["parameters"][1], lower[1], atol=1e-4):
                print("  Note: m_phi/H0 reached the lower search bound.")
            elif np.isclose(result["parameters"][1], upper[1], atol=1e-4):
                print("  Note: m_phi/H0 reached the upper search bound.")
        print(
            f"  AIC*={result['aic_score']:.3f}, BIC*={result['bic_score']:.3f}, "
            f"optimizer_success={result['success']}"
        )
    print(
        "\nScalar mass profile (Omega_m and BAO scale reoptimized; "
        f"m_phi/H0 range {scalar_mass_profile['mass_ratio_range'][0]:.2f} to "
        f"{scalar_mass_profile['mass_ratio_range'][1]:.2f}):"
    )
    print(
        f"  minimum at m_phi/H0={scalar_mass_profile['minimum_mass_ratio']:.6g}, "
        f"chi2={scalar_mass_profile['minimum_chi2']:.3f}"
    )
    for label, (lower, upper) in scalar_mass_profile["intervals"].items():
        lower_text = "open" if lower is None else f"{lower:.6g}"
        upper_text = "open" if upper is None else f"{upper:.6g}"
        print(f"  {label} profile interval: {lower_text} to {upper_text}")
    print(
        f"  profile_optimizer_success={scalar_mass_profile['optimizer_success']}"
    )
    print(
        "\nScalar-clock minus LCDM: "
        f"Delta chi2={delta_chi2:+.3f}, "
        f"Delta AIC*={delta_aic:+.3f}, Delta BIC*={delta_bic:+.3f}."
    )
    print("AIC*/BIC* omit Gaussian constants common to both fits; differences are valid.")
    return {
        "lcdm": lcdm,
        "neutromatherion": scalar_clock,
        "scalar_mass_profile": scalar_mass_profile,
    }


if __name__ == "__main__":
    run_background_fit()