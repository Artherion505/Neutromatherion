"""Homogeneous flat-FLRW branch of the scalar-clock model.

Units are natural (c = hbar = 1), with densities in GeV^4 and H in GeV.
The matter sector is comoving pressureless dust plus an adiabatic isotropic
photon gas, with optional collisionless thermal relic neutrinos. Photon
production and non-adiabatic mode mixing are neglected.
Relic neutrinos retain a collisionless thermal distribution and use the
first-order local dispersion relation; their homogeneous scalar backreaction
is included, while perturbation stress closure remains pending.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


BOLTZMANN_GEV_K = 8.617333262e-14
TCMB0_K = 2.7255
T_NU0_GEV = BOLTZMANN_GEV_K * TCMB0_K * (4.0 / 11.0) ** (1.0 / 3.0)
_NEUTRINO_NODES, _NEUTRINO_WEIGHTS = np.polynomial.legendre.leggauss(96)
_NEUTRINO_X = 20.0 * (_NEUTRINO_NODES + 1.0)
_NEUTRINO_WEIGHTS = (
    20.0 * _NEUTRINO_WEIGHTS / (np.exp(_NEUTRINO_X) + 1.0)
)


@dataclass(frozen=True)
class FLRWParameters:
    m_phi_GeV: float
    lambda_N: float
    g_GeV_inv: float
    h_GeV_inv: float
    rho_lambda_GeV4: float = 0.0
    Z_theta: float = 1.0
    Mpl_GeV: float = 2.4353234593382e18
    neutrino_masses_GeV: tuple = ()
    N_eff: float = 3.046
    T_nu0_GeV: float = T_NU0_GEV

    def __post_init__(self):
        values = (
            self.m_phi_GeV,
            self.lambda_N,
            self.g_GeV_inv,
            self.h_GeV_inv,
            self.rho_lambda_GeV4,
            self.Z_theta,
            self.Mpl_GeV,
            self.N_eff,
            self.T_nu0_GeV,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("All model parameters must be finite.")
        if self.m_phi_GeV < 0 or self.lambda_N < 0:
            raise ValueError("m_phi and lambda_N must be nonnegative.")
        if self.Z_theta <= 0 or self.Mpl_GeV <= 0:
            raise ValueError("Z_theta and Mpl must be positive.")
        if self.N_eff <= 0 or self.T_nu0_GeV <= 0:
            raise ValueError("N_eff and the present neutrino temperature must be positive.")
        try:
            neutrino_masses = tuple(float(mass) for mass in self.neutrino_masses_GeV)
        except (TypeError, ValueError) as error:
            raise ValueError("Neutrino masses must be a sequence of three values.") from error
        if len(neutrino_masses) not in (0, 3):
            raise ValueError("Provide either no neutrino masses or three mass eigenvalues.")
        if neutrino_masses and self.N_eff < len(neutrino_masses):
            raise ValueError("N_eff must be at least three when three neutrino states are supplied.")
        if any(not np.isfinite(mass) or mass < 0 for mass in neutrino_masses):
            raise ValueError("Neutrino masses must be finite and nonnegative.")
        object.__setattr__(self, "neutrino_masses_GeV", neutrino_masses)

    @property
    def eta_GeV_inv(self):
        return self.g_GeV_inv - self.h_GeV_inv


@dataclass(frozen=True)
class FLRWInitialConditions:
    N_GeV: float
    Ndot_GeV2: float
    rho_m_bare_i_GeV4: float
    rho_gamma_i_GeV4: float
    Theta_dot_i_GeV2: float
    Theta_i_GeV: float = 0.0
    t_i_GeVinv: float = 0.0
    scale_factor_i: float = 1.0

    def __post_init__(self):
        values = (
            self.N_GeV,
            self.Ndot_GeV2,
            self.rho_m_bare_i_GeV4,
            self.rho_gamma_i_GeV4,
            self.Theta_dot_i_GeV2,
            self.Theta_i_GeV,
            self.t_i_GeVinv,
            self.scale_factor_i,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("All initial conditions must be finite.")
        if self.rho_m_bare_i_GeV4 < 0 or self.rho_gamma_i_GeV4 < 0:
            raise ValueError("Initial matter and radiation densities must be nonnegative.")
        if self.Theta_dot_i_GeV2 <= 0:
            raise ValueError("Theta_dot must be positive to define the future clock frame.")
        if self.scale_factor_i <= 0:
            raise ValueError("The initial physical scale factor must be positive.")


def potential(N_GeV, parameters):
    return (
        0.5 * parameters.m_phi_GeV**2 * N_GeV**2
        + 0.25 * parameters.lambda_N * N_GeV**4
    )


def potential_derivative(N_GeV, parameters):
    return (
        parameters.m_phi_GeV**2 * N_GeV
        + parameters.lambda_N * N_GeV**3
    )


def photon_speed_ratio(N_GeV, parameters):
    field = np.asarray(N_GeV, dtype=float)
    xi = parameters.h_GeV_inv * field
    if np.any(~np.isfinite(xi)) or np.any(np.abs(xi) >= 1):
        raise ValueError("The local photon sector requires |h N| < 1.")
    speed = np.sqrt((1.0 - xi) / (1.0 + xi))
    return float(speed) if speed.ndim == 0 else speed


def photon_speed_log_derivative(N_GeV, parameters):
    """Return d ln(c_gamma)/dN for the local photon-speed relation."""
    photon_speed_ratio(N_GeV, parameters)
    field = np.asarray(N_GeV, dtype=float)
    xi = parameters.h_GeV_inv * field
    derivative = -parameters.h_GeV_inv / (1.0 - xi**2)
    return float(derivative) if derivative.ndim == 0 else derivative


def relic_neutrino_components(scale_factor, parameters, N_GeV=0.0):
    """Return relic-neutrino stress and scalar source to first order in N."""
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("The physical scale factor must be finite and positive.")
    if not np.isfinite(N_GeV):
        raise ValueError("The scalar field value must be finite.")
    masses = parameters.neutrino_masses_GeV
    if not masses:
        return {
            "rho_GeV4": 0.0,
            "pressure_GeV4": 0.0,
            "scalar_source_GeV3": 0.0,
            "rho_uncoupled_GeV4": 0.0,
            "pressure_uncoupled_GeV4": 0.0,
            "rho_plus_pressure_uncoupled_GeV4": 0.0,
            "rho_massive_GeV4": 0.0,
            "pressure_massive_GeV4": 0.0,
            "rho_massless_GeV4": 0.0,
            "pressure_massless_GeV4": 0.0,
        }

    temperature = parameters.T_nu0_GeV / scale_factor
    momentum = _NEUTRINO_X
    rho_massless_per_species = (
        temperature**4
        * np.dot(_NEUTRINO_WEIGHTS, momentum**3)
        / np.pi**2
    )
    rho_massive = 0.0
    pressure_massive = 0.0
    scalar_source_massive = 0.0
    rho_massive_uncoupled = 0.0
    pressure_massive_uncoupled = 0.0
    rho_massless = 0.0
    pressure_massless = 0.0
    massless_species = 0
    for mass in masses:
        if mass == 0.0:
            massless_species += 1
            continue
        mass_over_temperature = mass * scale_factor / parameters.T_nu0_GeV
        energy_dimensionless = np.sqrt(
            momentum**2 + mass_over_temperature**2
        )
        energy_uncoupled_GeV = temperature * energy_dimensionless
        rho_massive_uncoupled += (
            temperature**3
            * np.dot(
                _NEUTRINO_WEIGHTS,
                momentum**2 * energy_uncoupled_GeV,
            )
            / np.pi**2
        )
        pressure_massive_uncoupled += (
            temperature**4
            * np.dot(
                _NEUTRINO_WEIGHTS,
                momentum**4 / energy_dimensionless,
            )
            / (3.0 * np.pi**2)
        )
        scalar_response = (
            parameters.g_GeV_inv * mass**2 / energy_uncoupled_GeV
            - parameters.h_GeV_inv * energy_uncoupled_GeV
        )
        energy_GeV = energy_uncoupled_GeV + N_GeV * scalar_response
        group_velocity = (momentum / energy_dimensionless) * (
            1.0
            - parameters.g_GeV_inv
            * N_GeV
            * (mass / energy_uncoupled_GeV) ** 2
            - parameters.h_GeV_inv * N_GeV
        )
        if (
            np.any(~np.isfinite(energy_GeV))
            or np.any(energy_GeV < 0.0)
            or np.any(~np.isfinite(group_velocity))
            or np.any(group_velocity < 0.0)
        ):
            raise ValueError("The first-order neutrino dispersion is not physical.")
        rho_massive += (
            temperature**3
            * np.dot(_NEUTRINO_WEIGHTS, momentum**2 * energy_GeV)
            / np.pi**2
        )
        pressure_massive += (
            temperature**4
            * np.dot(_NEUTRINO_WEIGHTS, momentum**3 * group_velocity)
            / (3.0 * np.pi**2)
        )
        scalar_source_massive += (
            temperature**3
            * np.dot(
                _NEUTRINO_WEIGHTS, momentum**2 * scalar_response
            )
            / np.pi**2
        )

    additional_massless_species = parameters.N_eff - len(masses)
    massless_species += additional_massless_species
    massless_energy_factor = 1.0 - parameters.h_GeV_inv * N_GeV
    rho_massless = (
        massless_species * rho_massless_per_species * massless_energy_factor
    )
    pressure_massless = rho_massless / 3.0
    scalar_source_massless = (
        -parameters.h_GeV_inv
        * massless_species
        * rho_massless_per_species
    )
    rho_massless_uncoupled = massless_species * rho_massless_per_species
    pressure_massless_uncoupled = rho_massless_uncoupled / 3.0
    rho_uncoupled = rho_massive_uncoupled + rho_massless_uncoupled
    pressure_uncoupled = (
        pressure_massive_uncoupled + pressure_massless_uncoupled
    )

    return {
        "rho_GeV4": rho_massive + rho_massless,
        "pressure_GeV4": pressure_massive + pressure_massless,
        "scalar_source_GeV3": scalar_source_massive + scalar_source_massless,
        "rho_uncoupled_GeV4": rho_uncoupled,
        "pressure_uncoupled_GeV4": pressure_uncoupled,
        "rho_plus_pressure_uncoupled_GeV4": (
            rho_uncoupled + pressure_uncoupled
        ),
        "rho_massive_GeV4": rho_massive,
        "pressure_massive_GeV4": pressure_massive,
        "rho_massless_GeV4": rho_massless,
        "pressure_massless_GeV4": pressure_massless,
    }


def relic_neutrino_energy_pressure(scale_factor, parameters, N_GeV=0.0):
    """Return total collisionless thermal relic-neutrino density and pressure."""
    components = relic_neutrino_components(scale_factor, parameters, N_GeV)
    return components["rho_GeV4"], components["pressure_GeV4"]


def background_quantities(log_a, state, parameters, initial):
    """Return the algebraic FLRW densities and expanding-branch Hubble rate."""
    state = np.asarray(state, dtype=float)
    if state.shape != (4,):
        raise ValueError("The state must contain N, Ndot, Theta, and cosmic time.")
    N_GeV, Ndot_GeV2, Theta_GeV, time_GeVinv = state
    if not np.isfinite(log_a) or not np.all(np.isfinite(state)):
        raise ValueError("The background state must be finite.")

    scale = np.exp(log_a)
    physical_scale_factor = initial.scale_factor_i * scale
    eta = parameters.eta_GeV_inv
    matter_factor = 1.0 + eta * N_GeV
    if matter_factor <= 0:
        raise ValueError("The effective rest mass of comoving dust must stay positive.")

    speed = photon_speed_ratio(N_GeV, parameters)
    initial_speed = photon_speed_ratio(initial.N_GeV, parameters)
    xi = parameters.h_GeV_inv * N_GeV
    rho_m_bare = initial.rho_m_bare_i_GeV4 / scale**3
    rho_m = matter_factor * rho_m_bare
    rho_gamma = (
        initial.rho_gamma_i_GeV4 * speed / initial_speed / scale**4
    )
    rho_gamma_bare = rho_gamma / (1.0 - xi**2)
    pressure_gamma_bare = rho_gamma_bare / 3.0
    theta_dot = initial.Theta_dot_i_GeV2 / scale**3
    rho_theta = 0.5 * parameters.Z_theta * theta_dot**2
    rho_N = 0.5 * Ndot_GeV2**2 + potential(N_GeV, parameters)
    neutrinos = relic_neutrino_components(
        physical_scale_factor, parameters, N_GeV
    )
    rho_neutrino = neutrinos["rho_GeV4"]
    pressure_neutrino = neutrinos["pressure_GeV4"]
    rho_lambda = parameters.rho_lambda_GeV4
    rho_total = rho_N + rho_theta + rho_m + rho_gamma + rho_neutrino + rho_lambda
    if not np.isfinite(rho_total) or rho_total <= 0:
        raise ValueError("The flat expanding branch requires positive total density.")

    H_GeV = np.sqrt(rho_total / (3.0 * parameters.Mpl_GeV**2))
    return {
        "a_over_ai": scale,
        "scale_factor": physical_scale_factor,
        "H_GeV": H_GeV,
        "Theta_GeV": Theta_GeV,
        "cosmic_time_GeVinv": time_GeVinv,
        "xi": xi,
        "photon_speed_ratio": speed,
        "Theta_dot_GeV2": theta_dot,
        "rho_N_GeV4": rho_N,
        "rho_theta_GeV4": rho_theta,
        "rho_m_bare_GeV4": rho_m_bare,
        "rho_m_GeV4": rho_m,
        "rho_gamma_bare_GeV4": rho_gamma_bare,
        "pressure_gamma_bare_GeV4": pressure_gamma_bare,
        "rho_gamma_GeV4": rho_gamma,
        "rho_neutrino_GeV4": rho_neutrino,
        "scalar_source_neutrino_GeV3": neutrinos["scalar_source_GeV3"],
        "rho_neutrino_uncoupled_GeV4": neutrinos["rho_uncoupled_GeV4"],
        "pressure_neutrino_uncoupled_GeV4": neutrinos[
            "pressure_uncoupled_GeV4"
        ],
        "rho_plus_pressure_neutrino_uncoupled_GeV4": neutrinos[
            "rho_plus_pressure_uncoupled_GeV4"
        ],
        "rho_plus_pressure_matter_uncoupled_GeV4": (
            rho_m_bare
            + rho_gamma_bare
            + pressure_gamma_bare
            + neutrinos["rho_plus_pressure_uncoupled_GeV4"]
        ),
        "rho_neutrino_massive_GeV4": neutrinos["rho_massive_GeV4"],
        "rho_neutrino_massless_GeV4": neutrinos["rho_massless_GeV4"],
        "rho_lambda_GeV4": rho_lambda,
        "rho_total_GeV4": rho_total,
        "pressure_N_GeV4": 0.5 * Ndot_GeV2**2 - potential(N_GeV, parameters),
        "pressure_theta_GeV4": rho_theta,
        "pressure_gamma_GeV4": rho_gamma / 3.0,
        "pressure_neutrino_GeV4": pressure_neutrino,
        "pressure_neutrino_massive_GeV4": neutrinos[
            "pressure_massive_GeV4"
        ],
        "pressure_neutrino_massless_GeV4": neutrinos[
            "pressure_massless_GeV4"
        ],
        "pressure_lambda_GeV4": -rho_lambda,
    }


def rhs_log_scale_factor(log_a, state, parameters, initial):
    """Evolution equations for (N, dN/dt, Theta, t) with x = ln(a/a_i)."""
    N_GeV, Ndot_GeV2, _theta_GeV, _time_GeVinv = np.asarray(state, dtype=float)
    background = background_quantities(log_a, state, parameters, initial)
    H_GeV = background["H_GeV"]
    scalar_source = (
        parameters.eta_GeV_inv * background["rho_m_bare_GeV4"]
        - parameters.h_GeV_inv * background["rho_gamma_bare_GeV4"]
        + background["scalar_source_neutrino_GeV3"]
    )
    Nddot_GeV3 = (
        -3.0 * H_GeV * Ndot_GeV2
        - potential_derivative(N_GeV, parameters)
        - scalar_source
    )
    return np.array(
        (
            Ndot_GeV2 / H_GeV,
            Nddot_GeV3 / H_GeV,
            background["Theta_dot_GeV2"] / H_GeV,
            1.0 / H_GeV,
        )
    )


def solve_background(
    parameters,
    initial,
    log_a_span,
    *,
    method="DOP853",
    rtol=1e-9,
    atol=1e-12,
    max_step=np.inf,
    t_eval=None,
):
    """Integrate the coupled background in x = ln(a/a_i).

    Adjust ``atol`` to the field units and scales of a particular run. The
    initial conditions are defined at x = 0. The returned SciPy solution has
    N, Ndot, Theta, and cosmic time as its state components.
    """
    if len(log_a_span) != 2 or not all(np.isfinite(value) for value in log_a_span):
        raise ValueError("log_a_span must contain two finite endpoints.")
    if log_a_span[0] == log_a_span[1]:
        raise ValueError("log_a_span endpoints must differ.")
    if log_a_span[0] != 0.0:
        raise ValueError("Initial conditions are defined at log_a = 0.")
    atol_values = np.asarray(atol, dtype=float)
    if atol_values.ndim > 1 or (
        atol_values.ndim == 1 and atol_values.shape != (4,)
    ):
        raise ValueError("atol must be a positive scalar or a four-component sequence.")
    if not np.all(np.isfinite(atol_values)) or np.any(atol_values <= 0):
        raise ValueError("Absolute tolerances must be finite and positive.")
    if not np.isfinite(rtol) or rtol <= 0 or np.isnan(max_step) or max_step <= 0:
        raise ValueError("Solver tolerances and max_step must be positive.")

    initial_state = np.array(
        (
            initial.N_GeV,
            initial.Ndot_GeV2,
            initial.Theta_i_GeV,
            initial.t_i_GeVinv,
        ),
        dtype=float,
    )
    background_quantities(log_a_span[0], initial_state, parameters, initial)
    return solve_ivp(
        rhs_log_scale_factor,
        log_a_span,
        initial_state,
        args=(parameters, initial),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        t_eval=t_eval,
    )