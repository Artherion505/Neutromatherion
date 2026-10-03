"""Leading-order matter-sector blocks for FLRW perturbations.

Use signature (-,+,+,+), a Maxwell action, and collisionless neutrino
worldlines coupled through the uncoupled stress tensor in the trial action.
For photons, the interaction gives the local-frame Lagrangian
    L_gamma = ((1 + h N) E^2 - (1 - h N) B^2) / 2,
so c_gamma^2 = (1 - h N) / (1 + h N). Equivalently, with
e^mu = u^alpha F_alpha^mu, the excitation tensor is
    P^mu_nu = (1 - h N) F^mu_nu - 2 h N (u^mu e^nu - u^nu e^mu),
and Maxwell's equations are nabla_mu P^mu nu = 0 and dF = 0.

For a massive collisionless particle, the explicit worldline action is
    S = -m integral d tau [1 + g N - h N (u.U)^2].
Its local-frame Hamiltonian at fixed uncoupled momentum is expanded only to
first order in g N and h N. These are local kinetic blocks, not a complete
Einstein-Boltzmann solver.
"""

from collections.abc import Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from flrw_coupled import (
    photon_speed_log_derivative,
    photon_speed_ratio,
    potential_derivative,
)


def _uncoupled_particle_energy(momentum_GeV, mass_GeV):
    if (
        not np.isfinite(momentum_GeV)
        or not np.isfinite(mass_GeV)
        or momentum_GeV < 0.0
        or mass_GeV < 0.0
    ):
        raise ValueError("Momentum and mass must be finite and nonnegative.")
    return float(np.hypot(momentum_GeV, mass_GeV))


def neutrino_scalar_source_per_particle(
    momentum_GeV, mass_GeV, parameters
):
    """Return dE_nu/dN at fixed local momentum, to first order in gN,hN."""
    energy = _uncoupled_particle_energy(momentum_GeV, mass_GeV)
    if energy == 0.0:
        return 0.0
    return (
        parameters.g_GeV_inv * mass_GeV**2 / energy
        - parameters.h_GeV_inv * energy
    )


def neutrino_scalar_source_from_distribution_multipoles(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=1.0,
):
    """Return the delta-f part of delta S_N at fixed local momentum.

    Metric-induced momentum/measure variations and clock-frame terms are not
    included in this distribution-only contribution.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or moments.ndim != 2
        or moments.shape[0] != momenta.size
        or moments.shape[1] == 0
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(moments))
        or not np.isfinite(mass_GeV)
        or mass_GeV < 0.0
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Neutrino source inputs must be finite and physical.")

    particle_sources = np.asarray(
        [
            neutrino_scalar_source_per_particle(momentum, mass_GeV, parameters)
            for momentum in momenta
        ]
    )
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2
    )
    return np.sum(phase_space_weights * particle_sources * moments[:, 0])


def neutrino_scalar_source_spatial_metric_measure_response(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    curvature_perturbation,
    background_distribution,
    parameters,
    degeneracy=1.0,
):
    """Return the neutrino delta-S_N response to local p and measure shifts.

    At fixed comoving momentum, p -> p/(1 - Phi) and the local phase-space
    measure scales as (1 - Phi)^(-3). The source integrand is therefore
    varied as Phi * (3 s_N + p d s_N/dp), with s_N = dE_nu/dN.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(mass_GeV) != 0
        or not np.isreal(mass_GeV)
        or not np.isfinite(mass_GeV)
        or mass_GeV < 0.0
        or np.ndim(curvature_perturbation) != 0
        or not np.isfinite(curvature_perturbation)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Neutrino source metric inputs must be physical.")

    energies = np.hypot(momenta, mass_GeV)
    particle_sources = np.divide(
        parameters.g_GeV_inv * mass_GeV**2,
        energies,
        out=np.zeros_like(momenta),
        where=energies > 0.0,
    ) - parameters.h_GeV_inv * energies
    particle_source_derivatives = -np.divide(
        parameters.g_GeV_inv * mass_GeV**2 * momenta,
        energies**3,
        out=np.zeros_like(momenta),
        where=energies > 0.0,
    ) - parameters.h_GeV_inv * np.divide(
        momenta,
        energies,
        out=np.zeros_like(momenta),
        where=energies > 0.0,
    )
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    return curvature_perturbation * np.sum(
        phase_space_weights
        * (3.0 * particle_sources + momenta * particle_source_derivatives)
    )


def neutrino_scalar_source_perturbation(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    distribution_multipoles,
    background_distribution,
    curvature_perturbation,
    parameters,
    degeneracy=1.0,
):
    """Compose the linear neutrino scalar source in the local normal frame.

    Includes the delta-f monopole and spatial metric measure response. The
    first-order worldline source has no explicit delta-N response, and its
    clock-relative-velocity dependence starts at quadratic order.
    """
    return neutrino_scalar_source_from_distribution_multipoles(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        mass_GeV,
        distribution_multipoles,
        parameters,
        degeneracy,
    ) + neutrino_scalar_source_spatial_metric_measure_response(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        mass_GeV,
        curvature_perturbation,
        background_distribution,
        parameters,
        degeneracy,
    )


def coupled_pressureless_matter_scalar_source_perturbation(
    bare_energy_density_GeV4,
    density_contrast,
    parameters,
):
    """Return delta-S_N for clock-aligned coupled pressureless matter.

    The worldline source per unit bare rest energy is eta = g - h. Its
    relative-clock-velocity correction is quadratic about aligned backgrounds.
    """
    if (
        np.ndim(bare_energy_density_GeV4) != 0
        or not np.isreal(bare_energy_density_GeV4)
        or not np.isfinite(bare_energy_density_GeV4)
        or bare_energy_density_GeV4 < 0.0
        or np.ndim(density_contrast) != 0
        or not np.isfinite(density_contrast)
    ):
        raise ValueError("Coupled dust source inputs must be finite and physical.")
    return (
        parameters.eta_GeV_inv
        * bare_energy_density_GeV4
        * density_contrast
    )


def photon_scalar_source_from_distribution_multipoles(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=2.0,
):
    """Return the fixed-local-momentum delta-f part of the photon source."""
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or moments.ndim != 2
        or moments.shape[0] != momenta.size
        or moments.shape[1] == 0
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(moments))
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon source inputs must be finite and physical.")

    speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    particle_source = (
        -parameters.h_GeV_inv * speed * momenta / (1.0 - xi**2)
    )
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2
    )
    return np.sum(phase_space_weights * particle_source * moments[:, 0])


def photon_scalar_source_background_response(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    delta_N_GeV,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the fixed-local-momentum f0*delta-N photon source response."""
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or not np.isfinite(N_background_GeV)
        or np.ndim(delta_N_GeV) != 0
        or not np.isfinite(delta_N_GeV)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon source-response inputs must be physical.")

    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    source_response = (
        parameters.h_GeV_inv**2
        * photon_speed
        * momenta
        * (1.0 - 2.0 * xi)
        / (1.0 - xi**2) ** 2
    )
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    return delta_N_GeV * np.sum(phase_space_weights * source_response)


def photon_scalar_source_spatial_metric_measure_response(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    curvature_perturbation,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the f0 photon source response to spatial momentum and measure.

    The fixed-N photon source per mode is linear in momentum, so the combined
    p -> p/(1 - Phi) and d^3p -> d^3p/(1 - Phi)^3 response is 4 Phi times
    its isotropic background integral.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(N_background_GeV) != 0
        or not np.isfinite(N_background_GeV)
        or np.ndim(curvature_perturbation) != 0
        or not np.isfinite(curvature_perturbation)
        or np.ndim(degeneracy) != 0
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon source metric inputs must be physical.")

    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    source_per_mode = (
        -parameters.h_GeV_inv * photon_speed * momenta / (1.0 - xi**2)
    )
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    return 4.0 * curvature_perturbation * np.sum(
        phase_space_weights * source_per_mode
    )


def photon_scalar_source_perturbation(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    scale_factor,
    N_background_GeV,
    delta_N_GeV,
    curvature_perturbation,
    distribution_multipoles,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Compose the linear photon contribution to delta-S_N.

    This combines the fixed-local-momentum delta-f, f0*delta-N, and
    f0 spatial-momentum/measure responses. The scalar source has no separate
    first-order clock-velocity term for an isotropic background.
    """
    if (
        np.ndim(scale_factor) != 0
        or not np.isreal(scale_factor)
        or not np.isfinite(scale_factor)
        or scale_factor <= 0.0
    ):
        raise ValueError("The photon source scale factor must be positive.")
    physical_momenta = np.asarray(comoving_momenta_GeV, dtype=float) / scale_factor
    physical_weights = (
        np.asarray(comoving_quadrature_weights_GeV, dtype=float) / scale_factor
    )
    return sum_scalar_source_perturbations(
        photon_scalar_source_from_distribution_multipoles(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            distribution_multipoles,
            parameters,
            degeneracy,
        ),
        photon_scalar_source_background_response(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            delta_N_GeV,
            background_distribution,
            parameters,
            degeneracy,
        ),
        photon_scalar_source_spatial_metric_measure_response(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            curvature_perturbation,
            background_distribution,
            parameters,
            degeneracy,
        ),
    )


def scalar_field_perturbation_rhs(
    delta_N_GeV,
    delta_N_prime_GeV2,
    wavenumber_GeV,
    conformal_hubble_GeV,
    scale_factor,
    potential_second_derivative_GeV2,
    N_background_prime_GeV2,
    N_background_second_prime_GeV3,
    psi,
    psi_prime_GeV,
    phi_prime_GeV,
    scalar_source_perturbation_GeV3,
):
    """Return (delta N', delta N'') for the sourced linear N equation.

    The full covariant delta S_N is supplied by the caller; this helper does
    not complete its matter, metric, or clock-frame contributions.
    """
    values = (
        delta_N_GeV,
        delta_N_prime_GeV2,
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        potential_second_derivative_GeV2,
        N_background_prime_GeV2,
        N_background_second_prime_GeV3,
        psi,
        psi_prime_GeV,
        phi_prime_GeV,
        scalar_source_perturbation_GeV3,
    )
    if any(np.ndim(value) != 0 or not np.isfinite(value) for value in values):
        raise ValueError("Scalar-field perturbation inputs must be finite scalars.")
    if wavenumber_GeV < 0.0 or scale_factor <= 0.0:
        raise ValueError("Wavenumber must be nonnegative and scale factor positive.")

    delta_N_second_prime = (
        N_background_prime_GeV2 * (psi_prime_GeV + 3.0 * phi_prime_GeV)
        + 2.0
        * psi
        * (
            N_background_second_prime_GeV3
            + 2.0 * conformal_hubble_GeV * N_background_prime_GeV2
        )
        - scale_factor**2 * scalar_source_perturbation_GeV3
        - 2.0 * conformal_hubble_GeV * delta_N_prime_GeV2
        - (
            wavenumber_GeV**2
            + scale_factor**2 * potential_second_derivative_GeV2
        )
        * delta_N_GeV
    )
    return delta_N_prime_GeV2, delta_N_second_prime


def clock_field_perturbation_rhs(
    delta_Theta_GeV,
    delta_Theta_prime_GeV2,
    wavenumber_GeV,
    conformal_hubble_GeV,
    scale_factor,
    Theta_background_prime_GeV2,
    Theta_background_second_prime_GeV3,
    Z_theta,
    psi,
    psi_prime_GeV,
    phi_prime_GeV,
    clock_current_divergence_perturbation_GeV3,
):
    """Return (delta Theta', delta Theta'') for the sourced clock equation.

    The full covariant perturbation of the interaction-current divergence is
    supplied by the caller; this helper does not compute its matter terms.
    """
    values = (
        delta_Theta_GeV,
        delta_Theta_prime_GeV2,
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        Theta_background_prime_GeV2,
        Theta_background_second_prime_GeV3,
        Z_theta,
        psi,
        psi_prime_GeV,
        phi_prime_GeV,
        clock_current_divergence_perturbation_GeV3,
    )
    if any(np.ndim(value) != 0 or not np.isfinite(value) for value in values):
        raise ValueError("Clock perturbation inputs must be finite scalars.")
    if wavenumber_GeV < 0.0 or scale_factor <= 0.0 or Z_theta <= 0.0:
        raise ValueError(
            "Wavenumber must be nonnegative; scale factor and Z_theta positive."
        )

    delta_Theta_second_prime = (
        Theta_background_prime_GeV2 * (psi_prime_GeV + 3.0 * phi_prime_GeV)
        + 2.0
        * psi
        * (
            Theta_background_second_prime_GeV3
            + 2.0 * conformal_hubble_GeV * Theta_background_prime_GeV2
        )
        + 2.0
        * scale_factor**2
        * clock_current_divergence_perturbation_GeV3
        / Z_theta
        - 2.0 * conformal_hubble_GeV * delta_Theta_prime_GeV2
        - wavenumber_GeV**2 * delta_Theta_GeV
    )
    return delta_Theta_prime_GeV2, delta_Theta_second_prime


def clock_interaction_current_divergence_perturbation(
    wavenumber_GeV,
    h_GeV_inv,
    N_background_GeV,
    Theta_background_prime_GeV2,
    uncoupled_rho_plus_pressure_GeV4,
    uncoupled_energy_flux_projection_GeV4,
    clock_velocity_projection,
):
    """Return delta(div J) from the total linear energy flux and clock speed.

    Q is k_hat_i delta T^(hat 0 hat i) in the orthonormal frame normal to
    constant-time slices. The background stress is assumed isotropic; callers
    must supply complete moments of T^(0), not only distribution terms.
    """
    values = (
        wavenumber_GeV,
        h_GeV_inv,
        N_background_GeV,
        Theta_background_prime_GeV2,
        uncoupled_rho_plus_pressure_GeV4,
        uncoupled_energy_flux_projection_GeV4,
        clock_velocity_projection,
    )
    if any(np.ndim(value) != 0 or not np.isfinite(value) for value in values):
        raise ValueError("Clock-current inputs must be finite scalars.")
    if wavenumber_GeV < 0.0 or Theta_background_prime_GeV2 <= 0.0:
        raise ValueError("Wavenumber must be nonnegative and clock prime positive.")

    relative_flux = (
        -uncoupled_energy_flux_projection_GeV4
        + uncoupled_rho_plus_pressure_GeV4 * clock_velocity_projection
    )
    return (
        1j
        * wavenumber_GeV
        * h_GeV_inv
        * N_background_GeV
        / Theta_background_prime_GeV2
        * relative_flux
    )


def modeled_matter_clock_current_divergence_perturbation(
    wavenumber_GeV,
    N_background_GeV,
    Theta_background_prime_GeV2,
    clock_velocity_projection,
    parameters,
    background,
    total_uncoupled_stress_moments,
):
    """Compose the clock-current source from modeled background and T^(0).

    The supplied stress moments must include all modeled uncoupled matter
    fluxes in the normal orthonormal frame. This does not fill missing
    perturbative stress or baryon/electron sectors.
    """
    if not isinstance(background, Mapping):
        raise ValueError("Background quantities must be a mapping.")
    if not isinstance(total_uncoupled_stress_moments, Mapping):
        raise ValueError("Uncoupled stress moments must be a mapping.")
    enthalpy_key = "rho_plus_pressure_matter_uncoupled_GeV4"
    flux_key = "longitudinal_flux_GeV4"
    if enthalpy_key not in background or flux_key not in total_uncoupled_stress_moments:
        raise ValueError("Background enthalpy and total uncoupled flux are required.")
    return clock_interaction_current_divergence_perturbation(
        wavenumber_GeV,
        parameters.h_GeV_inv,
        N_background_GeV,
        Theta_background_prime_GeV2,
        background[enthalpy_key],
        total_uncoupled_stress_moments[flux_key],
        clock_velocity_projection,
    )


def neutrino_energy_local(momentum_GeV, mass_GeV, N_GeV, parameters):
    """Return the local neutrino energy through first order in gN and hN."""
    if not np.isfinite(N_GeV):
        raise ValueError("The scalar field value must be finite.")
    energy = _uncoupled_particle_energy(momentum_GeV, mass_GeV)
    correction = N_GeV * neutrino_scalar_source_per_particle(
        momentum_GeV, mass_GeV, parameters
    )
    modified_energy = energy + correction
    if not np.isfinite(modified_energy) or modified_energy < 0.0:
        raise ValueError("The first-order neutrino energy is not physical.")
    return modified_energy


def photon_group_velocity_local(N_GeV, parameters):
    """Return the photon group speed in the local clock frame."""
    return photon_speed_ratio(N_GeV, parameters)


def finite_h_thomson_collision_velocity_projection(
    baryon_velocity_projection,
    clock_velocity_projection,
    N_background_GeV,
    parameters,
):
    """Return the normal-frame velocity entering the Thomson dipole source.

    Transforming the photon dipole and electron velocity to the local clock
    frame gives V_collision = V_b/c_gamma + (c_gamma - 1/c_gamma) V_u.
    """
    try:
        baryon_velocity, clock_velocity, field = np.broadcast_arrays(
            np.asarray(baryon_velocity_projection, dtype=complex),
            np.asarray(clock_velocity_projection, dtype=complex),
            np.asarray(N_background_GeV, dtype=float),
        )
    except ValueError as error:
        raise ValueError(
            "Thomson collision velocities and field must be broadcast-compatible."
        ) from error
    if (
        np.any(~np.isfinite(baryon_velocity))
        or np.any(~np.isfinite(clock_velocity))
        or np.any(~np.isfinite(field))
    ):
        raise ValueError("Thomson collision velocities and field must be finite.")

    speed = photon_group_velocity_local(field, parameters)
    collision_velocity = baryon_velocity / speed + (
        speed - 1.0 / speed
    ) * clock_velocity
    if np.any(~np.isfinite(collision_velocity)):
        raise ValueError("The Thomson collision-frame velocity must be finite.")
    return (
        complex(collision_velocity)
        if collision_velocity.ndim == 0
        else collision_velocity
    )


FINE_STRUCTURE_CONSTANT = 1.0 / 137.035999084
ELECTRON_MASS_GEV = 5.1099895e-4
THOMSON_CROSS_SECTION_GEV_INV2 = (
    8.0
    * np.pi
    / 3.0
    * (FINE_STRUCTURE_CONSTANT / ELECTRON_MASS_GEV) ** 2
)


def standard_thomson_opacity_per_conformal_time(
    scale_factor,
    electron_number_density_GeV3,
):
    """Return a*n_e*sigma_T for physical free-electron density n_e.

    The ionization history is supplied by the caller; this helper only
    converts that physical density to the standard conformal scattering rate.
    """
    try:
        scale_factor, electron_density = np.broadcast_arrays(
            np.asarray(scale_factor, dtype=float),
            np.asarray(electron_number_density_GeV3, dtype=float),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Scale factor and electron density must be broadcast-compatible."
        ) from error
    if (
        np.any(~np.isfinite(scale_factor))
        or np.any(scale_factor <= 0.0)
        or np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
    ):
        raise ValueError("Thomson opacity inputs must be finite and physical.")

    opacity = (
        scale_factor
        * electron_density
        * THOMSON_CROSS_SECTION_GEV_INV2
    )
    if np.any(~np.isfinite(opacity)):
        raise ValueError("The standard Thomson opacity must be finite.")
    return float(opacity) if opacity.ndim == 0 else opacity


def finite_h_thomson_cross_section_ratio(N_background_GeV, parameters):
    """Return the stationary-electron Thomson cross-section ratio.

    The charged-particle worldline action with minimal charge coupling gives
    an inertial mass ratio 1 + (g + h)N at small velocity. The modified
    Maxwell outgoing Green function contributes an amplitude factor
    1/(1 - hN); equal incident and scattered impedances then give the squared
    factors below. N is treated as locally constant. This is a
    stationary-target ratio; the linear bulk-velocity dipole is handled
    separately, and recoil corrections are neglected.
    """
    field = np.asarray(N_background_GeV, dtype=float)
    photon_group_velocity_local(field, parameters)
    xi = parameters.h_GeV_inv * field
    electron_inertial_mass_ratio = 1.0 + (
        parameters.g_GeV_inv + parameters.h_GeV_inv
    ) * field
    if (
        np.any(~np.isfinite(electron_inertial_mass_ratio))
        or np.any(electron_inertial_mass_ratio <= 0.0)
    ):
        raise ValueError("The effective electron inertia must stay positive.")
    ratio = 1.0 / (
        (1.0 - xi) ** 2 * electron_inertial_mass_ratio**2
    )
    return float(ratio) if ratio.ndim == 0 else ratio


def finite_h_thomson_opacity_per_conformal_time(
    standard_opacity_per_conformal_time, N_background_GeV, parameters
):
    """Return the stationary-electron Thomson rate per conformal time.

    The cross-section ratio follows from the photon action and the charged
    worldline inertia in the Thomson limit. Multiplication by group speed
    converts it to a rate. The linear moving-electron dipole source is
    handled separately; recoil and second-order velocity terms are omitted.
    """
    try:
        standard_opacity, field = np.broadcast_arrays(
            np.asarray(standard_opacity_per_conformal_time, dtype=float),
            np.asarray(N_background_GeV, dtype=float),
        )
    except ValueError as error:
        raise ValueError(
            "Thomson opacity and field values must be broadcast-compatible."
        ) from error
    if (
        np.any(~np.isfinite(standard_opacity))
        or np.any(standard_opacity < 0.0)
        or np.any(~np.isfinite(field))
    ):
        raise ValueError("Thomson opacity inputs must be finite and physical.")

    photon_speed = photon_group_velocity_local(field, parameters)
    cross_section_ratio = finite_h_thomson_cross_section_ratio(
        field, parameters
    )
    effective_opacity = standard_opacity * photon_speed * cross_section_ratio
    if np.any(~np.isfinite(effective_opacity)):
        raise ValueError("The modified Thomson opacity must be finite.")
    return (
        float(effective_opacity)
        if effective_opacity.ndim == 0
        else effective_opacity
    )


def neutrino_group_velocity_local(
    momentum_GeV, mass_GeV, N_GeV, parameters
):
    """Return dE_nu/dp through first order in gN and hN."""
    try:
        momentum, mass, field = np.broadcast_arrays(
            np.asarray(momentum_GeV, dtype=float),
            np.asarray(mass_GeV, dtype=float),
            np.asarray(N_GeV, dtype=float),
        )
    except ValueError as error:
        raise ValueError(
            "Momentum, mass, and scalar-field values must be broadcast-compatible."
        ) from error
    if (
        np.any(~np.isfinite(momentum))
        or np.any(momentum < 0.0)
        or np.any(~np.isfinite(mass))
        or np.any(mass < 0.0)
        or np.any(~np.isfinite(field))
    ):
        raise ValueError("Momentum, mass, and scalar-field values must be physical.")

    energy = np.hypot(momentum, mass)
    momentum_fraction = np.divide(
        momentum, energy, out=np.zeros_like(energy), where=energy > 0.0
    )
    mass_fraction = np.divide(
        mass, energy, out=np.zeros_like(energy), where=energy > 0.0
    )
    velocity = momentum_fraction * (
        1.0
        - parameters.g_GeV_inv * field * mass_fraction**2
        - parameters.h_GeV_inv * field
    )
    if np.any(~np.isfinite(velocity)):
        raise ValueError("The neutrino group velocity must be finite.")
    return float(velocity) if velocity.ndim == 0 else velocity


def photon_hamiltonian_perturbation(
    comoving_momentum_GeV,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    frame_velocity_projection,
    parameters,
):
    """Return delta H_gamma in Newtonian gauge to first order in perturbations."""
    momentum = np.asarray(comoving_momentum_GeV, dtype=float)
    values = (N_background_GeV, delta_N_GeV, psi, phi, frame_velocity_projection)
    if (
        np.any(~np.isfinite(momentum))
        or np.any(momentum < 0.0)
        or not all(np.all(np.isfinite(value)) for value in values)
    ):
        raise ValueError("Photon Hamiltonian perturbations must be finite and physical.")
    speed = photon_speed_ratio(N_background_GeV, parameters)
    speed_derivative = speed * photon_speed_log_derivative(
        N_background_GeV, parameters
    )
    return momentum * (
        speed * (psi + phi)
        + speed_derivative * delta_N_GeV
        + (1.0 - speed**2) * frame_velocity_projection
    )


def photon_hamiltonian_mode_coefficients(
    comoving_momentum_GeV,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    clock_velocity_projection,
    parameters,
):
    """Return A_gamma and B_gamma for delta_H_gamma = A_gamma + B_gamma*mu.

    clock_velocity_projection is the scalar Fourier velocity component along
    k; the angular factor mu is applied to B_gamma by the hierarchy.
    """
    try:
        momentum, field, delta_field, lapse, curvature, clock_velocity = (
            np.broadcast_arrays(
                np.asarray(comoving_momentum_GeV, dtype=float),
                np.asarray(N_background_GeV, dtype=float),
                np.asarray(delta_N_GeV, dtype=complex),
                np.asarray(psi, dtype=complex),
                np.asarray(phi, dtype=complex),
                np.asarray(clock_velocity_projection, dtype=complex),
            )
        )
    except ValueError as error:
        raise ValueError("Photon mode inputs must have compatible shapes.") from error
    if (
        np.any(~np.isfinite(momentum))
        or np.any(momentum < 0.0)
        or np.any(~np.isfinite(field))
        or np.any(~np.isfinite(delta_field))
        or np.any(~np.isfinite(lapse))
        or np.any(~np.isfinite(curvature))
        or np.any(~np.isfinite(clock_velocity))
    ):
        raise ValueError("Photon mode inputs must be finite and physical.")

    speed = photon_speed_ratio(field, parameters)
    speed_derivative = speed * photon_speed_log_derivative(field, parameters)
    monopole = momentum * (
        speed * (lapse + curvature) + speed_derivative * delta_field
    )
    dipole = momentum * (1.0 - speed**2) * clock_velocity
    return monopole, dipole


def neutrino_hamiltonian_perturbation(
    comoving_momentum_GeV,
    mass_GeV,
    scale_factor,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    frame_velocity_projection,
    parameters,
):
    """Return delta H_nu in Newtonian gauge at first order in gN and hN."""
    values = (
        comoving_momentum_GeV,
        mass_GeV,
        scale_factor,
        N_background_GeV,
        delta_N_GeV,
        psi,
        phi,
        frame_velocity_projection,
    )
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("Neutrino Hamiltonian perturbations must be finite.")
    if comoving_momentum_GeV < 0.0 or mass_GeV < 0.0 or scale_factor <= 0.0:
        raise ValueError("Momentum and mass must be nonnegative and a must be positive.")

    physical_momentum = comoving_momentum_GeV / scale_factor
    energy = neutrino_energy_local(
        physical_momentum, mass_GeV, N_background_GeV, parameters
    )
    group_velocity = neutrino_group_velocity_local(
        physical_momentum, mass_GeV, N_background_GeV, parameters
    )
    scalar_response = neutrino_scalar_source_per_particle(
        physical_momentum, mass_GeV, parameters
    )
    frame_response = (
        2.0
        * parameters.h_GeV_inv
        * N_background_GeV
        * physical_momentum
        * frame_velocity_projection
    )
    return scale_factor * (
        psi * energy
        + phi * physical_momentum * group_velocity
        + delta_N_GeV * scalar_response
        + frame_response
    )


def neutrino_hamiltonian_mode_coefficients(
    comoving_momenta_GeV,
    mass_GeV,
    scale_factor,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    clock_velocity_projection,
    parameters,
):
    """Return A_nu and B_nu for delta_H_nu = A_nu + B_nu*mu."""
    try:
        momentum, mass, scale, field, delta_field, lapse, curvature, clock_velocity = (
            np.broadcast_arrays(
                np.asarray(comoving_momenta_GeV, dtype=float),
                np.asarray(mass_GeV, dtype=float),
                np.asarray(scale_factor, dtype=float),
                np.asarray(N_background_GeV, dtype=float),
                np.asarray(delta_N_GeV, dtype=complex),
                np.asarray(psi, dtype=complex),
                np.asarray(phi, dtype=complex),
                np.asarray(clock_velocity_projection, dtype=complex),
            )
        )
    except ValueError as error:
        raise ValueError("Neutrino mode inputs must have compatible shapes.") from error
    if (
        np.any(~np.isfinite(momentum))
        or np.any(momentum < 0.0)
        or np.any(~np.isfinite(mass))
        or np.any(mass < 0.0)
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0.0)
        or np.any(~np.isfinite(field))
        or np.any(~np.isfinite(delta_field))
        or np.any(~np.isfinite(lapse))
        or np.any(~np.isfinite(curvature))
        or np.any(~np.isfinite(clock_velocity))
    ):
        raise ValueError("Neutrino mode inputs must be finite and physical.")

    physical_momentum = momentum / scale
    uncoupled_energy = np.hypot(physical_momentum, mass)
    scalar_response = np.divide(
        parameters.g_GeV_inv * mass**2,
        uncoupled_energy,
        out=np.zeros_like(uncoupled_energy),
        where=uncoupled_energy > 0.0,
    ) - parameters.h_GeV_inv * uncoupled_energy
    energy = uncoupled_energy + field * scalar_response
    momentum_fraction = np.divide(
        physical_momentum,
        uncoupled_energy,
        out=np.zeros_like(uncoupled_energy),
        where=uncoupled_energy > 0.0,
    )
    mass_fraction = np.divide(
        mass,
        uncoupled_energy,
        out=np.zeros_like(uncoupled_energy),
        where=uncoupled_energy > 0.0,
    )
    group_velocity = momentum_fraction * (
        1.0
        - parameters.g_GeV_inv * field * mass_fraction**2
        - parameters.h_GeV_inv * field
    )
    if (
        np.any(~np.isfinite(energy))
        or np.any(energy < 0.0)
        or np.any(~np.isfinite(group_velocity))
        or np.any(group_velocity < 0.0)
    ):
        raise ValueError("The first-order neutrino dispersion is not physical.")

    monopole = scale * (
        lapse * energy
        + curvature * physical_momentum * group_velocity
        + delta_field * scalar_response
    )
    dipole = (
        scale
        * 2.0
        * parameters.h_GeV_inv
        * field
        * physical_momentum
        * clock_velocity
    )
    return monopole, dipole


def neutrino_scalar_mode_rhs(
    distribution_multipoles,
    wavenumber,
    comoving_momenta_GeV,
    mass_GeV,
    scale_factor,
    distribution_radial_gradient,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    clock_velocity_projection,
    next_multipole,
    parameters,
):
    """Return the collisionless massive-neutrino scalar hierarchy RHS.

    Multipoles and radial gradients use comoving momentum q; the local
    dispersion and group velocity use p=q/a. This does not assemble stress
    moments or close the Einstein constraints.
    """
    hamiltonian_monopole, hamiltonian_dipole = (
        neutrino_hamiltonian_mode_coefficients(
            comoving_momenta_GeV,
            mass_GeV,
            scale_factor,
            N_background_GeV,
            delta_N_GeV,
            psi,
            phi,
            clock_velocity_projection,
            parameters,
        )
    )
    group_velocity = neutrino_group_velocity_local(
        np.asarray(comoving_momenta_GeV, dtype=float) / scale_factor,
        mass_GeV,
        N_background_GeV,
        parameters,
    )
    return scalar_mode_multipole_rhs(
        distribution_multipoles,
        wavenumber,
        group_velocity,
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_multipole,
    )


def collisionless_fourier_transport_rhs(
    delta_distribution,
    wavenumber,
    direction_cosine,
    group_velocity,
    delta_hamiltonian,
    distribution_radial_gradient,
    collision_term=0.0,
):
    """Evaluate the linear Fourier-space Liouville RHS for isotropic f0."""
    wave_number = np.asarray(wavenumber, dtype=float)
    cosine = np.asarray(direction_cosine, dtype=float)
    velocity = np.asarray(group_velocity, dtype=float)
    distribution = np.asarray(delta_distribution)
    hamiltonian = np.asarray(delta_hamiltonian)
    radial_gradient = np.asarray(distribution_radial_gradient)
    collisions = np.asarray(collision_term)
    if (
        np.any(~np.isfinite(wave_number))
        or np.any(wave_number < 0.0)
        or np.any(~np.isfinite(cosine))
        or np.any(np.abs(cosine) > 1.0)
        or np.any(~np.isfinite(velocity))
        or np.any(velocity < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(~np.isfinite(hamiltonian))
        or np.any(~np.isfinite(radial_gradient))
        or np.any(~np.isfinite(collisions))
    ):
        raise ValueError("Fourier transport inputs must be finite and physical.")
    streaming = -1j * wave_number * cosine * velocity * distribution
    force = 1j * wave_number * cosine * hamiltonian * radial_gradient
    return streaming + force + collisions


def collisionless_multipole_streaming_rhs(
    multipoles,
    wavenumber,
    group_velocity,
    source_multipoles,
    next_multipole,
):
    """Stream plain Legendre or scalar E-mode moments with explicit closure.

    The convention is delta_f(mu) = sum_ell (2 ell + 1) F_ell P_ell(mu).
    Scalar E-mode moments use Q_E(mu) = (1 - mu**2) times the same series.
    """
    multipoles = np.asarray(multipoles, dtype=complex)
    sources = np.asarray(source_multipoles, dtype=complex)
    if (
        multipoles.ndim < 1
        or multipoles.shape[-1] == 0
        or sources.shape != multipoles.shape
        or np.any(~np.isfinite(multipoles))
        or np.any(~np.isfinite(sources))
    ):
        raise ValueError("Multipoles and streaming parameters must be physical.")
    try:
        leading_shape = multipoles.shape[:-1]
        wave_number = np.broadcast_to(
            np.asarray(wavenumber, dtype=float), leading_shape
        )
        velocity = np.broadcast_to(
            np.asarray(group_velocity, dtype=float), leading_shape
        )
    except ValueError as error:
        raise ValueError(
            "Wavenumber and velocity must match the leading moment shape."
        ) from error
    if (
        not np.all(np.isfinite(wave_number))
        or not np.all(np.isfinite(velocity))
        or np.any(wave_number < 0.0)
        or np.any(velocity < 0.0)
    ):
        raise ValueError("Wavenumber and velocity must be finite and nonnegative.")
    try:
        upper_multipole = np.broadcast_to(
            np.asarray(next_multipole, dtype=complex), multipoles.shape[:-1]
        )
    except ValueError as error:
        raise ValueError("next_multipole must match the leading moment shape.") from error
    if np.any(~np.isfinite(upper_multipole)):
        raise ValueError("next_multipole must be finite.")

    streaming_moments = np.zeros_like(multipoles)
    maximum_ell = multipoles.shape[-1] - 1
    for ell in range(maximum_ell + 1):
        lower = ell * multipoles[..., ell - 1] if ell > 0 else 0.0
        upper = (
            (ell + 1) * multipoles[..., ell + 1]
            if ell < maximum_ell
            else (ell + 1) * upper_multipole
        )
        streaming_moments[..., ell] = (lower + upper) / (2 * ell + 1)
    streaming_rate = (wave_number * velocity)[..., np.newaxis]
    return -1j * streaming_rate * streaming_moments + sources


def free_streaming_bessel_closure(
    multipoles,
    wavenumber_GeV,
    free_streaming_distance_GeVinv,
):
    """Estimate F_(L+1) with the late-time free-streaming Bessel recurrence.

    For x = k * chi, the plane-wave convention used here gives
    F_(L+1) = F_(L-1) - i * (2L + 1) * F_L / x. The closure is smoothly
    activated from x=L to x=2L to avoid its small-x singularity. chi is the
    caller-integrated group-velocity distance along the particle trajectory.
    """
    multipoles = np.asarray(multipoles, dtype=complex)
    if (
        multipoles.ndim < 1
        or multipoles.shape[-1] < 2
        or np.any(~np.isfinite(multipoles))
    ):
        raise ValueError("At least two finite multipoles are required.")
    try:
        leading_shape = multipoles.shape[:-1]
        wave_number = np.broadcast_to(
            np.asarray(wavenumber_GeV, dtype=float), leading_shape
        )
        distance = np.broadcast_to(
            np.asarray(free_streaming_distance_GeVinv, dtype=float), leading_shape
        )
    except ValueError as error:
        raise ValueError(
            "Wavenumber and free-streaming distance must match the leading shape."
        ) from error
    if (
        np.any(~np.isfinite(wave_number))
        or np.any(wave_number < 0.0)
        or np.any(~np.isfinite(distance))
        or np.any(distance < 0.0)
    ):
        raise ValueError("Closure parameters must be finite and nonnegative.")

    maximum_ell = multipoles.shape[-1] - 1
    phase = wave_number * distance
    if np.any(~np.isfinite(phase)):
        raise ValueError("Free-streaming phase must be finite.")
    safe_phase = np.where(phase > 0.0, phase, 1.0)
    recurrence = (
        multipoles[..., -2]
        - 1j * (2 * maximum_ell + 1) * multipoles[..., -1] / safe_phase
    )
    transition = np.clip((phase - maximum_ell) / maximum_ell, 0.0, 1.0)
    activation = transition**2 * (3.0 - 2.0 * transition)
    return activation * recurrence


def free_streaming_multipole_streaming_rhs(
    multipoles,
    wavenumber,
    group_velocity,
    source_multipoles,
    free_streaming_distance_GeVinv,
):
    """Evaluate the streaming hierarchy using the Bessel upper-moment closure."""
    next_multipole = free_streaming_bessel_closure(
        multipoles,
        wavenumber,
        free_streaming_distance_GeVinv,
    )
    return collisionless_multipole_streaming_rhs(
        multipoles,
        wavenumber,
        group_velocity,
        source_multipoles,
        next_multipole,
    )


def free_streaming_distance_history(
    log_scale_factor,
    conformal_hubble_GeV,
    group_velocity,
):
    """Accumulate chi(x) = integral v/(a H) dx on a sampled background.

    The first axis of ``group_velocity`` is the background history; trailing
    axes, such as a comoving-momentum grid, are preserved in the result.
    The trapezoidal integral returns distances in GeV^-1.
    """
    log_scale_factor = np.asarray(log_scale_factor, dtype=float)
    conformal_hubble = np.asarray(conformal_hubble_GeV, dtype=float)
    group_velocity = np.asarray(group_velocity, dtype=float)
    if (
        log_scale_factor.ndim != 1
        or log_scale_factor.size < 2
        or np.any(~np.isfinite(log_scale_factor))
        or np.any(np.diff(log_scale_factor) <= 0.0)
    ):
        raise ValueError("log_scale_factor must be a finite, increasing history.")
    if (
        conformal_hubble.shape != log_scale_factor.shape
        or np.any(~np.isfinite(conformal_hubble))
        or np.any(conformal_hubble <= 0.0)
    ):
        raise ValueError("Conformal Hubble values must be finite, positive, and match the history.")
    if (
        group_velocity.ndim < 1
        or group_velocity.shape[0] != log_scale_factor.size
        or np.any(~np.isfinite(group_velocity))
        or np.any(group_velocity < 0.0)
    ):
        raise ValueError("Group velocities must be finite, nonnegative, and time-first.")

    trailing_dimensions = (1,) * (group_velocity.ndim - 1)
    conformal_hubble = conformal_hubble.reshape(
        (log_scale_factor.size,) + trailing_dimensions
    )
    integrand = group_velocity / conformal_hubble
    step_shape = (log_scale_factor.size - 1,) + trailing_dimensions
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(
        log_scale_factor
    ).reshape(step_shape)
    distance = np.zeros_like(group_velocity, dtype=float)
    distance[1:] = np.cumsum(increments, axis=0)
    return distance


def integrate_free_streaming_multipoles(
    log_scale_factor,
    initial_multipoles,
    wavenumber_GeV,
    conformal_hubble_GeV,
    group_velocity_history,
    source_multipole_history,
    *,
    method="DOP853",
    rtol=1e-9,
    atol=1e-12,
    max_step=np.inf,
):
    """Integrate a Bessel-closed hierarchy on a caller-supplied background.

    Background velocities, conformal Hubble values, and source multipoles are
    linearly interpolated from the supplied log-scale-factor grid. Source
    multipoles are per conformal time; the returned moments are time-first.
    This does not evolve the background or compute the source multipoles.
    """
    log_scale_factor = np.asarray(log_scale_factor, dtype=float)
    initial_multipoles = np.asarray(initial_multipoles, dtype=complex)
    if (
        log_scale_factor.ndim != 1
        or log_scale_factor.size < 2
        or np.any(~np.isfinite(log_scale_factor))
        or np.any(np.diff(log_scale_factor) <= 0.0)
    ):
        raise ValueError("log_scale_factor must be a finite, increasing history.")
    if (
        initial_multipoles.ndim < 1
        or initial_multipoles.shape[-1] < 2
        or np.any(~np.isfinite(initial_multipoles))
    ):
        raise ValueError("At least two finite initial multipoles are required.")

    time_count = log_scale_factor.size
    leading_shape = initial_multipoles.shape[:-1]
    conformal_hubble = np.asarray(conformal_hubble_GeV, dtype=float)
    if (
        conformal_hubble.shape != log_scale_factor.shape
        or np.any(~np.isfinite(conformal_hubble))
        or np.any(conformal_hubble <= 0.0)
    ):
        raise ValueError("Conformal Hubble values must be finite, positive, and match the history.")

    velocities = np.asarray(group_velocity_history, dtype=float)
    if velocities.shape == log_scale_factor.shape and leading_shape:
        velocities = velocities.reshape(
            (time_count,) + (1,) * len(leading_shape)
        )
    try:
        velocities = np.broadcast_to(
            velocities, (time_count,) + leading_shape
        ).copy()
        wave_number = np.broadcast_to(
            np.asarray(wavenumber_GeV, dtype=float), leading_shape
        )
    except ValueError as error:
        raise ValueError(
            "Velocities and wavenumber must match the leading multipole shape."
        ) from error
    if (
        np.any(~np.isfinite(velocities))
        or np.any(velocities < 0.0)
        or np.any(~np.isfinite(wave_number))
        or np.any(wave_number < 0.0)
    ):
        raise ValueError("Velocities and wavenumber must be finite and nonnegative.")

    sources = np.asarray(source_multipole_history, dtype=complex)
    expected_source_shape = (time_count,) + initial_multipoles.shape
    if sources.shape == initial_multipoles.shape:
        sources = np.broadcast_to(sources, expected_source_shape).copy()
    if sources.shape != expected_source_shape or np.any(~np.isfinite(sources)):
        raise ValueError("Source history must be finite and match the time/multipole shape.")

    if (
        not np.isfinite(rtol)
        or rtol <= 0.0
        or not np.isfinite(atol)
        or atol <= 0.0
        or np.isnan(max_step)
        or max_step <= 0.0
    ):
        raise ValueError("Solver tolerances and max_step must be positive.")

    distance_history = free_streaming_distance_history(
        log_scale_factor,
        conformal_hubble,
        velocities,
    )
    interpolate_options = {
        "kind": "linear",
        "axis": 0,
        "bounds_error": True,
        "assume_sorted": True,
    }
    hubble_at = interp1d(log_scale_factor, conformal_hubble, **interpolate_options)
    velocity_at = interp1d(log_scale_factor, velocities, **interpolate_options)
    source_at = interp1d(log_scale_factor, sources, **interpolate_options)
    distance_at = interp1d(
        log_scale_factor, distance_history, **interpolate_options
    )

    state_size = initial_multipoles.size
    initial_state = np.concatenate(
        (initial_multipoles.real.ravel(), initial_multipoles.imag.ravel())
    )

    def hierarchy_rhs(log_a, packed_state):
        moments = (
            packed_state[:state_size]
            + 1j * packed_state[state_size:]
        ).reshape(initial_multipoles.shape)
        derivative_tau = free_streaming_multipole_streaming_rhs(
            moments,
            wave_number,
            velocity_at(log_a),
            source_at(log_a),
            distance_at(log_a),
        )
        derivative_log_a = derivative_tau / float(hubble_at(log_a))
        return np.concatenate(
            (derivative_log_a.real.ravel(), derivative_log_a.imag.ravel())
        )

    solution = solve_ivp(
        hierarchy_rhs,
        (log_scale_factor[0], log_scale_factor[-1]),
        initial_state,
        method=method,
        t_eval=log_scale_factor,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    integrated_state = (
        solution.y[:state_size] + 1j * solution.y[state_size:]
    )
    multipole_history = integrated_state.T.reshape(
        (solution.t.size,) + initial_multipoles.shape
    )
    return {
        "solution": solution,
        "multipoles": multipole_history,
        "free_streaming_distance_GeVinv": distance_at(solution.t),
    }


def collisionless_hamiltonian_force_multipoles(
    wavenumber,
    distribution_radial_gradient,
    isotropic_hamiltonian_perturbation,
    dipole_hamiltonian_perturbation,
    maximum_ell,
):
    """Project the scalar-mode force source for delta_H = A + B mu."""
    if (
        isinstance(maximum_ell, (bool, np.bool_))
        or not isinstance(maximum_ell, (int, np.integer))
        or maximum_ell < 0
    ):
        raise ValueError("maximum_ell must be a nonnegative integer.")
    try:
        wave_number, radial_gradient, monopole, dipole = np.broadcast_arrays(
            np.asarray(wavenumber, dtype=float),
            np.asarray(distribution_radial_gradient, dtype=float),
            np.asarray(isotropic_hamiltonian_perturbation, dtype=complex),
            np.asarray(dipole_hamiltonian_perturbation, dtype=complex),
        )
    except ValueError as error:
        raise ValueError("Force-source inputs must have compatible shapes.") from error
    if (
        np.any(~np.isfinite(wave_number))
        or np.any(wave_number < 0.0)
        or np.any(~np.isfinite(radial_gradient))
        or np.any(~np.isfinite(monopole))
        or np.any(~np.isfinite(dipole))
    ):
        raise ValueError("Force-source inputs must be finite and physical.")

    source_multipoles = np.zeros(
        wave_number.shape + (maximum_ell + 1,), dtype=complex
    )
    prefactor = 1j * wave_number * radial_gradient
    source_multipoles[..., 0] = prefactor * dipole / 3.0
    if maximum_ell >= 1:
        source_multipoles[..., 1] = prefactor * monopole / 3.0
    if maximum_ell >= 2:
        source_multipoles[..., 2] = 2.0 * prefactor * dipole / 15.0
    return source_multipoles


def scalar_mode_multipole_rhs(
    multipoles,
    wavenumber,
    group_velocity,
    distribution_radial_gradient,
    hamiltonian_monopole,
    hamiltonian_dipole,
    next_multipole,
    collision_multipoles=0.0,
):
    """Combine streaming, Hamiltonian forcing, and supplied collision moments."""
    multipoles = np.asarray(multipoles, dtype=complex)
    if multipoles.ndim < 1 or multipoles.shape[-1] == 0:
        raise ValueError("At least one Legendre multipole is required.")
    maximum_ell = multipoles.shape[-1] - 1
    force_multipoles = collisionless_hamiltonian_force_multipoles(
        wavenumber,
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        maximum_ell,
    )
    try:
        collisions = np.broadcast_to(
            np.asarray(collision_multipoles, dtype=complex), multipoles.shape
        )
    except ValueError as error:
        raise ValueError("Collision moments must match the multipole shape.") from error
    return collisionless_multipole_streaming_rhs(
        multipoles,
        wavenumber,
        group_velocity,
        force_multipoles + collisions,
        next_multipole,
    )


def photon_scalar_intensity_multipole_rhs(
    intensity_multipoles,
    wavenumber,
    comoving_momenta_GeV,
    distribution_radial_gradient,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    clock_velocity_projection,
    next_multipole,
    parameters,
    collision_multipoles=0.0,
):
    """Return finite-h photon intensity RHS with caller-supplied collisions.

    The Hamiltonian and streaming speed use the modified Maxwell action.
    This helper does not calculate photon-electron scattering or provide a
    finite-h Thomson collision kernel.
    """
    hamiltonian_monopole, hamiltonian_dipole = (
        photon_hamiltonian_mode_coefficients(
            comoving_momenta_GeV,
            N_background_GeV,
            delta_N_GeV,
            psi,
            phi,
            clock_velocity_projection,
            parameters,
        )
    )
    return scalar_mode_multipole_rhs(
        intensity_multipoles,
        wavenumber,
        photon_group_velocity_local(N_background_GeV, parameters),
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_multipole,
        collision_multipoles,
    )


def photon_scalar_mode_rhs(
    intensity_multipoles,
    polarization_multipoles,
    wavenumber,
    comoving_momenta_GeV,
    distribution_radial_gradient,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    clock_velocity_projection,
    next_intensity_multipole,
    next_polarization_multipole,
    parameters,
    intensity_collision_multipoles=0.0,
    polarization_collision_multipoles=0.0,
):
    """Return intensity and E-mode RHSs with finite-h photon streaming.

    Both sectors use the modified photon speed. Hamiltonian forcing acts on
    intensity; collision terms for both sectors are supplied by the caller.
    No finite-h scattering or polarization collision kernel is calculated.
    """
    hamiltonian_monopole, hamiltonian_dipole = (
        photon_hamiltonian_mode_coefficients(
            comoving_momenta_GeV,
            N_background_GeV,
            delta_N_GeV,
            psi,
            phi,
            clock_velocity_projection,
            parameters,
        )
    )
    return _standard_thomson_photon_rhs_from_collisions(
        intensity_multipoles,
        polarization_multipoles,
        wavenumber,
        photon_group_velocity_local(N_background_GeV, parameters),
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_intensity_multipole,
        next_polarization_multipole,
        intensity_collision_multipoles,
        polarization_collision_multipoles,
    )


def standard_thomson_scalar_collision_multipoles(
    intensity_multipoles,
    polarization_multipoles,
    comoving_momenta_GeV,
    homogeneous_distribution_derivative,
    opacity_per_conformal_time,
    baryon_velocity_projection,
):
    """Return scalar Thomson collision terms in the standard h=0 limit.

    Intensity uses plain Legendre moments; Q_E uses (1 - mu**2) times its
    Legendre series with the same spectral normalization.
    The baryon velocity is the coefficient V_b in v_b dot qhat = mu V_b.
    The polarized quadrupole source is Pi = F_2 + P_0 + P_2.
    """
    intensity = np.asarray(intensity_multipoles, dtype=complex)
    polarization = np.asarray(polarization_multipoles, dtype=complex)
    if (
        intensity.ndim < 1
        or intensity.shape[-1] < 3
        or polarization.shape != intensity.shape
        or np.any(~np.isfinite(intensity))
        or np.any(~np.isfinite(polarization))
    ):
        raise ValueError(
            "Intensity and polarization need matching multipoles through ell=2."
        )
    try:
        leading_shape = intensity.shape[:-1]
        momentum = np.broadcast_to(
            np.asarray(comoving_momenta_GeV, dtype=float), leading_shape
        )
        distribution_derivative = np.broadcast_to(
            np.asarray(homogeneous_distribution_derivative, dtype=float),
            leading_shape,
        )
        opacity = np.broadcast_to(
            np.asarray(opacity_per_conformal_time, dtype=float), leading_shape
        )
        baryon_velocity = np.broadcast_to(
            np.asarray(baryon_velocity_projection, dtype=complex), leading_shape
        )
    except ValueError as error:
        raise ValueError(
            "Thomson parameters must match the leading momentum shape."
        ) from error
    if (
        np.any(~np.isfinite(momentum))
        or np.any(momentum < 0.0)
        or np.any(~np.isfinite(distribution_derivative))
        or np.any(~np.isfinite(opacity))
        or np.any(opacity < 0.0)
        or np.any(~np.isfinite(baryon_velocity))
    ):
        raise ValueError("Thomson inputs must be finite and physical.")

    rate = opacity[..., np.newaxis]
    intensity_collisions = -rate * intensity
    intensity_collisions[..., 0] = 0.0
    intensity_collisions[..., 1] = -opacity * (
        intensity[..., 1]
        + momentum * distribution_derivative * baryon_velocity / 3.0
    )
    quadrupole_source = (
        intensity[..., 2]
        + polarization[..., 0]
        + polarization[..., 2]
    )
    intensity_collisions[..., 2] = -opacity * (
        intensity[..., 2] - quadrupole_source / 10.0
    )

    polarization_collisions = -rate * polarization
    polarization_collisions[..., 0] = -opacity * (
        polarization[..., 0] - quadrupole_source / 2.0
    )
    polarization_collisions[..., 2] = -opacity * (
        polarization[..., 2] - quadrupole_source / 10.0
    )
    return intensity_collisions, polarization_collisions


def _standard_thomson_photon_rhs_from_collisions(
    intensity_multipoles,
    polarization_multipoles,
    wavenumber,
    group_velocity,
    distribution_radial_gradient,
    hamiltonian_monopole,
    hamiltonian_dipole,
    next_intensity_multipole,
    next_polarization_multipole,
    intensity_collisions,
    polarization_collisions,
):
    intensity_rhs = scalar_mode_multipole_rhs(
        intensity_multipoles,
        wavenumber,
        group_velocity,
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_intensity_multipole,
        intensity_collisions,
    )
    polarization_rhs = collisionless_multipole_streaming_rhs(
        polarization_multipoles,
        wavenumber,
        group_velocity,
        polarization_collisions,
        next_polarization_multipole,
    )
    return intensity_rhs, polarization_rhs


def standard_thomson_scalar_photon_mode_rhs(
    intensity_multipoles,
    polarization_multipoles,
    wavenumber,
    group_velocity,
    distribution_radial_gradient,
    hamiltonian_monopole,
    hamiltonian_dipole,
    next_intensity_multipole,
    next_polarization_multipole,
    comoving_momenta_GeV,
    homogeneous_distribution_derivative,
    opacity_per_conformal_time,
    baryon_velocity_projection,
):
    """Combine photon transport with the conventional h=0 Thomson operator.

    Hamiltonian perturbations and group velocity are supplied by the caller;
    this helper does not model coupling-dependent scattering corrections.
    """
    intensity_collisions, polarization_collisions = (
        standard_thomson_scalar_collision_multipoles(
            intensity_multipoles,
            polarization_multipoles,
            comoving_momenta_GeV,
            homogeneous_distribution_derivative,
            opacity_per_conformal_time,
            baryon_velocity_projection,
        )
    )
    return _standard_thomson_photon_rhs_from_collisions(
        intensity_multipoles,
        polarization_multipoles,
        wavenumber,
        group_velocity,
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_intensity_multipole,
        next_polarization_multipole,
        intensity_collisions,
        polarization_collisions,
    )


def standard_thomson_photon_baryon_mode_rhs(
    intensity_multipoles,
    polarization_multipoles,
    wavenumber,
    group_velocity,
    distribution_radial_gradient,
    hamiltonian_monopole,
    hamiltonian_dipole,
    next_intensity_multipole,
    next_polarization_multipole,
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    homogeneous_distribution_derivative,
    opacity_per_conformal_time,
    scale_factor,
    baryon_density_GeV4,
    baryon_density_contrast,
    baryon_velocity_projection,
    conformal_hubble,
    psi,
    phi_prime,
    photon_degeneracy=2.0,
    sound_speed_squared=0.0,
    collision_velocity_projection=None,
):
    """Return standard Thomson-coupled photon and baryon scalar-mode RHSs.

    The collision momentum transfer is shared with the baryon Euler equation
    with the opposite sign. An optional collision-frame velocity changes only
    the linear dipole source; the physical baryon velocity remains unchanged.
    """
    if collision_velocity_projection is None:
        collision_velocity_projection = baryon_velocity_projection
    intensity_collisions, polarization_collisions = (
        standard_thomson_scalar_collision_multipoles(
            intensity_multipoles,
            polarization_multipoles,
            comoving_momenta_GeV,
            homogeneous_distribution_derivative,
            opacity_per_conformal_time,
            collision_velocity_projection,
        )
    )
    intensity_rhs, polarization_rhs = _standard_thomson_photon_rhs_from_collisions(
        intensity_multipoles,
        polarization_multipoles,
        wavenumber,
        group_velocity,
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_intensity_multipole,
        next_polarization_multipole,
        intensity_collisions,
        polarization_collisions,
    )
    momentum_exchange = standard_thomson_baryon_velocity_collision_source(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        intensity_collisions,
        scale_factor,
        baryon_density_GeV4,
        photon_degeneracy,
    )
    baryon_density_rhs, baryon_velocity_rhs = standard_baryon_scalar_rhs(
        baryon_density_contrast,
        baryon_velocity_projection,
        wavenumber,
        conformal_hubble,
        psi,
        phi_prime,
        momentum_exchange["baryon_velocity_source_GeV"],
        sound_speed_squared,
    )
    return {
        "intensity_multipole_rhs": intensity_rhs,
        "polarization_multipole_rhs": polarization_rhs,
        "baryon_density_contrast_rhs": baryon_density_rhs,
        "baryon_velocity_rhs": baryon_velocity_rhs,
        "photon_momentum_transfer_GeV5": momentum_exchange[
            "photon_momentum_transfer_GeV5"
        ],
        "baryon_velocity_collision_source_GeV": momentum_exchange[
            "baryon_velocity_source_GeV"
        ],
    }


def standard_thomson_finite_h_photon_baryon_mode_rhs(
    intensity_multipoles,
    polarization_multipoles,
    wavenumber,
    distribution_radial_gradient,
    N_background_GeV,
    delta_N_GeV,
    psi,
    phi,
    clock_velocity_projection,
    next_intensity_multipole,
    next_polarization_multipole,
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    homogeneous_distribution_derivative,
    opacity_per_conformal_time,
    scale_factor,
    baryon_density_GeV4,
    baryon_density_contrast,
    baryon_velocity_projection,
    conformal_hubble,
    phi_prime,
    parameters,
    photon_degeneracy=2.0,
    sound_speed_squared=0.0,
    N_background_prime_GeV2=None,
    clock_velocity_prime_GeV=0.0,
):
    """Compose finite-h propagation and Thomson collisions with baryon motion.

    The local-clock velocity transformation supplies the finite-h dipole
    source. Higher intensity multipoles and E-polarization use stationary
    Thomson kernels; recoil and second-order velocity terms are omitted.
    Supplying N_background_prime_GeV2 selects the action-coupled pressureless
    dust Euler equation, with baryon_density_GeV4 interpreted as rest density.
    That branch requires zero sound speed; omitting it retains the standard
    baryon Euler equation.
    """
    hamiltonian_monopole, hamiltonian_dipole = (
        photon_hamiltonian_mode_coefficients(
            comoving_momenta_GeV,
            N_background_GeV,
            delta_N_GeV,
            psi,
            phi,
            clock_velocity_projection,
            parameters,
        )
    )
    finite_h_opacity = finite_h_thomson_opacity_per_conformal_time(
        opacity_per_conformal_time, N_background_GeV, parameters
    )
    collision_velocity = finite_h_thomson_collision_velocity_projection(
        baryon_velocity_projection,
        clock_velocity_projection,
        N_background_GeV,
        parameters,
    )
    result = standard_thomson_photon_baryon_mode_rhs(
        intensity_multipoles,
        polarization_multipoles,
        wavenumber,
        photon_group_velocity_local(N_background_GeV, parameters),
        distribution_radial_gradient,
        hamiltonian_monopole,
        hamiltonian_dipole,
        next_intensity_multipole,
        next_polarization_multipole,
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        homogeneous_distribution_derivative,
        finite_h_opacity,
        scale_factor,
        baryon_density_GeV4,
        baryon_density_contrast,
        baryon_velocity_projection,
        conformal_hubble,
        psi,
        phi_prime,
        photon_degeneracy=photon_degeneracy,
        sound_speed_squared=sound_speed_squared,
        collision_velocity_projection=collision_velocity,
    )
    if N_background_prime_GeV2 is None:
        return result
    if sound_speed_squared != 0.0:
        raise ValueError("Coupled pressureless baryons require zero sound speed.")

    rest_mass_factor = 1.0 + parameters.eta_GeV_inv * N_background_GeV
    inertial_mass_factor = 1.0 + (
        parameters.g_GeV_inv + parameters.h_GeV_inv
    ) * N_background_GeV
    if (
        not np.isfinite(rest_mass_factor)
        or rest_mass_factor <= 0.0
        or not np.isfinite(inertial_mass_factor)
        or inertial_mass_factor <= 0.0
    ):
        raise ValueError("Coupled baryon rest energy and inertia must be positive.")
    collision_acceleration = (
        rest_mass_factor
        / inertial_mass_factor
        * result["baryon_velocity_collision_source_GeV"]
    )
    baryon_density_rhs, baryon_velocity_rhs = (
        coupled_pressureless_matter_scalar_rhs(
            baryon_density_contrast,
            baryon_velocity_projection,
            wavenumber,
            conformal_hubble,
            psi,
            phi_prime,
            N_background_GeV,
            N_background_prime_GeV2,
            delta_N_GeV,
            parameters,
            clock_velocity_projection=clock_velocity_projection,
            clock_velocity_prime_GeV=clock_velocity_prime_GeV,
            collision_velocity_source=collision_acceleration,
        )
    )
    result["baryon_density_contrast_rhs"] = baryon_density_rhs
    result["baryon_velocity_rhs"] = baryon_velocity_rhs
    result["baryon_velocity_collision_source_GeV"] = collision_acceleration
    return result


def standard_thomson_baryon_velocity_collision_source(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    intensity_collision_multipoles,
    scale_factor,
    baryon_density_GeV4,
    photon_degeneracy=2.0,
):
    """Return the equal-and-opposite baryon velocity source from Thomson drag."""
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    collisions = np.asarray(intensity_collision_multipoles, dtype=complex)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or collisions.ndim != 2
        or collisions.shape[0] != momenta.size
        or collisions.shape[1] < 2
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(collisions))
        or not np.isfinite(scale_factor)
        or scale_factor <= 0.0
        or not np.isfinite(baryon_density_GeV4)
        or baryon_density_GeV4 <= 0.0
        or not np.isfinite(photon_degeneracy)
        or photon_degeneracy <= 0.0
    ):
        raise ValueError("Thomson momentum-transfer inputs must be physical.")

    photon_momentum_transfer = (
        photon_degeneracy
        / (2.0 * np.pi**2 * scale_factor**4)
        * np.sum(weights * momenta**3 * collisions[:, 1])
    )
    return {
        "photon_momentum_transfer_GeV5": photon_momentum_transfer,
        "baryon_velocity_source_GeV": (
            -photon_momentum_transfer / baryon_density_GeV4
        ),
    }


def standard_baryon_scalar_rhs(
    density_contrast,
    velocity_projection,
    wavenumber,
    conformal_hubble,
    psi,
    phi_prime,
    collision_velocity_source=0.0,
    sound_speed_squared=0.0,
):
    """Return standard Newtonian-gauge baryon continuity and Euler RHS terms.

    V_b is the Fourier velocity component along k; coupling-dependent forces
    and non-Thomson interaction terms are not included.
    """
    values = (
        density_contrast,
        velocity_projection,
        wavenumber,
        conformal_hubble,
        psi,
        phi_prime,
        collision_velocity_source,
        sound_speed_squared,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Baryon perturbation inputs must be finite.")
    if wavenumber < 0.0 or sound_speed_squared < 0.0:
        raise ValueError("Wavenumber and baryon sound speed must be nonnegative.")

    density_rhs = -1j * wavenumber * velocity_projection + 3.0 * phi_prime
    velocity_rhs = (
        -conformal_hubble * velocity_projection
        - 1j
        * wavenumber
        * (psi + sound_speed_squared * density_contrast)
        + collision_velocity_source
    )
    return density_rhs, velocity_rhs


def coupled_pressureless_matter_scalar_rhs(
    density_contrast,
    velocity_projection,
    wavenumber_GeV,
    conformal_hubble_GeV,
    psi,
    phi_prime_GeV,
    N_background_GeV,
    N_background_prime_GeV2,
    delta_N_GeV,
    parameters,
    clock_velocity_projection=0.0,
    clock_velocity_prime_GeV=0.0,
    collision_velocity_source=0.0,
):
    """Return continuity and Euler RHSs from the coupled dust worldline action.

    density_contrast follows the conserved bare particle number; use the
    stress-moment helper for the field-dependent rest-energy perturbation.
    The action gives distinct rest-energy and inertial factors, with couplings
    g-h and g+h respectively. Relative clock-frame motion contributes through
    the canonical momentum even when the background velocities are aligned.
    collision_velocity_source is an externally computed Euler acceleration,
    already normalized by the coupled dust inertia.
    """
    background_values = (
        wavenumber_GeV,
        conformal_hubble_GeV,
        N_background_GeV,
        N_background_prime_GeV2,
    )
    if any(
        np.ndim(value) != 0 or not np.isreal(value) or not np.isfinite(value)
        for value in background_values
    ):
        raise ValueError("Dust background inputs must be finite scalars.")
    perturbation_values = (
        density_contrast,
        velocity_projection,
        psi,
        phi_prime_GeV,
        delta_N_GeV,
        clock_velocity_projection,
        clock_velocity_prime_GeV,
        collision_velocity_source,
    )
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in perturbation_values
    ):
        raise ValueError("Dust perturbations must be finite scalars.")
    if wavenumber_GeV < 0.0 or conformal_hubble_GeV < 0.0:
        raise ValueError("Wavenumber and conformal Hubble rate must be nonnegative.")

    rest_mass_factor = (
        1.0 + parameters.eta_GeV_inv * N_background_GeV
    )
    inertial_coupling = parameters.g_GeV_inv + parameters.h_GeV_inv
    inertial_mass_factor = 1.0 + inertial_coupling * N_background_GeV
    if not np.isfinite(rest_mass_factor) or rest_mass_factor <= 0.0:
        raise ValueError("The effective dust rest energy must stay positive.")
    if not np.isfinite(inertial_mass_factor) or inertial_mass_factor <= 0.0:
        raise ValueError("The effective dust inertia must stay positive.")
    mass_variation_rate = (
        inertial_coupling
        * N_background_prime_GeV2
        / inertial_mass_factor
    )
    scalar_force_potential = (
        parameters.eta_GeV_inv * delta_N_GeV / inertial_mass_factor
    )
    clock_velocity_source = (
        2.0
        * parameters.h_GeV_inv
        / inertial_mass_factor
        * (
            N_background_GeV * clock_velocity_prime_GeV
            + (
                N_background_prime_GeV2
                + conformal_hubble_GeV * N_background_GeV
            )
            * clock_velocity_projection
        )
    )
    density_rhs = -1j * wavenumber_GeV * velocity_projection + 3.0 * phi_prime_GeV
    velocity_rhs = (
        -(conformal_hubble_GeV + mass_variation_rate) * velocity_projection
        - 1j
        * wavenumber_GeV
        * (
            rest_mass_factor / inertial_mass_factor * psi
            + scalar_force_potential
        )
        + clock_velocity_source
        + collision_velocity_source
    )
    return density_rhs, velocity_rhs


def perfect_fluid_scalar_stress_moments(
    energy_density_GeV4,
    pressure_GeV4,
    density_contrast,
    velocity_projection,
    sound_speed_squared=0.0,
):
    """Return linear scalar stress moments for an isotropic perfect fluid.

    The moments use the orthonormal frame normal to constant-time slices and
    assume a barotropic pressure perturbation with no anisotropic stress.
    """
    background_values = (
        energy_density_GeV4,
        pressure_GeV4,
        sound_speed_squared,
    )
    perturbation_values = (density_contrast, velocity_projection)
    if any(
        np.ndim(value) != 0 or not np.isreal(value) or not np.isfinite(value)
        for value in background_values
    ) or any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in perturbation_values
    ):
        raise ValueError("Perfect-fluid stress inputs must be finite scalars.")
    if energy_density_GeV4 < 0.0 or sound_speed_squared < 0.0:
        raise ValueError("Energy density and sound speed must be nonnegative.")

    density_perturbation = energy_density_GeV4 * density_contrast
    return {
        "delta_rho_GeV4": density_perturbation,
        "delta_pressure_GeV4": sound_speed_squared * density_perturbation,
        "longitudinal_flux_GeV4": (
            energy_density_GeV4 + pressure_GeV4
        )
        * velocity_projection,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def coupled_pressureless_matter_stress_moments(
    bare_energy_density_GeV4,
    density_contrast,
    velocity_projection,
    N_background_GeV,
    delta_N_GeV,
    parameters,
    clock_velocity_projection=0.0,
):
    """Return linear dust stress moments from the coupled worldline action.

    density_contrast describes the conserved bare particle number. The
    returned density perturbation uses the rest-energy factor 1 + (g-h)N.
    The flux instead follows the inertial momentum from the action,
    rho_bare * [(1 + (g+h)N) V_m - 2 h N V_Theta], where V_Theta is the
    clock-frame velocity projection. The supplied bare density perturbation
    must already include metric/frame effects.
    """
    background_values = (bare_energy_density_GeV4, N_background_GeV)
    if any(
        np.ndim(value) != 0 or not np.isreal(value) or not np.isfinite(value)
        for value in background_values
    ):
        raise ValueError("Dust stress background inputs must be finite scalars.")
    perturbation_values = (
        density_contrast,
        velocity_projection,
        delta_N_GeV,
        clock_velocity_projection,
    )
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in perturbation_values
    ):
        raise ValueError("Dust stress perturbations must be finite scalars.")
    if bare_energy_density_GeV4 < 0.0:
        raise ValueError("Bare dust density must be nonnegative.")

    mass_factor = 1.0 + parameters.eta_GeV_inv * N_background_GeV
    if not np.isfinite(mass_factor) or mass_factor <= 0.0:
        raise ValueError("The effective dust rest mass must stay positive.")
    inertial_mass_factor = (
        1.0
        + (parameters.g_GeV_inv + parameters.h_GeV_inv)
        * N_background_GeV
    )
    if not np.isfinite(inertial_mass_factor) or inertial_mass_factor <= 0.0:
        raise ValueError("The effective dust inertia must stay positive.")
    bare_density_perturbation = bare_energy_density_GeV4 * density_contrast
    return {
        "delta_rho_GeV4": (
            mass_factor * bare_density_perturbation
            + parameters.eta_GeV_inv * bare_energy_density_GeV4 * delta_N_GeV
        ),
        "delta_pressure_GeV4": 0.0j,
        "longitudinal_flux_GeV4": bare_energy_density_GeV4
        * (
            inertial_mass_factor * velocity_projection
            - 2.0
            * parameters.h_GeV_inv
            * N_background_GeV
            * clock_velocity_projection
        ),
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def coupled_massive_worldline_hilbert_stress_tensor(
    bare_number_density_GeV3,
    particle_mass_GeV,
    N_GeV,
    clock_four_velocity_orthonormal,
    particle_four_velocity_orthonormal,
    parameters,
):
    """Return the action-derived Hilbert tensor for massive worldlines.

    Number density is measured on the local normal-frame slice. The vectors
    are future-directed contravariant unit velocities in that orthonormal
    frame. The decomposition retains the trace and clock-frame interaction
    tensors separately for use in phase-space integration.
    """
    scalar_values = (
        bare_number_density_GeV3,
        particle_mass_GeV,
        N_GeV,
    )
    if any(
        np.ndim(value) != 0
        or not np.isreal(value)
        or not np.isfinite(value)
        for value in scalar_values
    ):
        raise ValueError("Worldline stress inputs must be finite real scalars.")
    if bare_number_density_GeV3 < 0.0 or particle_mass_GeV <= 0.0:
        raise ValueError("Require nonnegative number density and positive mass.")

    velocities = []
    for value in (
        clock_four_velocity_orthonormal,
        particle_four_velocity_orthonormal,
    ):
        vector = np.asarray(value)
        if (
            vector.shape != (4,)
            or not np.issubdtype(vector.dtype, np.number)
            or not np.isrealobj(vector)
            or np.any(~np.isfinite(vector))
        ):
            raise ValueError("Worldline four-velocities must be finite real 4-vectors.")
        velocities.append(vector.astype(float))
    clock_velocity, particle_velocity = velocities

    def minkowski_norm(vector):
        return -vector[0] ** 2 + np.dot(vector[1:], vector[1:])

    for vector in velocities:
        if vector[0] <= 0.0 or not np.isclose(
            minkowski_norm(vector), -1.0, rtol=0.0, atol=1e-10
        ):
            raise ValueError("Worldline four-velocities must be future unit timelike.")

    relative_clock_factor = (
        -clock_velocity[0] * particle_velocity[0]
        + np.dot(clock_velocity[1:], particle_velocity[1:])
    )
    measure = (
        bare_number_density_GeV3
        * particle_mass_GeV
        / particle_velocity[0]
    )
    bare_tensor = measure * np.outer(particle_velocity, particle_velocity)
    trace_interaction = (
        parameters.g_GeV_inv * N_GeV * bare_tensor
    )
    clock_interaction = (
        measure
        * parameters.h_GeV_inv
        * N_GeV
        * relative_clock_factor**2
        * (
            np.outer(particle_velocity, particle_velocity)
            - 2.0 * np.outer(clock_velocity, clock_velocity)
        )
    )
    return {
        "bare_stress_tensor_GeV4": bare_tensor,
        "trace_interaction_stress_tensor_GeV4": trace_interaction,
        "clock_interaction_stress_tensor_GeV4": clock_interaction,
        "total_stress_tensor_GeV4": (
            bare_tensor + trace_interaction + clock_interaction
        ),
    }


def uncoupled_kinetic_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    mass_GeV,
    scale_factor,
    distribution_multipoles,
    degeneracy=1.0,
):
    """Integrate delta-f contributions to local-frame uncoupled stress moments.

    Returns the density, isotropic-pressure, longitudinal-flux, and
    longitudinal-anisotropic-stress contributions from F_0, F_1, and F_2.
    Interaction stress and explicit metric/frame variations are not included.
    """
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or moments.ndim != 2
        or moments.shape[0] != momenta.size
        or moments.shape[1] < 3
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(moments))
        or not np.isfinite(mass_GeV)
        or mass_GeV < 0.0
        or not np.isfinite(scale_factor)
        or scale_factor <= 0.0
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Kinetic stress inputs must be finite and physical.")

    physical_momenta = momenta / scale_factor
    energies = np.hypot(physical_momenta, mass_GeV)
    velocities = np.divide(
        physical_momenta,
        energies,
        out=np.zeros_like(physical_momenta),
        where=energies > 0.0,
    )
    momentum_measure = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        / scale_factor**3
    )
    density = np.sum(momentum_measure * energies * moments[:, 0])
    pressure = np.sum(
        momentum_measure * physical_momenta * velocities * moments[:, 0] / 3.0
    )
    longitudinal_flux = np.sum(
        momentum_measure * physical_momenta * moments[:, 1]
    )
    anisotropic_stress = np.sum(
        momentum_measure
        * np.divide(
            2.0 * physical_momenta**2,
            3.0 * energies,
            out=np.zeros_like(physical_momenta),
            where=energies > 0.0,
        )
        * moments[:, 2]
    )
    return {
        "delta_rho_GeV4": density,
        "delta_pressure_GeV4": pressure,
        "longitudinal_flux_GeV4": longitudinal_flux,
        "longitudinal_anisotropic_stress_GeV4": anisotropic_stress,
    }


def neutrino_kinetic_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    mass_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=1.0,
):
    """Return delta-f moments weighted by the background-coupled dispersion.

    Direct f_0 responses to delta-Hamiltonian, metric/frame measure changes,
    and remaining contact terms from varying the interaction action are not
    included.
    """
    if not np.isfinite(N_background_GeV):
        raise ValueError("The scalar field value must be finite.")
    uncoupled_moments = uncoupled_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        mass_GeV,
        scale_factor,
        distribution_multipoles,
        degeneracy,
    )
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    physical_momenta = momenta / scale_factor
    energies = np.hypot(physical_momenta, mass_GeV)
    scalar_response = np.divide(
        parameters.g_GeV_inv * mass_GeV**2,
        energies,
        out=np.zeros_like(energies),
        where=energies > 0.0,
    ) - parameters.h_GeV_inv * energies
    coupled_energies = energies + N_background_GeV * scalar_response
    group_velocities = neutrino_group_velocity_local(
        physical_momenta, mass_GeV, N_background_GeV, parameters
    )
    uncoupled_velocities = np.divide(
        physical_momenta,
        energies,
        out=np.zeros_like(physical_momenta),
        where=energies > 0.0,
    )
    if np.any(~np.isfinite(coupled_energies)) or np.any(coupled_energies < 0.0):
        raise ValueError("The first-order neutrino dispersion is not physical.")

    momentum_measure = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        / scale_factor**3
    )
    velocity_correction = group_velocities - uncoupled_velocities
    return {
        "delta_rho_GeV4": (
            uncoupled_moments["delta_rho_GeV4"]
            + np.sum(
                momentum_measure
                * N_background_GeV
                * scalar_response
                * moments[:, 0]
            )
        ),
        "delta_pressure_GeV4": (
            uncoupled_moments["delta_pressure_GeV4"]
            + np.sum(
                momentum_measure
                * physical_momenta
                * velocity_correction
                * moments[:, 0]
                / 3.0
            )
        ),
        "longitudinal_flux_GeV4": uncoupled_moments[
            "longitudinal_flux_GeV4"
        ],
        "longitudinal_anisotropic_stress_GeV4": (
            uncoupled_moments["longitudinal_anisotropic_stress_GeV4"]
            + np.sum(
                momentum_measure
                * (2.0 / 3.0)
                * physical_momenta
                * velocity_correction
                * moments[:, 2]
            )
        ),
    }


def neutrino_dispersion_interaction_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    mass_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=1.0,
):
    """Return the fixed-clock-frame delta-f correction from neutrino dispersion.

    This is the coupled-dispersion stress minus the uncoupled kinetic stress;
    direct metric and clock-frame variations are not included.
    """
    coupled_moments = neutrino_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        mass_GeV,
        scale_factor,
        N_background_GeV,
        distribution_multipoles,
        parameters,
        degeneracy,
    )
    uncoupled_moments = uncoupled_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        mass_GeV,
        scale_factor,
        distribution_multipoles,
        degeneracy,
    )
    return {
        "delta_rho_GeV4": (
            coupled_moments["delta_rho_GeV4"]
            - uncoupled_moments["delta_rho_GeV4"]
        ),
        "delta_pressure_GeV4": (
            coupled_moments["delta_pressure_GeV4"]
            - uncoupled_moments["delta_pressure_GeV4"]
        ),
        "longitudinal_flux_GeV4": (
            coupled_moments["longitudinal_flux_GeV4"]
            - uncoupled_moments["longitudinal_flux_GeV4"]
        ),
        "longitudinal_anisotropic_stress_GeV4": (
            coupled_moments["longitudinal_anisotropic_stress_GeV4"]
            - uncoupled_moments["longitudinal_anisotropic_stress_GeV4"]
        ),
    }


def neutrino_worldline_hilbert_contact_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    mass_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=1.0,
):
    """Return the Hilbert contact correction to coupled-dispersion moments.

    Assumes the background clock is aligned with the local normal. The result
    is the direct worldline interaction tensor minus the interaction already
    represented by the coupled Hamiltonian moments. It covers delta-f only;
    delta-N, clock-frame, and metric responses remain separate.
    """
    if (
        np.ndim(mass_GeV) != 0
        or not np.isreal(mass_GeV)
        or not np.isfinite(mass_GeV)
        or mass_GeV <= 0.0
    ):
        raise ValueError("The worldline contact stress requires positive mass.")

    dispersion_moments = neutrino_dispersion_interaction_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        mass_GeV,
        scale_factor,
        N_background_GeV,
        distribution_multipoles,
        parameters,
        degeneracy,
    )
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    physical_momenta = momenta / scale_factor
    energies = np.hypot(physical_momenta, mass_GeV)
    momentum_measure = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        / scale_factor**3
    )
    direct_interaction = {
        "delta_rho_GeV4": np.sum(
            momentum_measure
            * N_background_GeV
            * (
                parameters.g_GeV_inv * energies
                + parameters.h_GeV_inv
                * energies
                * (energies**2 / mass_GeV**2 - 2.0)
            )
            * moments[:, 0]
        ),
        "delta_pressure_GeV4": np.sum(
            momentum_measure
            * N_background_GeV
            * (
                parameters.g_GeV_inv * physical_momenta**2 / energies
                + parameters.h_GeV_inv
                * energies
                * physical_momenta**2
                / mass_GeV**2
            )
            * moments[:, 0]
            / 3.0
        ),
        "longitudinal_flux_GeV4": np.sum(
            momentum_measure
            * N_background_GeV
            * physical_momenta
            * (
                parameters.g_GeV_inv
                + parameters.h_GeV_inv * energies**2 / mass_GeV**2
            )
            * moments[:, 1]
        ),
        "longitudinal_anisotropic_stress_GeV4": np.sum(
            momentum_measure
            * (2.0 / 3.0)
            * N_background_GeV
            * (
                parameters.g_GeV_inv * physical_momenta**2 / energies
                + parameters.h_GeV_inv
                * energies
                * physical_momenta**2
                / mass_GeV**2
            )
            * moments[:, 2]
        ),
    }
    return {
        key: direct_interaction[key] - dispersion_moments[key]
        for key in direct_interaction
    }


def neutrino_worldline_hilbert_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    mass_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=1.0,
):
    """Return action-derived delta-f stress moments for massive neutrinos.

    Combines the coupled-Hamiltonian moments with the remaining Hilbert
    contact term. Background delta-N, metric, and clock-frame responses are
    not included.
    """
    kinetic_moments = neutrino_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        mass_GeV,
        scale_factor,
        N_background_GeV,
        distribution_multipoles,
        parameters,
        degeneracy,
    )
    contact_moments = neutrino_worldline_hilbert_contact_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        mass_GeV,
        scale_factor,
        N_background_GeV,
        distribution_multipoles,
        parameters,
        degeneracy,
    )
    return {
        key: kinetic_moments[key] + contact_moments[key]
        for key in kinetic_moments
    }


def photon_kinetic_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=2.0,
):
    """Return fixed-background delta-f moments of the full Maxwell stress.

    The local mode weights include the hN interaction stress. The separate
    photon_maxwell_stress_multipoles helper returns its T^(0) part. Responses
    to delta-N and metric/frame perturbations of the mode measure are separate.
    """
    uncoupled_moments = uncoupled_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        0.0,
        scale_factor,
        distribution_multipoles,
        degeneracy,
    )
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    physical_momenta = momenta / scale_factor
    momentum_measure = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        / scale_factor**3
    )
    speed_correction = photon_speed - 1.0
    return {
        "delta_rho_GeV4": (
            uncoupled_moments["delta_rho_GeV4"]
            + np.sum(
                momentum_measure
                * speed_correction
                * physical_momenta
                * moments[:, 0]
            )
        ),
        "delta_pressure_GeV4": (
            uncoupled_moments["delta_pressure_GeV4"]
            + np.sum(
                momentum_measure
                * speed_correction
                * physical_momenta
                * moments[:, 0]
                / 3.0
            )
        ),
        "longitudinal_flux_GeV4": (
            photon_speed**2
            * uncoupled_moments["longitudinal_flux_GeV4"]
        ),
        "longitudinal_anisotropic_stress_GeV4": (
            uncoupled_moments["longitudinal_anisotropic_stress_GeV4"]
            + np.sum(
                momentum_measure
                * (2.0 / 3.0)
                * speed_correction
                * physical_momenta
                * moments[:, 2]
            )
        ),
    }


def photon_maxwell_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=2.0,
):
    """Return delta-f moments of T^(0)_gamma on the modified photon cone.

    For a local plane-wave mode, the Maxwell energy and anisotropic-stress
    weight is E_gamma/(1 - (hN)^2), while its Poynting-flux weight is
    E_gamma/sqrt(1 - (hN)^2). Direct metric/frame responses are not included.
    """
    uncoupled_moments = uncoupled_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        0.0,
        scale_factor,
        distribution_multipoles,
        degeneracy,
    )
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    moments = np.asarray(distribution_multipoles, dtype=complex)
    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    maxwell_energy_ratio = photon_speed / (1.0 - xi**2)
    maxwell_flux_ratio = photon_speed / np.sqrt(1.0 - xi**2)
    density = (
        maxwell_energy_ratio * uncoupled_moments["delta_rho_GeV4"]
    )
    return {
        "delta_rho_GeV4": density,
        "delta_pressure_GeV4": density / 3.0,
        "longitudinal_flux_GeV4": (
            maxwell_flux_ratio
            * uncoupled_moments["longitudinal_flux_GeV4"]
        ),
        "longitudinal_anisotropic_stress_GeV4": (
            maxwell_energy_ratio
            * uncoupled_moments[
                "longitudinal_anisotropic_stress_GeV4"
            ]
        ),
    }


def photon_interaction_stress_multipoles(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    scale_factor,
    N_background_GeV,
    distribution_multipoles,
    parameters,
    degeneracy=2.0,
):
    """Return the fixed-background delta-f stress from the hN interaction."""
    uncoupled_moments = uncoupled_kinetic_stress_multipoles(
        comoving_momenta_GeV,
        comoving_quadrature_weights_GeV,
        0.0,
        scale_factor,
        distribution_multipoles,
        degeneracy,
    )
    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    interaction_energy_ratio = (
        -xi**2 * photon_speed / (1.0 - xi**2)
    )
    interaction_flux_ratio = -xi / (1.0 + xi)
    density = (
        interaction_energy_ratio * uncoupled_moments["delta_rho_GeV4"]
    )
    return {
        "delta_rho_GeV4": density,
        "delta_pressure_GeV4": density / 3.0,
        "longitudinal_flux_GeV4": (
            interaction_flux_ratio
            * uncoupled_moments["longitudinal_flux_GeV4"]
        ),
        "longitudinal_anisotropic_stress_GeV4": (
            interaction_energy_ratio
            * uncoupled_moments[
                "longitudinal_anisotropic_stress_GeV4"
            ]
        ),
    }


def photon_interaction_hilbert_stress_tensor(
    field_strength,
    metric_covariant,
    clock_velocity,
    N_background_GeV,
    parameters,
):
    """Return the covariant Hilbert tensor of the photon hN interaction.

    The variation holds F_mu_nu and N fixed and varies the metric while
    keeping the contravariant clock velocity unit-normalized.
    """
    field = np.asarray(field_strength, dtype=float)
    metric = np.asarray(metric_covariant, dtype=float)
    velocity = np.asarray(clock_velocity, dtype=float)
    if (
        field.shape != (4, 4)
        or metric.shape != (4, 4)
        or velocity.shape != (4,)
        or np.any(~np.isfinite(field))
        or np.any(~np.isfinite(metric))
        or np.any(~np.isfinite(velocity))
        or np.ndim(N_background_GeV) != 0
        or not np.isfinite(N_background_GeV)
        or not np.allclose(metric, metric.T, rtol=0.0, atol=1e-12)
        or not np.allclose(field, -field.T, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Photon Hilbert-stress inputs must be finite tensors.")

    metric_eigenvalues = np.linalg.eigvalsh(metric)
    if metric_eigenvalues[0] >= 0.0 or metric_eigenvalues[1] <= 0.0:
        raise ValueError("The photon metric must have signature (-,+,+,+).")
    velocity_norm = velocity @ metric @ velocity
    if not np.isclose(velocity_norm, -1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("The photon clock velocity must be unit-normalized.")

    inverse_metric = np.linalg.inv(metric)
    electric_field = field @ velocity
    electric_squared = electric_field @ inverse_metric @ electric_field
    field_squared = np.sum(
        field * (inverse_metric @ field @ inverse_metric)
    )
    energy_density = electric_squared + 0.25 * field_squared
    field_contraction = field @ inverse_metric @ field.T
    covariant_velocity = metric @ velocity
    interaction_strength = parameters.h_GeV_inv * N_background_GeV
    if not np.isfinite(interaction_strength):
        raise ValueError("The photon interaction strength must be finite.")

    return interaction_strength * (
        metric * energy_density
        - 2.0 * np.outer(electric_field, electric_field)
        - field_contraction
        + 2.0
        * electric_squared
        * np.outer(covariant_velocity, covariant_velocity)
    )


def neutrino_background_dispersion_stress_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    N_background_GeV,
    delta_N_GeV,
    background_distribution,
    parameters,
    degeneracy=1.0,
):
    """Return the fixed-local-momentum f0*delta-N neutrino stress response."""
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or not np.isfinite(mass_GeV)
        or mass_GeV < 0.0
        or not np.isfinite(N_background_GeV)
        or np.ndim(delta_N_GeV) != 0
        or not np.isfinite(delta_N_GeV)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Neutrino dispersion-response inputs must be physical.")

    uncoupled_energy = np.hypot(momenta, mass_GeV)
    energy_response = np.divide(
        parameters.g_GeV_inv * mass_GeV**2,
        uncoupled_energy,
        out=np.zeros_like(uncoupled_energy),
        where=uncoupled_energy > 0.0,
    ) - parameters.h_GeV_inv * uncoupled_energy
    coupled_energy = uncoupled_energy + N_background_GeV * energy_response
    group_velocity = neutrino_group_velocity_local(
        momenta, mass_GeV, N_background_GeV, parameters
    )
    velocity_response = -parameters.g_GeV_inv * np.divide(
        mass_GeV**2 * momenta,
        uncoupled_energy**3,
        out=np.zeros_like(uncoupled_energy),
        where=uncoupled_energy > 0.0,
    ) - parameters.h_GeV_inv * np.divide(
        momenta,
        uncoupled_energy,
        out=np.zeros_like(uncoupled_energy),
        where=uncoupled_energy > 0.0,
    )
    if np.any(~np.isfinite(coupled_energy)) or np.any(coupled_energy < 0.0):
        raise ValueError("The first-order neutrino dispersion is not physical.")

    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    return {
        "delta_rho_GeV4": delta_N_GeV
        * np.sum(phase_space_weights * energy_response),
        "delta_pressure_GeV4": delta_N_GeV
        * np.sum(phase_space_weights * momenta * velocity_response / 3.0),
        "longitudinal_flux_GeV4": 0.0j,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def neutrino_worldline_hilbert_background_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    N_background_GeV,
    delta_N_GeV,
    background_distribution,
    parameters,
    degeneracy=1.0,
):
    """Return the fixed-local-momentum Hilbert response to f0*delta-N.

    Adds the aligned-clock worldline contact to the Hamiltonian dispersion
    response. Spatial-metric, lapse, and clock-frame responses are separate.
    """
    if (
        np.ndim(mass_GeV) != 0
        or not np.isreal(mass_GeV)
        or not np.isfinite(mass_GeV)
        or mass_GeV <= 0.0
    ):
        raise ValueError("The worldline response requires positive mass.")

    dispersion_response = neutrino_background_dispersion_stress_moments(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        mass_GeV,
        N_background_GeV,
        delta_N_GeV,
        background_distribution,
        parameters,
        degeneracy,
    )
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    energies = np.hypot(momenta, mass_GeV)
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    contact_energy_response = np.sum(
        phase_space_weights
        * momenta**2
        / energies
        * (
            parameters.g_GeV_inv
            + parameters.h_GeV_inv * energies**2 / mass_GeV**2
        )
    )
    contact_pressure_response = np.sum(
        phase_space_weights
        * momenta**2
        / energies
        * (
            parameters.g_GeV_inv * (1.0 + mass_GeV**2 / energies**2)
            + parameters.h_GeV_inv * (1.0 + energies**2 / mass_GeV**2)
        )
        / 3.0
    )
    return {
        "delta_rho_GeV4": (
            dispersion_response["delta_rho_GeV4"]
            + delta_N_GeV * contact_energy_response
        ),
        "delta_pressure_GeV4": (
            dispersion_response["delta_pressure_GeV4"]
            + delta_N_GeV * contact_pressure_response
        ),
        "longitudinal_flux_GeV4": dispersion_response[
            "longitudinal_flux_GeV4"
        ],
        "longitudinal_anisotropic_stress_GeV4": dispersion_response[
            "longitudinal_anisotropic_stress_GeV4"
        ],
    }


def neutrino_spatial_metric_measure_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    N_background_GeV,
    curvature_perturbation,
    background_distribution,
    parameters,
    degeneracy=1.0,
):
    """Return isotropic f0 stress responses to the spatial metric measure.

    Includes only p = q/[a(1 - Phi)] and its phase-space measure change;
    lapse, clock-frame, and direct Hamiltonian responses are separate.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or not np.isfinite(mass_GeV)
        or mass_GeV < 0.0
        or not np.isfinite(N_background_GeV)
        or np.ndim(curvature_perturbation) != 0
        or not np.isfinite(curvature_perturbation)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Neutrino metric-response inputs must be physical.")

    energies = np.hypot(momenta, mass_GeV)
    energy_response = np.divide(
        parameters.g_GeV_inv * mass_GeV**2,
        energies,
        out=np.zeros_like(energies),
        where=energies > 0.0,
    ) - parameters.h_GeV_inv * energies
    coupled_energies = energies + N_background_GeV * energy_response
    group_velocities = neutrino_group_velocity_local(
        momenta, mass_GeV, N_background_GeV, parameters
    )
    velocity_derivative = np.divide(
        mass_GeV**2,
        energies**3,
        out=np.zeros_like(energies),
        where=energies > 0.0,
    )
    velocity_derivative += N_background_GeV * (
        -parameters.g_GeV_inv
        * np.divide(
            mass_GeV**2 * (mass_GeV**2 - 2.0 * momenta**2),
            energies**5,
            out=np.zeros_like(energies),
            where=energies > 0.0,
        )
        - parameters.h_GeV_inv
        * np.divide(
            mass_GeV**2,
            energies**3,
            out=np.zeros_like(energies),
            where=energies > 0.0,
        )
    )
    if (
        np.any(~np.isfinite(coupled_energies))
        or np.any(coupled_energies < 0.0)
        or np.any(~np.isfinite(velocity_derivative))
    ):
        raise ValueError("The first-order neutrino dispersion is not physical.")

    phase_space_weights = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        * distribution
    )
    density_response = 3.0 * coupled_energies + momenta * group_velocities
    pressure_response = (
        (4.0 / 3.0) * momenta * group_velocities
        + momenta**2 * velocity_derivative / 3.0
    )
    return {
        "delta_rho_GeV4": curvature_perturbation
        * np.sum(phase_space_weights * density_response),
        "delta_pressure_GeV4": curvature_perturbation
        * np.sum(phase_space_weights * pressure_response),
        "longitudinal_flux_GeV4": 0.0j,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def neutrino_worldline_hilbert_spatial_metric_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    N_background_GeV,
    curvature_perturbation,
    background_distribution,
    parameters,
    degeneracy=1.0,
):
    """Return the aligned-clock Hilbert response to f0 spatial curvature.

    Adds the spatial-measure variation of the worldline contact to the
    coupled-Hamiltonian response. Lapse and clock-frame responses are separate.
    """
    if (
        np.ndim(mass_GeV) != 0
        or not np.isreal(mass_GeV)
        or not np.isfinite(mass_GeV)
        or mass_GeV <= 0.0
    ):
        raise ValueError("The worldline response requires positive mass.")

    dispersion_response = neutrino_spatial_metric_measure_response_moments(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        mass_GeV,
        N_background_GeV,
        curvature_perturbation,
        background_distribution,
        parameters,
        degeneracy,
    )
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    energies = np.hypot(momenta, mass_GeV)
    velocity_squared = momenta**2 / energies**2
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    contact_density_response = np.sum(
        phase_space_weights
        * momenta**2
        / energies
        * (
            parameters.g_GeV_inv * (5.0 - velocity_squared)
            + parameters.h_GeV_inv
            * energies**2
            / mass_GeV**2
            * (5.0 + velocity_squared)
        )
    )
    contact_pressure_response = np.sum(
        phase_space_weights
        * momenta**2
        / (3.0 * energies)
        * (
            parameters.g_GeV_inv
            * (
                5.0
                - velocity_squared
                + mass_GeV**2
                / energies**2
                * (5.0 - 3.0 * velocity_squared)
            )
            + parameters.h_GeV_inv
            * (
                5.0
                - velocity_squared
                + energies**2
                / mass_GeV**2
                * (5.0 + velocity_squared)
            )
        )
    )
    return {
        "delta_rho_GeV4": (
            dispersion_response["delta_rho_GeV4"]
            + curvature_perturbation
            * N_background_GeV
            * contact_density_response
        ),
        "delta_pressure_GeV4": (
            dispersion_response["delta_pressure_GeV4"]
            + curvature_perturbation
            * N_background_GeV
            * contact_pressure_response
        ),
        "longitudinal_flux_GeV4": dispersion_response[
            "longitudinal_flux_GeV4"
        ],
        "longitudinal_anisotropic_stress_GeV4": dispersion_response[
            "longitudinal_anisotropic_stress_GeV4"
        ],
    }


def neutrino_worldline_hilbert_clock_frame_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    mass_GeV,
    N_background_GeV,
    frame_velocity_projection,
    background_distribution,
    parameters,
    degeneracy=1.0,
):
    """Return the linear f0 Hilbert response to a clock-frame tilt.

    The background clock is aligned with the local normal and the distribution
    is isotropic in that frame. At first order only the longitudinal flux is
    nonzero; the Hamiltonian depends on the relative clock velocity quadratically.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(mass_GeV) != 0
        or not np.isreal(mass_GeV)
        or not np.isfinite(mass_GeV)
        or mass_GeV <= 0.0
        or np.ndim(N_background_GeV) != 0
        or not np.isreal(N_background_GeV)
        or not np.isfinite(N_background_GeV)
        or np.ndim(frame_velocity_projection) != 0
        or not np.isfinite(frame_velocity_projection)
        or np.ndim(degeneracy) != 0
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Neutrino clock-frame inputs must be physical.")

    energies = np.hypot(momenta, mass_GeV)
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    flux_response = -2.0 * (
        parameters.h_GeV_inv
        * N_background_GeV
        * frame_velocity_projection
        * np.sum(
            phase_space_weights
            * energies
            * (1.0 + momenta**2 / (3.0 * mass_GeV**2))
        )
    )
    return {
        "delta_rho_GeV4": 0.0j,
        "delta_pressure_GeV4": 0.0j,
        "longitudinal_flux_GeV4": flux_response,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def photon_background_dispersion_stress_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    delta_N_GeV,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the fixed-local-momentum f0*delta-N photon stress response."""
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or not np.isfinite(N_background_GeV)
        or np.ndim(delta_N_GeV) != 0
        or not np.isfinite(delta_N_GeV)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon dispersion-response inputs must be physical.")

    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    speed_response = -parameters.h_GeV_inv * photon_speed / (1.0 - xi**2)
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    density_response = np.sum(
        phase_space_weights * speed_response * momenta
    )
    return {
        "delta_rho_GeV4": delta_N_GeV * density_response,
        "delta_pressure_GeV4": delta_N_GeV * density_response / 3.0,
        "longitudinal_flux_GeV4": 0.0j,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def photon_maxwell_background_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    delta_N_GeV,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return fixed-local-momentum f0*delta-N moments of T^(0)_gamma."""
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or not np.isfinite(N_background_GeV)
        or np.ndim(delta_N_GeV) != 0
        or not np.isfinite(delta_N_GeV)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Maxwell stress-response inputs must be physical.")

    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    density_response = (
        parameters.h_GeV_inv
        * photon_speed
        * momenta
        * (2.0 * xi - 1.0)
        / (1.0 - xi**2) ** 2
    )
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    density = delta_N_GeV * np.sum(phase_space_weights * density_response)
    return {
        "delta_rho_GeV4": density,
        "delta_pressure_GeV4": density / 3.0,
        "longitudinal_flux_GeV4": 0.0j,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def photon_interaction_background_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    delta_N_GeV,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the fixed-local-momentum f0*delta-N interaction stress."""
    total_response = photon_background_dispersion_stress_moments(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        N_background_GeV,
        delta_N_GeV,
        background_distribution,
        parameters,
        degeneracy,
    )
    maxwell_response = photon_maxwell_background_response_moments(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        N_background_GeV,
        delta_N_GeV,
        background_distribution,
        parameters,
        degeneracy,
    )
    return {
        moment_name: total_response[moment_name] - maxwell_response[moment_name]
        for moment_name in total_response
    }


def photon_spatial_metric_measure_response_moments(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    scale_factor,
    N_background_GeV,
    curvature_perturbation,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return isotropic f0 responses to the spatial momentum measure.

    Includes only local momentum and phase-space measure changes from
    g_ij = a^2 (1 - 2 Phi) delta_ij, separately for total, Maxwell, and
    interaction photon energy weights. It excludes delta-H and frame terms.
    """
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        distribution.shape != momenta.shape
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(N_background_GeV) != 0
        or not np.isfinite(N_background_GeV)
        or np.ndim(curvature_perturbation) != 0
        or not np.isfinite(curvature_perturbation)
    ):
        raise ValueError("Photon metric-response inputs must be physical.")

    isotropic_multipoles = np.zeros((momenta.size, 3), dtype=complex)
    isotropic_multipoles[:, 0] = distribution
    uncoupled_moments = uncoupled_kinetic_stress_multipoles(
        momenta,
        weights,
        0.0,
        scale_factor,
        isotropic_multipoles,
        degeneracy,
    )
    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    maxwell_ratio = photon_speed / (1.0 - xi**2)
    base_density = uncoupled_moments["delta_rho_GeV4"]
    sector_densities = {
        "modified_action": photon_speed * base_density,
        "maxwell": maxwell_ratio * base_density,
        "interaction": -xi**2 * maxwell_ratio * base_density,
    }
    return {
        sector: {
            "delta_rho_GeV4": 4.0 * curvature_perturbation * density,
            "delta_pressure_GeV4": (
                4.0 * curvature_perturbation * density / 3.0
            ),
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for sector, density in sector_densities.items()
    }


def isotropic_fluid_scalar_source(rho_GeV4, pressure_GeV4, parameters):
    """Return -g T^mu_mu - h T_uu for an isotropic fluid in the u frame."""
    if not np.isfinite(rho_GeV4) or not np.isfinite(pressure_GeV4):
        raise ValueError("Fluid density and pressure must be finite.")
    return (
        parameters.eta_GeV_inv * rho_GeV4
        - 3.0 * parameters.g_GeV_inv * pressure_GeV4
    )


def isotropic_fluid_scalar_source_perturbation(
    delta_rho_GeV4,
    delta_pressure_GeV4,
    parameters,
):
    """Contract complete clock-frame density and pressure perturbations.

    The component has an isotropic background. Any required metric and frame
    variations must already be included in the supplied moments.
    """
    values = (delta_rho_GeV4, delta_pressure_GeV4)
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in values
    ):
        raise ValueError("Scalar-source perturbations must be finite scalars.")
    return (
        parameters.eta_GeV_inv * delta_rho_GeV4
        - 3.0 * parameters.g_GeV_inv * delta_pressure_GeV4
    )


def photon_clock_frame_stress_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    frame_velocity_projection,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return isotropic f0 moments of the photon ray-energy current response.

    Uses H times the group velocity for the conserved ray-energy current in
    the nondispersive geometric-optics limit.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(N_background_GeV) != 0
        or not np.isfinite(N_background_GeV)
        or np.ndim(frame_velocity_projection) != 0
        or not np.isfinite(frame_velocity_projection)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon clock-frame response inputs must be physical.")

    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    phase_space_weights = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        * distribution
    )
    background_density = np.sum(
        phase_space_weights * photon_speed * momenta
    )
    return {
        "delta_rho_GeV4": 0.0j,
        "delta_pressure_GeV4": 0.0j,
        "longitudinal_flux_GeV4": (
            (4.0 / 3.0)
            * (1.0 - photon_speed**2)
            * background_density
            * frame_velocity_projection
        ),
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def photon_maxwell_clock_frame_stress_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    frame_velocity_projection,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the clock-frame flux response of the bare Maxwell stress.

    The isotropic-stress boost contributes 4 r_E rho_bare V_u / 3, while the
    induced distribution dipole contributes -4 c_gamma r_F rho_bare V_u / 3
    after integration by parts. Here r_E and r_F are the Maxwell energy and
    flux weights; the momentum-space boundary term must vanish.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(N_background_GeV) != 0
        or not np.isfinite(N_background_GeV)
        or np.ndim(frame_velocity_projection) != 0
        or not np.isfinite(frame_velocity_projection)
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon clock-frame response inputs must be physical.")
    speed = photon_group_velocity_local(N_background_GeV, parameters)
    xi = parameters.h_GeV_inv * N_background_GeV
    energy_ratio = speed / (1.0 - xi**2)
    flux_ratio = speed / np.sqrt(1.0 - xi**2)
    phase_space_weights = (
        degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
    )
    bare_density = np.sum(phase_space_weights * momenta)
    frame_flux = (
        (4.0 / 3.0)
        * (energy_ratio - speed * flux_ratio)
        * bare_density
        * frame_velocity_projection
    )
    return {
        "delta_rho_GeV4": 0.0j,
        "delta_pressure_GeV4": 0.0j,
        "longitudinal_flux_GeV4": frame_flux,
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def photon_interaction_hilbert_clock_frame_contact_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    frame_velocity_projection,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the explicit clock-velocity variation of photon T_int.

    This contact term varies the normalized clock vector at fixed metric and
    fixed field strength. Angular and polarization averaging gives
    Q_contact = 4/3 * xi * c_gamma/(1 + xi) * rho_bare * V_u.
    """
    momenta = np.asarray(physical_momenta_GeV, dtype=float)
    weights = np.asarray(physical_quadrature_weights_GeV, dtype=float)
    distribution = np.asarray(background_distribution, dtype=float)
    if (
        momenta.ndim != 1
        or momenta.size == 0
        or weights.shape != momenta.shape
        or distribution.shape != momenta.shape
        or np.any(~np.isfinite(momenta))
        or np.any(momenta < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(~np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or np.ndim(N_background_GeV) != 0
        or not np.isfinite(N_background_GeV)
        or np.ndim(frame_velocity_projection) != 0
        or not np.isfinite(frame_velocity_projection)
        or np.ndim(degeneracy) != 0
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise ValueError("Photon Hilbert clock-contact inputs must be physical.")

    photon_speed = photon_group_velocity_local(N_background_GeV, parameters)
    interaction_strength = parameters.h_GeV_inv * N_background_GeV
    if not np.isfinite(interaction_strength) or abs(interaction_strength) >= 1.0:
        raise ValueError("The photon interaction strength must satisfy |hN| < 1.")
    phase_space_weights = (
        degeneracy
        / (2.0 * np.pi**2)
        * weights
        * momenta**2
        * distribution
    )
    bare_density = np.sum(phase_space_weights * momenta)
    return {
        "delta_rho_GeV4": 0.0j,
        "delta_pressure_GeV4": 0.0j,
        "longitudinal_flux_GeV4": (
            (4.0 / 3.0)
            * interaction_strength
            * photon_speed
            / (1.0 + interaction_strength)
            * bare_density
            * frame_velocity_projection
        ),
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


def photon_interaction_clock_frame_stress_response_moments(
    physical_momenta_GeV,
    physical_quadrature_weights_GeV,
    N_background_GeV,
    frame_velocity_projection,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Return the hN clock-frame stress response as total minus Maxwell."""
    total_response = photon_clock_frame_stress_response_moments(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        N_background_GeV,
        frame_velocity_projection,
        background_distribution,
        parameters,
        degeneracy,
    )
    maxwell_response = photon_maxwell_clock_frame_stress_response_moments(
        physical_momenta_GeV,
        physical_quadrature_weights_GeV,
        N_background_GeV,
        frame_velocity_projection,
        background_distribution,
        parameters,
        degeneracy,
    )
    return {
        moment_name: total_response[moment_name] - maxwell_response[moment_name]
        for moment_name in total_response
    }


def photon_action_stress_perturbation_moments(
    comoving_momenta_GeV,
    comoving_quadrature_weights_GeV,
    scale_factor,
    N_background_GeV,
    delta_N_GeV,
    curvature_perturbation,
    clock_velocity_projection,
    distribution_multipoles,
    background_distribution,
    parameters,
    degeneracy=2.0,
):
    """Compose total photon stress moments and the bare-Maxwell N source.

    The returned Maxwell and interaction mappings separately combine the
    implemented fixed-background delta-f, fixed-momentum delta-N,
    spatial-metric-measure, and ray-current responses. The interaction
    mapping also includes the explicit fixed-field Hilbert clock-vector
    contact. The fixed-background interaction delta-f moments and clock
    contact are checked against angular and polarization averaging of the
    field-level Hilbert tensor; general perturbed metric variations of that
    tensor are not independently evaluated here. The N source contracts only
    the uncoupled Maxwell density and pressure.
    """
    momenta = np.asarray(comoving_momenta_GeV, dtype=float)
    weights = np.asarray(comoving_quadrature_weights_GeV, dtype=float)
    if (
        np.ndim(scale_factor) != 0
        or not np.isreal(scale_factor)
        or not np.isfinite(scale_factor)
        or scale_factor <= 0.0
    ):
        raise ValueError("The photon stress scale factor must be positive.")
    physical_momenta = momenta / scale_factor
    physical_weights = weights / scale_factor
    metric_responses = photon_spatial_metric_measure_response_moments(
        momenta,
        weights,
        scale_factor,
        N_background_GeV,
        curvature_perturbation,
        background_distribution,
        parameters,
        degeneracy,
    )
    maxwell_stress = sum_scalar_stress_moments(
        photon_maxwell_stress_multipoles(
            momenta,
            weights,
            scale_factor,
            N_background_GeV,
            distribution_multipoles,
            parameters,
            degeneracy,
        ),
        photon_maxwell_background_response_moments(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            delta_N_GeV,
            background_distribution,
            parameters,
            degeneracy,
        ),
        metric_responses["maxwell"],
        photon_maxwell_clock_frame_stress_response_moments(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            clock_velocity_projection,
            background_distribution,
            parameters,
            degeneracy,
        ),
    )
    interaction_stress = sum_scalar_stress_moments(
        photon_interaction_stress_multipoles(
            momenta,
            weights,
            scale_factor,
            N_background_GeV,
            distribution_multipoles,
            parameters,
            degeneracy,
        ),
        photon_interaction_background_response_moments(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            delta_N_GeV,
            background_distribution,
            parameters,
            degeneracy,
        ),
        metric_responses["interaction"],
        photon_interaction_clock_frame_stress_response_moments(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            clock_velocity_projection,
            background_distribution,
            parameters,
            degeneracy,
        ),
        photon_interaction_hilbert_clock_frame_contact_moments(
            physical_momenta,
            physical_weights,
            N_background_GeV,
            clock_velocity_projection,
            background_distribution,
            parameters,
            degeneracy,
        ),
    )
    total_stress = sum_scalar_stress_moments(
        maxwell_stress,
        interaction_stress,
    )
    return {
        "total_stress_moments": total_stress,
        "maxwell_stress_moments": maxwell_stress,
        "interaction_stress_moments": interaction_stress,
        "bare_delta_rho_GeV4": maxwell_stress["delta_rho_GeV4"],
        "bare_delta_pressure_GeV4": maxwell_stress["delta_pressure_GeV4"],
        "scalar_source_perturbation_GeV3": (
            isotropic_fluid_scalar_source_perturbation(
                maxwell_stress["delta_rho_GeV4"],
                maxwell_stress["delta_pressure_GeV4"],
                parameters,
            )
        ),
    }


def newtonian_gauge_metric_constraints(
    wavenumber_GeV,
    conformal_hubble_GeV,
    scale_factor,
    reduced_planck_mass_GeV,
    total_delta_rho_GeV4,
    total_longitudinal_flux_GeV4,
    total_longitudinal_anisotropic_stress_GeV4,
):
    """Solve scalar Einstein constraints for complete total stress moments.

    Uses Fourier modes exp(i k.x) and Q = k_hat_i delta T^(hat 0 hat i).
    The supplied moments must include every species and all interaction,
    metric, and frame contributions; this helper does not assemble them.
    """
    background_values = (
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        reduced_planck_mass_GeV,
    )
    if any(
        np.ndim(value) != 0 or not np.isreal(value) or not np.isfinite(value)
        for value in background_values
    ):
        raise ValueError("Metric-constraint background inputs must be finite scalars.")
    stress_moments = (
        total_delta_rho_GeV4,
        total_longitudinal_flux_GeV4,
        total_longitudinal_anisotropic_stress_GeV4,
    )
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in stress_moments
    ):
        raise ValueError("Total stress moments must be finite scalars.")
    if (
        wavenumber_GeV <= 0.0
        or conformal_hubble_GeV < 0.0
        or scale_factor <= 0.0
        or reduced_planck_mass_GeV <= 0.0
    ):
        raise ValueError(
            "Require positive wavenumber, scale factor, and Planck mass, "
            "and nonnegative conformal Hubble rate."
        )

    constraint_scale = 2.0 * reduced_planck_mass_GeV**2
    momentum_constraint = (
        1j
        * scale_factor**2
        * total_longitudinal_flux_GeV4
        / (constraint_scale * wavenumber_GeV)
    )
    phi = (
        -scale_factor**2 * total_delta_rho_GeV4 / constraint_scale
        - 3.0 * conformal_hubble_GeV * momentum_constraint
    ) / wavenumber_GeV**2
    psi = phi + (
        3.0
        * scale_factor**2
        * total_longitudinal_anisotropic_stress_GeV4
        / (constraint_scale * wavenumber_GeV**2)
    )
    phi_prime = momentum_constraint - conformal_hubble_GeV * psi
    return {
        "phi": phi,
        "psi": psi,
        "phi_prime_GeV": phi_prime,
        "phi_prime_plus_hubble_psi_GeV": momentum_constraint,
    }


def canonical_scalar_stress_moments(
    background_field_prime_GeV2,
    field_perturbation_GeV,
    field_perturbation_prime_GeV2,
    scale_factor,
    psi,
    wavenumber_GeV,
    kinetic_normalization=1.0,
    potential_derivative_GeV3=0.0,
):
    """Return linear stress moments of a canonical scalar in the normal frame.

    The constant kinetic normalization covers N (one) and the clock field
    (Z_theta); potential_derivative_GeV3 is V_,field for the perturbed field.
    """
    real_values = (
        background_field_prime_GeV2,
        scale_factor,
        wavenumber_GeV,
        kinetic_normalization,
        potential_derivative_GeV3,
    )
    if any(
        np.ndim(value) != 0 or not np.isreal(value) or not np.isfinite(value)
        for value in real_values
    ):
        raise ValueError("Canonical-scalar background inputs must be finite scalars.")
    perturbation_values = (
        field_perturbation_GeV,
        field_perturbation_prime_GeV2,
        psi,
    )
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in perturbation_values
    ):
        raise ValueError("Canonical-scalar perturbations must be finite scalars.")
    if (
        scale_factor <= 0.0
        or wavenumber_GeV < 0.0
        or kinetic_normalization <= 0.0
    ):
        raise ValueError(
            "Require positive scale factor and kinetic normalization, "
            "and nonnegative wavenumber."
        )

    kinetic_perturbation = (
        kinetic_normalization
        / scale_factor**2
        * (
            background_field_prime_GeV2 * field_perturbation_prime_GeV2
            - psi * background_field_prime_GeV2**2
        )
    )
    potential_perturbation = (
        potential_derivative_GeV3 * field_perturbation_GeV
    )
    return {
        "delta_rho_GeV4": kinetic_perturbation + potential_perturbation,
        "delta_pressure_GeV4": kinetic_perturbation - potential_perturbation,
        "longitudinal_flux_GeV4": (
            -1j
            * wavenumber_GeV
            * kinetic_normalization
            * background_field_prime_GeV2
            * field_perturbation_GeV
            / scale_factor**2
        ),
        "longitudinal_anisotropic_stress_GeV4": 0.0j,
    }


_SCALAR_STRESS_MOMENT_KEYS = (
    "delta_rho_GeV4",
    "delta_pressure_GeV4",
    "longitudinal_flux_GeV4",
    "longitudinal_anisotropic_stress_GeV4",
)


def sum_scalar_stress_moments(*component_moments):
    """Sum supplied sector stress moments without assuming completeness.

    Each component must provide all four scalar stress keys. The caller is
    responsible for including every physical species and interaction term.
    """
    if not component_moments:
        raise ValueError("At least one component stress mapping is required.")

    total = {key: 0.0j for key in _SCALAR_STRESS_MOMENT_KEYS}
    for component in component_moments:
        if not isinstance(component, Mapping):
            raise ValueError("Each stress component must be a mapping.")
        for key in _SCALAR_STRESS_MOMENT_KEYS:
            if key not in component:
                raise ValueError(f"Stress component is missing {key}.")
            value = np.asarray(component[key])
            if (
                value.ndim != 0
                or not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value)
            ):
                raise ValueError("Stress moments must be finite numeric scalars.")
            total[key] += value.item()
    return total


def sum_scalar_source_perturbations(*component_sources):
    """Sum scalar source terms returned as scalars or photon mappings."""
    if not component_sources:
        raise ValueError("At least one scalar-source component is required.")

    total = 0.0j
    for component in component_sources:
        if isinstance(component, Mapping):
            source_key = "scalar_source_perturbation_GeV3"
            if source_key not in component:
                raise ValueError(
                    f"Scalar-source mapping is missing {source_key}."
                )
            component = component[source_key]
        value = np.asarray(component)
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value)
        ):
            raise ValueError("Scalar-source perturbations must be finite scalars.")
        total += value.item()
    return total


def assemble_coupled_dust_scalar_mode(
    wavenumber_GeV,
    conformal_hubble_GeV,
    scale_factor,
    parameters,
    background,
    perturbations,
):
    """Assemble the coupled dust, N, and clock scalar-mode subsystem.

    The caller supplies a consistent homogeneous background and the mode
    state. The dust source and stress follow the coupled worldline action,
    including the finite-h clock-current exchange. Photons, neutrinos,
    baryons, and collision terms are excluded. The algebraic Einstein
    constraints include the canonical scalars' lapse contribution to
    delta-rho and are solved self-consistently.
    """
    if not isinstance(background, Mapping) or not isinstance(
        perturbations, Mapping
    ):
        raise ValueError("Background and perturbations must be mappings.")

    background_keys = (
        "rho_m_bare_GeV4",
        "N_GeV",
        "N_prime_GeV2",
        "N_second_prime_GeV3",
        "Theta_prime_GeV2",
        "Theta_second_prime_GeV3",
    )
    perturbation_keys = (
        "delta_m",
        "velocity_m",
        "delta_N_GeV",
        "delta_N_prime_GeV2",
        "delta_Theta_GeV",
        "delta_Theta_prime_GeV2",
    )
    missing_background = [key for key in background_keys if key not in background]
    missing_perturbations = [
        key for key in perturbation_keys if key not in perturbations
    ]
    if missing_background or missing_perturbations:
        missing = missing_background + missing_perturbations
        raise ValueError(f"Mode assembly is missing inputs: {', '.join(missing)}.")

    real_inputs = (
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        *(background[key] for key in background_keys),
    )
    if any(
        np.ndim(value) != 0 or not np.isreal(value) or not np.isfinite(value)
        for value in real_inputs
    ):
        raise ValueError("Mode background inputs must be finite real scalars.")
    perturbation_values = tuple(
        perturbations[key] for key in perturbation_keys
    )
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in perturbation_values
    ):
        raise ValueError("Mode perturbations must be finite scalars.")
    if (
        wavenumber_GeV <= 0.0
        or conformal_hubble_GeV < 0.0
        or scale_factor <= 0.0
        or background["rho_m_bare_GeV4"] < 0.0
        or background["Theta_prime_GeV2"] <= 0.0
    ):
        raise ValueError(
            "Require positive wavenumber, scale factor, and clock prime; "
            "density must be nonnegative and conformal Hubble nonnegative."
        )

    bare_density = background["rho_m_bare_GeV4"]
    field = background["N_GeV"]
    field_prime = background["N_prime_GeV2"]
    theta_prime = background["Theta_prime_GeV2"]
    delta_m = perturbations["delta_m"]
    velocity_m = perturbations["velocity_m"]
    delta_field = perturbations["delta_N_GeV"]
    delta_field_prime = perturbations["delta_N_prime_GeV2"]
    delta_theta = perturbations["delta_Theta_GeV"]
    delta_theta_prime = perturbations["delta_Theta_prime_GeV2"]
    clock_velocity_projection = (
        -1j * wavenumber_GeV * delta_theta / theta_prime
    )
    clock_velocity_prime_GeV = -1j * wavenumber_GeV * (
        delta_theta_prime / theta_prime
        - delta_theta
        * background["Theta_second_prime_GeV3"]
        / theta_prime**2
    )
    potential_prime = potential_derivative(field, parameters)
    potential_second = (
        parameters.m_phi_GeV**2
        + 3.0 * parameters.lambda_N * field**2
    )

    dust_stress = coupled_pressureless_matter_stress_moments(
        bare_density,
        delta_m,
        velocity_m,
        field,
        delta_field,
        parameters,
        clock_velocity_projection=clock_velocity_projection,
    )
    field_stress_without_lapse = canonical_scalar_stress_moments(
        field_prime,
        delta_field,
        delta_field_prime,
        scale_factor,
        0.0j,
        wavenumber_GeV,
        potential_derivative_GeV3=potential_prime,
    )
    clock_stress_without_lapse = canonical_scalar_stress_moments(
        theta_prime,
        delta_theta,
        delta_theta_prime,
        scale_factor,
        0.0j,
        wavenumber_GeV,
        kinetic_normalization=parameters.Z_theta,
    )
    stress_without_lapse = sum_scalar_stress_moments(
        dust_stress,
        field_stress_without_lapse,
        clock_stress_without_lapse,
    )

    scalar_kinetic_background = (
        field_prime**2 + parameters.Z_theta * theta_prime**2
    )
    constraint_denominator = wavenumber_GeV**2 - scalar_kinetic_background / (
        2.0 * parameters.Mpl_GeV**2
    )
    if not np.isfinite(constraint_denominator) or constraint_denominator == 0.0:
        raise ValueError("The algebraic Einstein constraint is singular.")
    momentum_constraint = (
        1j
        * scale_factor**2
        * stress_without_lapse["longitudinal_flux_GeV4"]
        / (2.0 * parameters.Mpl_GeV**2 * wavenumber_GeV)
    )
    phi_from_closed_constraint = (
        -scale_factor**2
        * stress_without_lapse["delta_rho_GeV4"]
        / (2.0 * parameters.Mpl_GeV**2)
        - 3.0 * conformal_hubble_GeV * momentum_constraint
    ) / constraint_denominator

    field_stress = canonical_scalar_stress_moments(
        field_prime,
        delta_field,
        delta_field_prime,
        scale_factor,
        phi_from_closed_constraint,
        wavenumber_GeV,
        potential_derivative_GeV3=potential_prime,
    )
    clock_stress = canonical_scalar_stress_moments(
        theta_prime,
        delta_theta,
        delta_theta_prime,
        scale_factor,
        phi_from_closed_constraint,
        wavenumber_GeV,
        kinetic_normalization=parameters.Z_theta,
    )
    total_stress = sum_scalar_stress_moments(
        dust_stress, field_stress, clock_stress
    )
    metric = newtonian_gauge_metric_constraints(
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        parameters.Mpl_GeV,
        total_stress["delta_rho_GeV4"],
        total_stress["longitudinal_flux_GeV4"],
        total_stress["longitudinal_anisotropic_stress_GeV4"],
    )

    scalar_source = coupled_pressureless_matter_scalar_source_perturbation(
        bare_density, delta_m, parameters
    )
    dust_density_prime, dust_velocity_prime = coupled_pressureless_matter_scalar_rhs(
        delta_m,
        velocity_m,
        wavenumber_GeV,
        conformal_hubble_GeV,
        metric["psi"],
        metric["phi_prime_GeV"],
        field,
        field_prime,
        delta_field,
        parameters,
        clock_velocity_projection=clock_velocity_projection,
        clock_velocity_prime_GeV=clock_velocity_prime_GeV,
    )
    field_prime_rhs, field_second_prime_rhs = scalar_field_perturbation_rhs(
        delta_field,
        delta_field_prime,
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        potential_second,
        field_prime,
        background["N_second_prime_GeV3"],
        metric["psi"],
        metric["phi_prime_GeV"],
        metric["phi_prime_GeV"],
        scalar_source,
    )
    clock_current_divergence = clock_interaction_current_divergence_perturbation(
        wavenumber_GeV,
        parameters.h_GeV_inv,
        field,
        theta_prime,
        bare_density,
        bare_density * velocity_m,
        clock_velocity_projection,
    )
    theta_prime_rhs, theta_second_prime_rhs = clock_field_perturbation_rhs(
        delta_theta,
        delta_theta_prime,
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        theta_prime,
        background["Theta_second_prime_GeV3"],
        parameters.Z_theta,
        metric["psi"],
        metric["phi_prime_GeV"],
        metric["phi_prime_GeV"],
        clock_current_divergence,
    )
    return {
        "metric": metric,
        "stress_moments": total_stress,
        "component_stress_moments": {
            "dust": dust_stress,
            "N": field_stress,
            "Theta": clock_stress,
        },
        "scalar_source_perturbation_GeV3": scalar_source,
        "clock_current_divergence_perturbation_GeV3": clock_current_divergence,
        "rhs": {
            "dust": {
                "delta_m_prime_GeV": dust_density_prime,
                "velocity_m_prime_GeV": dust_velocity_prime,
            },
            "N": {
                "delta_N_prime_GeV2": field_prime_rhs,
                "delta_N_second_prime_GeV3": field_second_prime_rhs,
            },
            "Theta": {
                "delta_Theta_prime_GeV2": theta_prime_rhs,
                "delta_Theta_second_prime_GeV3": theta_second_prime_rhs,
            },
        },
    }


def assemble_h_zero_dust_scalar_mode(
    wavenumber_GeV,
    conformal_hubble_GeV,
    scale_factor,
    parameters,
    background,
    perturbations,
):
    """Assemble the h=0 dust-scalar mode, preserving its restricted contract."""
    if parameters.h_GeV_inv != 0.0:
        raise ValueError("This closed perturbation slice requires h_GeV_inv=0.")
    return assemble_coupled_dust_scalar_mode(
        wavenumber_GeV,
        conformal_hubble_GeV,
        scale_factor,
        parameters,
        background,
        perturbations,
    )