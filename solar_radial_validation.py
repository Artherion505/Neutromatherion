import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson, quad, simpson, solve_bvp, solve_ivp
from scipy.interpolate import PchipInterpolator

from solar_thinshell import (
    M_sun,
    Mpl_GeV,
    R_sun,
    find_N0,
    m_to_GeVinv,
    rho_kgm3_to_GeV4,
    rho_out,
)

AU_METERS = 149_597_870_700.0
MAX_EXTERIOR_NONLINEARITY_RATIO = 1e-2
MAX_EXTERIOR_MATCH_RADIUS_RATIO = 1e10
PROFILE_COLUMNS = ("r_over_R_sun", "rho_kgm3")
DEFAULT_SSM_PROFILE = (
    Path(__file__).resolve().parent
    / "data"
    / "solar_density_profile_aag21.csv"
)
DEFAULT_SSM_PROFILE_NAME = (
    "B23/SF-III AAG21 (Herrera & Serenelli 2023, v1.2; "
    "DOI: 10.5281/zenodo.10822316)"
)


def validate_density_profile(r_over_r_sun, rho_kgm3):
    radii = np.asarray(r_over_r_sun, dtype=float)
    densities = np.asarray(rho_kgm3, dtype=float)
    if (
        radii.ndim != 1
        or densities.ndim != 1
        or radii.size != densities.size
        or radii.size < 3
        or np.any(~np.isfinite(radii))
        or np.any(~np.isfinite(densities))
        or np.any(densities < 0.0)
        or np.any(np.diff(radii) <= 0.0)
    ):
        raise ValueError("El perfil debe contener al menos 3 filas válidas y radios crecientes.")
    if not np.isclose(radii[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("El perfil debe comenzar en r/R_sun=0.")
    if not np.isclose(radii[-1], 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("El perfil debe terminar en r/R_sun=1.")
    radii = radii.copy()
    radii[0] = 0.0
    radii[-1] = 1.0
    if densities[0] <= 0.0:
        raise ValueError("La densidad central del perfil debe ser positiva.")
    return radii, densities


def load_density_profile(profile_path):
    profile_path = Path(profile_path)
    with profile_path.open("r", newline="", encoding="utf-8") as profile_file:
        profile_rows = (
            line
            for line in profile_file
            if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(profile_rows)
        missing = set(PROFILE_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Faltan columnas en el perfil solar: "
                f"{', '.join(sorted(missing))}"
            )
        rows = list(reader)
    try:
        radii = [float(row[PROFILE_COLUMNS[0]]) for row in rows]
        densities = [float(row[PROFILE_COLUMNS[1]]) for row in rows]
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("El perfil contiene valores no numéricos.") from error
    return validate_density_profile(radii, densities)


def solar_n3_polytrope_profile(samples=501):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("samples debe ser un entero de al menos 100.")

    xi_start = 1e-6

    def lane_emden_rhs(xi, state):
        theta, theta_prime = state
        return theta_prime, -2.0 * theta_prime / xi - max(theta, 0.0) ** 3

    def surface_event(xi, state):
        return state[0]

    surface_event.terminal = True
    surface_event.direction = -1
    initial_state = (
        1.0 - xi_start**2 / 6.0 + xi_start**4 / 40.0,
        -xi_start / 3.0 + xi_start**3 / 10.0,
    )
    lane_emden = solve_ivp(
        lane_emden_rhs,
        (xi_start, 10.0),
        initial_state,
        method="DOP853",
        events=surface_event,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.05,
    )
    if not lane_emden.success or lane_emden.t_events[0].size == 0:
        raise RuntimeError("No se pudo construir el polytropo de Lane-Emden n=3.")

    xi_surface = lane_emden.t_events[0][0]
    theta_prime_surface = lane_emden.y_events[0][0][1]
    dimensionless_mass = -xi_surface**2 * theta_prime_surface
    mean_density = 3.0 * M_sun / (4.0 * math.pi * R_sun**3)
    central_density = (
        mean_density * xi_surface**3 / (3.0 * dimensionless_mass)
    )

    interior_samples = max(50, samples // 3)
    surface_samples = samples - interior_samples
    interior_radii = np.linspace(0.0, 0.9, interior_samples)
    surface_distances = np.geomspace(0.1, 1e-12, surface_samples)
    radii = np.unique(
        np.concatenate(
            (interior_radii, 1.0 - surface_distances, [1.0])
        )
    )
    theta = np.ones_like(radii)
    theta[1:] = np.maximum(
        lane_emden.sol(radii[1:] * xi_surface)[0], 0.0
    )
    theta[-1] = 0.0
    densities = central_density * theta**3
    return validate_density_profile(radii, densities)


def _effective_mass(field, m0_GeV, lambda_value):
    return math.sqrt(max(0.0, m0_GeV**2 + 3.0 * lambda_value * field**2))


def _exterior_nonlinearity_ratio(
    surface_excess_GeV,
    outside_field_GeV,
    outside_mass_GeV,
    lambda_value,
):
    nonlinear_scale = abs(lambda_value) * (
        3.0 * abs(outside_field_GeV * surface_excess_GeV)
        + surface_excess_GeV**2
    )
    linear_scale = outside_mass_GeV**2
    if linear_scale == 0.0:
        return 0.0 if nonlinear_scale == 0.0 else math.inf
    return nonlinear_scale / linear_scale


def _force_ratio_from_gradient(
    radius_m,
    scalar_gradient_GeV2,
    effective_coupling_GeVinv,
    source_mass_GeV,
):
    radius_GeVinv = radius_m * m_to_GeVinv
    if source_mass_GeV <= 0.0:
        return 0.0
    newtonian_acceleration_GeV = (
        source_mass_GeV
        / (8.0 * math.pi * Mpl_GeV**2 * radius_GeVinv**2)
    )
    scalar_acceleration_GeV = abs(
        effective_coupling_GeVinv * scalar_gradient_GeV2
    )
    return scalar_acceleration_GeV / newtonian_acceleration_GeV


def finite_range_scalar_force_ratio(
    source_solution,
    probe_solution,
    center_separation_m,
):
    """Return signed scalar/Newton force for two isolated screened bodies.

    Positive values are attractive. This linear-tail result requires a shared
    ambient solution and non-overlapping matched exterior regions.
    """
    if (
        np.ndim(center_separation_m) != 0
        or not np.isfinite(center_separation_m)
        or center_separation_m <= 0.0
    ):
        raise ValueError("La separación entre centros debe ser finita y positiva.")

    source_radius_m = source_solution["body_radius_m"]
    probe_radius_m = probe_solution["body_radius_m"]
    if center_separation_m <= source_radius_m + probe_radius_m:
        raise ValueError("Los cuerpos no deben solaparse.")

    source_mass = source_solution["outside_effective_mass_GeV"]
    probe_mass = probe_solution["outside_effective_mass_GeV"]
    source_field = source_solution["outside_field_GeV"]
    probe_field = probe_solution["outside_field_GeV"]
    if not math.isclose(source_mass, probe_mass, rel_tol=1e-10, abs_tol=0.0):
        raise ValueError("Las soluciones deben compartir la masa escalar ambiental.")
    if not math.isclose(source_field, probe_field, rel_tol=1e-10, abs_tol=1e-300):
        raise ValueError("Las soluciones deben compartir el campo ambiental.")
    if not source_solution["linear_exterior_valid"] or not probe_solution[
        "linear_exterior_valid"
    ]:
        raise ValueError(
            "La fuerza entre cuerpos requiere colas exteriores lineales; "
            "este perfil necesita un solver exterior no lineal."
        )
    source_match_radius_m = source_solution["exterior_match_radius_m"]
    probe_match_radius_m = probe_solution["exterior_match_radius_m"]
    if source_match_radius_m + probe_match_radius_m >= center_separation_m:
        raise ValueError(
            "Las regiones exteriores no lineales se solapan a esta "
            "separación; se requiere un solver conjunto de dos cuerpos."
        )

    beta_source = source_solution["beta_scalar_match"]
    beta_probe = probe_solution["beta_scalar_match"]
    if not np.isfinite(beta_source) or not np.isfinite(beta_probe):
        raise ValueError("Las cargas escalares de ambos cuerpos deben ser finitas.")

    separation_GeVinv = center_separation_m * m_to_GeVinv
    match_gap_GeVinv = (
        center_separation_m
        - source_match_radius_m
        - probe_match_radius_m
    ) * m_to_GeVinv
    range_coordinate = source_mass * match_gap_GeVinv
    if range_coordinate > 745.0:
        return 0.0
    return (
        2.0
        * beta_source
        * beta_probe
        * (1.0 + source_mass * separation_GeVinv)
        * math.exp(-range_coordinate)
    )


def _scalar_potential_GeV4(field_GeV, m0_GeV, lambda_value):
    return (
        0.5 * m0_GeV**2 * field_GeV**2
        + 0.25 * lambda_value * field_GeV**4
    )


def _exterior_metric_sources(solution, radius_GeVinv):
    match_radius_GeVinv = solution["exterior_match_radius_GeVinv"]
    outside_mass_GeV = solution["outside_effective_mass_GeV"]
    match_excess_GeV = (
        solution["exterior_match_field_GeV"]
        - solution["outside_field_GeV"]
    )
    if match_excess_GeV == 0.0:
        return 0.0, 0.0

    range_coordinate = outside_mass_GeV * (
        radius_GeVinv - match_radius_GeVinv
    )
    if range_coordinate > 745.0:
        return 0.0, 0.0
    radial_factor = (
        match_radius_GeVinv
        / radius_GeVinv
        * math.exp(-range_coordinate)
    )
    field_excess_GeV = match_excess_GeV * radial_factor
    gradient_GeV2 = -field_excess_GeV * (
        outside_mass_GeV + 1.0 / radius_GeVinv
    )
    potential_excess_GeV4 = (
        0.5 * outside_mass_GeV**2 * field_excess_GeV**2
    )
    scalar_density_GeV4 = (
        0.5 * gradient_GeV2**2 + potential_excess_GeV4
    )
    radial_pressure_GeV4 = (
        0.5 * gradient_GeV2**2 - potential_excess_GeV4
    )
    return scalar_density_GeV4, radial_pressure_GeV4


def _integrate_exterior_metric_source(solution, lower_radius_GeVinv, moment):
    match_radius_GeVinv = solution["exterior_match_radius_GeVinv"]
    outside_mass_GeV = solution["outside_effective_mass_GeV"]
    if lower_radius_GeVinv < match_radius_GeVinv:
        raise ValueError("La integral exterior debe comenzar en el radio de empalme.")
    if (
        solution["exterior_match_field_GeV"]
        == solution["outside_field_GeV"]
    ):
        return 0.0

    def weighted_source(radius_GeVinv):
        density_GeV4, pressure_GeV4 = _exterior_metric_sources(
            solution, radius_GeVinv
        )
        if moment == "mass":
            return 4.0 * math.pi * radius_GeVinv**2 * density_GeV4
        if moment == "density":
            return 4.0 * math.pi * radius_GeVinv * density_GeV4
        return 4.0 * math.pi * radius_GeVinv * pressure_GeV4

    if outside_mass_GeV > 0.0:
        lower_coordinate = outside_mass_GeV * (
            lower_radius_GeVinv - match_radius_GeVinv
        )
        if lower_coordinate > 745.0:
            return 0.0

        if outside_mass_GeV * match_radius_GeVinv > 1.0:
            def scaled_integrand(coordinate):
                if coordinate > 745.0:
                    return 0.0
                radius_GeVinv = (
                    match_radius_GeVinv + coordinate / outside_mass_GeV
                )
                return weighted_source(radius_GeVinv) / outside_mass_GeV

            return quad(
                scaled_integrand,
                lower_coordinate,
                math.inf,
                epsabs=0.0,
                epsrel=1e-8,
                limit=250,
            )[0]

    lower_log_radius = math.log(lower_radius_GeVinv / match_radius_GeVinv)

    def logarithmic_integrand(log_radius):
        if log_radius > 700.0:
            return 0.0
        radius_GeVinv = match_radius_GeVinv * math.exp(log_radius)
        if outside_mass_GeV * (radius_GeVinv - match_radius_GeVinv) > 745.0:
            return 0.0
        return weighted_source(radius_GeVinv) * radius_GeVinv

    return quad(
        logarithmic_integrand,
        lower_log_radius,
        math.inf,
        epsabs=0.0,
        epsrel=1e-8,
        limit=250,
    )[0]


def weak_field_metric_at_radius(solution, radius_m):
    """Evaluate the static weak-field metric for the solved scalar profile.

    This postprocesses the flat-space scalar profile with aligned pressureless
    matter and canonical scalar stress. It omits clock stress, unclosed metric
    variations of the interaction, and the modified photon propagation cone.
    """
    if (
        np.ndim(radius_m) != 0
        or not np.isfinite(radius_m)
        or radius_m < 0.0
    ):
        raise ValueError("El radio métrico debe ser un escalar finito no negativo.")

    radius_GeVinv = radius_m * m_to_GeVinv
    profile_radius_GeVinv = np.asarray(
        solution["profile_radius_GeVinv"], dtype=float
    )
    profile_density_GeV4 = np.asarray(
        solution.get(
            "density_GeV4",
            rho_kgm3_to_GeV4(solution["density_kgm3"]),
        ),
        dtype=float,
    )
    outside_field_GeV = solution["outside_field_GeV"]
    outside_density_GeV4 = solution["outside_density_GeV4"]
    effective_coupling = solution["effective_coupling_GeVinv"]
    profile_field_GeV = np.asarray(solution["field_GeV"], dtype=float)
    profile_gradient_GeV2 = np.asarray(
        solution["field_gradient_GeV2"], dtype=float
    )
    profile_mass_factor = 1.0 + effective_coupling * profile_field_GeV
    outside_mass_factor = 1.0 + effective_coupling * outside_field_GeV
    if np.any(profile_mass_factor <= 0.0) or outside_mass_factor <= 0.0:
        raise ValueError("La masa efectiva del polvo debe permanecer positiva.")

    matter_density_excess_GeV4 = (
        profile_mass_factor * profile_density_GeV4
        - outside_mass_factor * outside_density_GeV4
    )
    potential_excess_GeV4 = np.asarray(
        [
            _scalar_potential_GeV4(
                field,
                solution["m0_GeV"],
                solution["lambda_value"],
            )
            - _scalar_potential_GeV4(
                outside_field_GeV,
                solution["m0_GeV"],
                solution["lambda_value"],
            )
            for field in profile_field_GeV
        ]
    )
    scalar_density_GeV4 = (
        0.5 * profile_gradient_GeV2**2 + potential_excess_GeV4
    )
    total_density_excess_GeV4 = (
        matter_density_excess_GeV4 + scalar_density_GeV4
    )
    radial_pressure_GeV4 = (
        0.5 * profile_gradient_GeV2**2 - potential_excess_GeV4
    )

    enclosed_mass_GeV = 4.0 * math.pi * cumulative_simpson(
        total_density_excess_GeV4 * profile_radius_GeVinv**2,
        x=profile_radius_GeVinv,
        initial=0.0,
    )
    density_interpolator = PchipInterpolator(
        profile_radius_GeVinv, total_density_excess_GeV4
    )
    pressure_interpolator = PchipInterpolator(
        profile_radius_GeVinv, radial_pressure_GeV4
    )
    profile_match_radius_GeVinv = profile_radius_GeVinv[-1]

    def integrate_profile_tail(interpolator, lower_radius_GeVinv):
        if lower_radius_GeVinv >= profile_match_radius_GeVinv:
            return 0.0
        interior_radii = profile_radius_GeVinv[
            (profile_radius_GeVinv > lower_radius_GeVinv)
            & (profile_radius_GeVinv < profile_match_radius_GeVinv)
        ]
        integration_radii = np.concatenate(
            (
                [lower_radius_GeVinv],
                interior_radii,
                [profile_match_radius_GeVinv],
            )
        )
        return 4.0 * math.pi * simpson(
            integration_radii * interpolator(integration_radii),
            x=integration_radii,
        )

    if radius_GeVinv <= profile_match_radius_GeVinv:
        mass_at_radius_GeV = float(
            np.interp(
                radius_GeVinv,
                profile_radius_GeVinv,
                enclosed_mass_GeV,
            )
        )
        density_tail_GeV2 = integrate_profile_tail(
            density_interpolator, radius_GeVinv
        )
        pressure_tail_GeV2 = integrate_profile_tail(
            pressure_interpolator, radius_GeVinv
        )
        density_tail_GeV2 += _integrate_exterior_metric_source(
            solution, profile_match_radius_GeVinv, "density"
        )
        pressure_tail_GeV2 += _integrate_exterior_metric_source(
            solution, profile_match_radius_GeVinv, "pressure"
        )
    else:
        exterior_mass_total_GeV = _integrate_exterior_metric_source(
            solution, profile_match_radius_GeVinv, "mass"
        )
        exterior_mass_tail_GeV = _integrate_exterior_metric_source(
            solution, radius_GeVinv, "mass"
        )
        mass_at_radius_GeV = (
            float(enclosed_mass_GeV[-1])
            + exterior_mass_total_GeV
            - exterior_mass_tail_GeV
        )
        density_tail_GeV2 = _integrate_exterior_metric_source(
            solution, radius_GeVinv, "density"
        )
        pressure_tail_GeV2 = _integrate_exterior_metric_source(
            solution, radius_GeVinv, "pressure"
        )

    gravity_GeVinv2 = 1.0 / (8.0 * math.pi * Mpl_GeV**2)
    enclosed_mass_over_radius = (
        mass_at_radius_GeV / radius_GeVinv
        if radius_GeVinv > 0.0
        else 0.0
    )
    phi = -gravity_GeVinv2 * (
        enclosed_mass_over_radius + density_tail_GeV2
    )
    psi = -gravity_GeVinv2 * (
        enclosed_mass_over_radius + density_tail_GeV2 + pressure_tail_GeV2
    )
    radial_metric = (
        1.0 + 2.0 * gravity_GeVinv2 * enclosed_mass_over_radius
        if radius_GeVinv > 0.0
        else 1.0
    )
    return {
        "radius_m": radius_m,
        "Phi": phi,
        "Psi": psi,
        "g_tt": -(1.0 + 2.0 * psi),
        "g_rr_areal": radial_metric,
        "enclosed_mass_GeV": mass_at_radius_GeV,
        "exterior_nonlinearity_ratio": solution[
            "exterior_nonlinearity_ratio"
        ],
        "linear_exterior_valid": solution["linear_exterior_valid"],
    }


def solve_radial_profile(
    g_GeVinv,
    lambda_value,
    m0_eV,
    r_over_r_sun,
    rho_kgm3,
    kappa_over_lambda=0.0,
    rho_out_kgm3=rho_out,
    observation_radius_m=AU_METERS,
    profile_name="tabulated solar profile",
    tolerance=1e-7,
    max_nodes=30000,
    body_radius_m=R_sun,
):
    parameters = (
        g_GeVinv,
        lambda_value,
        m0_eV,
        kappa_over_lambda,
        rho_out_kgm3,
        observation_radius_m,
        tolerance,
        body_radius_m,
    )
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("Los parámetros deben ser finitos.")
    if lambda_value < 0.0 or m0_eV < 0.0 or rho_out_kgm3 < 0.0:
        raise ValueError("lambda, m0 y rho exterior deben ser no negativos.")
    if body_radius_m <= 0.0:
        raise ValueError("El radio del cuerpo debe ser positivo.")
    if observation_radius_m < body_radius_m:
        raise ValueError("El radio de observación debe ser exterior al cuerpo.")
    if tolerance <= 0.0:
        raise ValueError("tolerance debe ser positiva.")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 100:
        raise ValueError("max_nodes debe ser un entero de al menos 100.")

    radii, densities_kgm3 = validate_density_profile(r_over_r_sun, rho_kgm3)
    densities_GeV4 = np.asarray(
        rho_kgm3_to_GeV4(densities_kgm3), dtype=float
    )
    m0_GeV = m0_eV * 1e-9
    effective_coupling = g_GeVinv - kappa_over_lambda
    outside_density_GeV4 = rho_kgm3_to_GeV4(rho_out_kgm3)
    center_field_GeV = find_N0(
        densities_GeV4[0],
        g_GeVinv,
        lambda_value,
        m0_GeV,
        kappa_over_lambda,
    )
    outside_field_GeV = find_N0(
        outside_density_GeV4,
        g_GeVinv,
        lambda_value,
        m0_GeV,
        kappa_over_lambda,
    )
    center_mass_GeV = _effective_mass(
        center_field_GeV, m0_GeV, lambda_value
    )
    outside_mass_GeV = _effective_mass(
        outside_field_GeV, m0_GeV, lambda_value
    )

    body_radius_GeVinv = body_radius_m * m_to_GeVinv
    mass_scale_GeV = max(
        center_mass_GeV,
        outside_mass_GeV,
        m0_GeV,
        1.0 / body_radius_GeVinv,
    )
    surface_coordinate = mass_scale_GeV * body_radius_GeVinv
    field_scale_GeV = max(abs(center_field_GeV), abs(outside_field_GeV))
    if field_scale_GeV == 0.0:
        field_scale_GeV = 1.0
    outside_field_scaled = outside_field_GeV / field_scale_GeV
    center_field_scaled = center_field_GeV / field_scale_GeV
    density_interpolator = PchipInterpolator(radii, densities_GeV4)

    def source_scaled(field_scaled, density_GeV4):
        field_GeV = field_scale_GeV * field_scaled
        return (
            m0_GeV**2 * field_GeV
            + lambda_value * field_GeV**3
            + effective_coupling * density_GeV4
        ) / (mass_scale_GeV**2 * field_scale_GeV)

    coordinate_transition = 2.0 * surface_coordinate

    def solver_coordinate_from_radius(radius_coordinate):
        radius_array = np.asarray(radius_coordinate, dtype=float)
        solver_coordinate = np.where(
            radius_array <= coordinate_transition,
            radius_array,
            coordinate_transition
            + coordinate_transition
            * np.log(radius_array / coordinate_transition),
        )
        return (
            float(solver_coordinate)
            if solver_coordinate.ndim == 0
            else solver_coordinate
        )

    def radius_from_solver_coordinate(solver_coordinate):
        solver_array = np.asarray(solver_coordinate, dtype=float)
        radius_coordinate = np.where(
            solver_array <= coordinate_transition,
            solver_array,
            coordinate_transition
            * np.exp(
                (solver_array - coordinate_transition)
                / coordinate_transition
            ),
        )
        return (
            float(radius_coordinate)
            if radius_coordinate.ndim == 0
            else radius_coordinate
        )

    def surface_differential_equations(coordinate, state, _parameters):
        radius_fraction = np.clip(coordinate / surface_coordinate, 0.0, 1.0)
        local_density = density_interpolator(radius_fraction)
        source = source_scaled(state[0], local_density)
        return np.vstack((state[1], source - 2.0 * state[1] / coordinate))

    def differential_equations(solver_coordinate, state, _parameters):
        coordinate = radius_from_solver_coordinate(solver_coordinate)
        coordinate_scale = np.where(
            solver_coordinate <= coordinate_transition,
            1.0,
            coordinate / coordinate_transition,
        )
        radius_fraction = np.clip(coordinate / surface_coordinate, 0.0, 1.0)
        local_density = np.where(
            coordinate <= surface_coordinate,
            density_interpolator(radius_fraction),
            outside_density_GeV4,
        )
        source = source_scaled(state[0], local_density)
        return np.vstack(
            (
                coordinate_scale * state[1],
                coordinate_scale * (source - 2.0 * state[1] / coordinate),
            )
        )

    center_coordinate = min(1e-4, surface_coordinate * 1e-8)
    center_offset = max(
        1e-7,
        1e-6 * surface_coordinate,
        16.0 * np.spacing(surface_coordinate),
    )
    largest_offset = min(
        0.06 * surface_coordinate,
        surface_coordinate - center_coordinate,
    )
    if largest_offset <= center_offset:
        raise ValueError("La escala radial no permite construir la malla del solver.")

    interior_mesh = np.linspace(
        center_coordinate,
        surface_coordinate - largest_offset,
        300,
    )
    surface_offsets = np.geomspace(largest_offset, center_offset, 800)
    surface_mesh = surface_coordinate - surface_offsets
    interior_mesh = np.unique(
        np.concatenate((interior_mesh, surface_mesh, [surface_coordinate]))
    )

    equilibrium_fields = np.asarray(
        [
            find_N0(
                density,
                g_GeVinv,
                lambda_value,
                m0_GeV,
                kappa_over_lambda,
            )
            for density in densities_GeV4
        ]
    )

    initial_field = np.interp(
        interior_mesh / surface_coordinate,
        radii,
        equilibrium_fields,
    ) / field_scale_GeV
    initial_derivative = np.gradient(initial_field, interior_mesh)
    initial_derivative[0] = (
        source_scaled(center_field_scaled, densities_GeV4[0])
        * center_coordinate
        / 3.0
    )
    initial_state = np.vstack((initial_field, initial_derivative))
    surface_robin = (
        outside_mass_GeV / mass_scale_GeV + 1.0 / surface_coordinate
    )

    def surface_boundary_conditions(
        inner_state, surface_state, unknown_parameters
    ):
        central_field = unknown_parameters[0]
        central_source = source_scaled(central_field, densities_GeV4[0])
        return np.array(
            [
                inner_state[0]
                - central_field
                - central_source * center_coordinate**2 / 6.0,
                inner_state[1] - central_source * center_coordinate / 3.0,
                surface_state[1]
                + surface_robin
                * (surface_state[0] - outside_field_scaled),
            ]
        )

    surface_solution = solve_bvp(
        surface_differential_equations,
        surface_boundary_conditions,
        interior_mesh,
        initial_state,
        p=np.array([center_field_scaled]),
        tol=tolerance,
        max_nodes=max_nodes,
    )
    if not surface_solution.success:
        raise RuntimeError(
            "El solver radial no convergió: " + surface_solution.message
        )
    surface_state = surface_solution.sol(surface_coordinate)
    surface_field_GeV = surface_state[0] * field_scale_GeV
    surface_excess_GeV = surface_field_GeV - outside_field_GeV
    surface_exterior_nonlinearity_ratio = _exterior_nonlinearity_ratio(
        surface_excess_GeV,
        outside_field_GeV,
        outside_mass_GeV,
        lambda_value,
    )
    surface_linear_exterior_valid = (
        surface_exterior_nonlinearity_ratio
        <= MAX_EXTERIOR_NONLINEARITY_RATIO
    )

    def mesh_for_match(match_radius_ratio):
        if match_radius_ratio == 1.0:
            return interior_mesh
        exterior_span = surface_coordinate * (match_radius_ratio - 1.0)
        if exterior_span > center_offset:
            exterior_offsets = np.geomspace(
                center_offset, exterior_span, 1200
            )
        else:
            exterior_offsets = np.array([exterior_span])
        mesh_parts = [
            interior_mesh,
            surface_coordinate + exterior_offsets,
        ]
        if match_radius_ratio > 2.0:
            mesh_parts.append(np.array([coordinate_transition]))
        return np.unique(
            np.concatenate(mesh_parts)
        )

    def initial_state_for(
        mesh,
        previous_solution,
        previous_match_ratio,
        surface_solution_for_seed,
    ):
        solver_mesh = solver_coordinate_from_radius(mesh)
        initial_state = np.empty((2, mesh.size), dtype=float)
        if previous_solution is not None:
            previous_match_coordinate = (
                surface_coordinate * previous_match_ratio
            )
            inherited = mesh <= previous_match_coordinate
            initial_state[:, inherited] = previous_solution.sol(
                solver_mesh[inherited]
            )
            extended = ~inherited
            if np.any(extended):
                previous_state = previous_solution.sol(
                    solver_coordinate_from_radius(previous_match_coordinate)
                )
                previous_excess = previous_state[0] - outside_field_scaled
                attenuation_exponent = -(
                    outside_mass_GeV / mass_scale_GeV
                ) * (mesh[extended] - previous_match_coordinate)
                attenuation = np.exp(
                    np.maximum(attenuation_exponent, -745.0)
                )
                attenuation[attenuation_exponent < -745.0] = 0.0
                excess = (
                    previous_excess
                    * previous_match_coordinate
                    / mesh[extended]
                    * attenuation
                )
                initial_state[0, extended] = outside_field_scaled + excess
                initial_state[1, extended] = -(
                    outside_mass_GeV / mass_scale_GeV
                    + 1.0 / mesh[extended]
                ) * excess
        else:
            interior = mesh <= surface_coordinate
            if surface_solution_for_seed is None:
                initial_state[0, interior] = np.interp(
                    mesh[interior] / surface_coordinate,
                    radii,
                    equilibrium_fields,
                ) / field_scale_GeV
            else:
                surface_guess = surface_solution_for_seed.sol(mesh[interior])
                initial_state[0, interior] = surface_guess[0]
                initial_state[1, interior] = surface_guess[1]
            extended = ~interior
            if np.any(extended):
                surface_excess = (
                    initial_state[0, np.flatnonzero(interior)[-1]]
                    - outside_field_scaled
                )
                attenuation_exponent = -(
                    outside_mass_GeV / mass_scale_GeV
                ) * (mesh[extended] - surface_coordinate)
                attenuation = np.exp(
                    np.maximum(attenuation_exponent, -745.0)
                )
                attenuation[attenuation_exponent < -745.0] = 0.0
                excess = (
                    surface_excess
                    * surface_coordinate
                    / mesh[extended]
                    * attenuation
                )
                initial_state[0, extended] = outside_field_scaled + excess
                initial_state[1, extended] = -(
                    outside_mass_GeV / mass_scale_GeV
                    + 1.0 / mesh[extended]
                ) * excess
            if surface_solution_for_seed is None:
                initial_state[1, interior] = np.gradient(
                    initial_state[0, interior], mesh[interior]
                )
            if np.any(extended):
                initial_state[1, extended] = -(
                    outside_mass_GeV / mass_scale_GeV
                    + 1.0 / mesh[extended]
                ) * (initial_state[0, extended] - outside_field_scaled)
        center_source = source_scaled(
            center_field_scaled, densities_GeV4[0]
        )
        initial_state[1, 0] = center_source * mesh[0] / 3.0
        return initial_state

    def solve_at_match(
        match_radius_ratio,
        previous_solution=None,
        previous_match_ratio=1.0,
        surface_solution_for_seed=None,
    ):
        mesh = mesh_for_match(match_radius_ratio)

        def boundary_conditions(inner_state, outer_state, unknown_parameters):
            central_field = unknown_parameters[0]
            central_source = source_scaled(central_field, densities_GeV4[0])
            inner_coordinate = mesh[0]
            outer_coordinate = mesh[-1]
            return np.array(
                [
                    inner_state[0]
                    - central_field
                    - central_source * inner_coordinate**2 / 6.0,
                    inner_state[1]
                    - central_source * inner_coordinate / 3.0,
                    outer_state[1]
                    + (
                        outside_mass_GeV
                        / mass_scale_GeV
                        + 1.0 / outer_coordinate
                    )
                    * (outer_state[0] - outside_field_scaled),
                ]
            )

        solution = solve_bvp(
            differential_equations,
            boundary_conditions,
            solver_coordinate_from_radius(mesh),
            initial_state_for(
                mesh,
                previous_solution,
                previous_match_ratio,
                surface_solution_for_seed,
            ),
            p=np.array([center_field_scaled]),
            tol=max(tolerance, 1e-5),
            max_nodes=max_nodes,
        )
        if not solution.success:
            raise RuntimeError(
                "El solver radial no convergió: " + solution.message
            )
        return solution

    if surface_linear_exterior_valid:
        solution = surface_solution
        solution_in_hybrid_coordinate = False
        match_radius_ratio = 1.0
        match_coordinate = surface_coordinate
        match_state = surface_state
        match_field_GeV = surface_field_GeV
        match_excess_GeV = surface_excess_GeV
        exterior_nonlinearity_ratio = surface_exterior_nonlinearity_ratio
        linear_exterior_valid = True
    else:
        if outside_mass_GeV == 0.0:
            raise RuntimeError(
                "El exterior no alcanza un régimen lineal de alcance finito."
            )
        match_radius_ratio = max(
            2.0,
            10.0 / (outside_mass_GeV * body_radius_GeVinv),
        )
        if match_radius_ratio > MAX_EXTERIOR_MATCH_RADIUS_RATIO:
            raise RuntimeError(
                "No se alcanzó una cola exterior lineal antes del radio "
                "máximo de empalme."
            )
        previous_solution = None
        previous_match_radius_ratio = 1.0
        for _ in range(20):
            solution = solve_at_match(
                match_radius_ratio,
                previous_solution,
                previous_match_radius_ratio,
                surface_solution,
            )
            match_coordinate = surface_coordinate * match_radius_ratio
            match_state = solution.sol(
                solver_coordinate_from_radius(match_coordinate)
            )
            match_field_GeV = match_state[0] * field_scale_GeV
            match_excess_GeV = match_field_GeV - outside_field_GeV
            exterior_nonlinearity_ratio = _exterior_nonlinearity_ratio(
                match_excess_GeV,
                outside_field_GeV,
                outside_mass_GeV,
                lambda_value,
            )
            if (
                exterior_nonlinearity_ratio
                <= MAX_EXTERIOR_NONLINEARITY_RATIO
            ):
                break
            previous_solution = solution
            previous_match_radius_ratio = match_radius_ratio
            match_radius_ratio *= 2.0
            if match_radius_ratio > MAX_EXTERIOR_MATCH_RADIUS_RATIO:
                raise RuntimeError(
                    "No se alcanzó una cola exterior lineal antes del radio "
                    "máximo de empalme."
                )
        else:
            raise RuntimeError(
                "No se alcanzó una cola exterior lineal tras ampliar el dominio."
            )
        solution_in_hybrid_coordinate = True
        linear_exterior_valid = True

    if solution_in_hybrid_coordinate:
        profile_coordinate = radius_from_solver_coordinate(solution.x)
    else:
        profile_coordinate = solution.x
    profile_derivative_scaled = solution.y[1]

    def state_in_radius_coordinate(coordinate):
        solver_coordinate = (
            solver_coordinate_from_radius(coordinate)
            if solution_in_hybrid_coordinate
            else coordinate
        )
        return solution.sol(solver_coordinate)

    profile_radius = profile_coordinate / surface_coordinate
    profile_field = solution.y[0] * field_scale_GeV
    profile_derivative = (
        profile_derivative_scaled
        * field_scale_GeV
        * mass_scale_GeV
    )
    profile_density_GeV4 = np.maximum(
        np.where(
            profile_radius <= 1.0,
            density_interpolator(np.clip(profile_radius, 0.0, 1.0)),
            outside_density_GeV4,
        ),
        0.0,
    )
    center_field_solution = solution.p[0] * field_scale_GeV
    if profile_radius[0] > 0.0:
        profile_radius = np.insert(profile_radius, 0, 0.0)
        profile_field = np.insert(profile_field, 0, center_field_solution)
        profile_derivative = np.insert(profile_derivative, 0, 0.0)
        profile_density_GeV4 = np.insert(
            profile_density_GeV4, 0, densities_GeV4[0]
        )
    profile_density_kgm3 = (
        profile_density_GeV4 / rho_kgm3_to_GeV4(1.0)
    )
    surface_state = state_in_radius_coordinate(surface_coordinate)
    surface_field_GeV = surface_state[0] * field_scale_GeV
    surface_derivative_GeV2 = (
        surface_state[1]
        * field_scale_GeV
        * mass_scale_GeV
    )
    match_state = state_in_radius_coordinate(match_coordinate)
    match_field_GeV = match_state[0] * field_scale_GeV
    match_excess_GeV = match_field_GeV - outside_field_GeV
    match_radius_GeVinv = match_coordinate / mass_scale_GeV
    match_radius_m = match_radius_GeVinv / m_to_GeVinv
    profile_radius_GeVinv = profile_radius * body_radius_GeVinv
    profile_radius_GeVinv[-1] = match_radius_GeVinv
    body_profile = profile_radius <= 1.0
    bare_mass_GeV = 4.0 * math.pi * simpson(
        (profile_density_GeV4[body_profile] - outside_density_GeV4)
        * profile_radius_GeVinv[body_profile] ** 2,
        x=profile_radius_GeVinv[body_profile],
    )
    surface_excess_GeV = surface_field_GeV - outside_field_GeV
    surface_exterior_nonlinearity_ratio = _exterior_nonlinearity_ratio(
        surface_excess_GeV,
        outside_field_GeV,
        outside_mass_GeV,
        lambda_value,
    )
    linear_exterior_valid = (
        exterior_nonlinearity_ratio
        <= MAX_EXTERIOR_NONLINEARITY_RATIO
    )
    beta_scalar_surface = (
        -4.0
        * math.pi
        * Mpl_GeV
        * body_radius_GeVinv
        * surface_excess_GeV
        / bare_mass_GeV
        if bare_mass_GeV > 0.0
        else (0.0 if surface_excess_GeV == 0.0 else float("nan"))
    )
    beta_scalar_match = (
        -4.0
        * math.pi
        * Mpl_GeV
        * match_radius_GeVinv
        * match_excess_GeV
        / bare_mass_GeV
        if bare_mass_GeV > 0.0
        else (0.0 if match_excess_GeV == 0.0 else float("nan"))
    )
    surface_force_ratio = _force_ratio_from_gradient(
        body_radius_m,
        surface_derivative_GeV2,
        effective_coupling,
        bare_mass_GeV,
    )
    observation_coordinate = (
        observation_radius_m * m_to_GeVinv * mass_scale_GeV
    )
    if observation_coordinate <= match_coordinate:
        observation_state = state_in_radius_coordinate(
            observation_coordinate
        )
        observation_gradient_GeV2 = (
            observation_state[1]
            * field_scale_GeV
            * mass_scale_GeV
        )
    else:
        observation_radius_GeVinv = observation_radius_m * m_to_GeVinv
        attenuation_exponent = -outside_mass_GeV * (
            observation_radius_GeVinv - match_radius_GeVinv
        )
        if attenuation_exponent < -745.0:
            observation_gradient_GeV2 = 0.0
        else:
            observation_excess_GeV = (
                match_excess_GeV
                * match_radius_GeVinv
                / observation_radius_GeVinv
                * math.exp(attenuation_exponent)
            )
            observation_gradient_GeV2 = -observation_excess_GeV * (
                outside_mass_GeV + 1.0 / observation_radius_GeVinv
            )
    observation_force_ratio = _force_ratio_from_gradient(
        observation_radius_m,
        observation_gradient_GeV2,
        effective_coupling,
        bare_mass_GeV,
    )
    compton_outside_m = (
        math.inf
        if outside_mass_GeV == 0.0
        else 1.0 / (outside_mass_GeV * m_to_GeVinv)
    )

    result = {
        "profile_name": profile_name,
        "solver_nodes": int(solution.x.size),
        "center_field_GeV": center_field_solution,
        "surface_field_GeV": surface_field_GeV,
        "outside_field_GeV": outside_field_GeV,
        "center_effective_mass_GeV": center_mass_GeV,
        "outside_effective_mass_GeV": outside_mass_GeV,
        "m0_GeV": m0_GeV,
        "lambda_value": lambda_value,
        "effective_coupling_GeVinv": effective_coupling,
        "outside_density_GeV4": outside_density_GeV4,
        "center_mass_times_solar_radius": center_mass_GeV
        * R_sun
        * m_to_GeVinv,
        "center_mass_times_body_radius": center_mass_GeV
        * body_radius_GeVinv,
        "outside_compton_length_m": compton_outside_m,
        "body_radius_m": body_radius_m,
        "exterior_match_radius_m": match_radius_m,
        "exterior_match_radius_GeVinv": match_radius_GeVinv,
        "exterior_match_field_GeV": match_field_GeV,
        "bare_mass_GeV": bare_mass_GeV,
        "beta_scalar_surface": beta_scalar_surface,
        "beta_scalar_match": beta_scalar_match,
        "exterior_nonlinearity_ratio": exterior_nonlinearity_ratio,
        "surface_exterior_nonlinearity_ratio": (
            surface_exterior_nonlinearity_ratio
        ),
        "linear_exterior_valid": linear_exterior_valid,
        "surface_force_over_newtonian": surface_force_ratio,
        "observation_radius_m": observation_radius_m,
        "force_over_newtonian_at_observation": observation_force_ratio,
        "radius_fraction": profile_radius,
        "profile_radius_GeVinv": profile_radius_GeVinv,
        "density_kgm3": profile_density_kgm3,
        "density_GeV4": profile_density_GeV4,
        "field_GeV": profile_field,
        "field_gradient_GeV2": profile_derivative,
    }
    result["weak_field_metric_at_observation"] = weak_field_metric_at_radius(
        result, observation_radius_m
    )
    return result


def write_radial_profile(output_path, result):
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "r_over_R_sun",
                "rho_kgm3",
                "N_GeV",
                "dN_dr_GeV2",
            ]
        )
        writer.writerows(
            zip(
                result["radius_fraction"],
                result["density_kgm3"],
                result["field_GeV"],
                result["field_gradient_GeV2"],
            )
        )
    return output_path


def _main():
    parser = argparse.ArgumentParser(
        description="Resuelve el perfil radial solar del campo escalar."
    )
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument("--m0-eV", type=float, required=True)
    parser.add_argument("--kappa-over-lambda", type=float, default=0.0)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_SSM_PROFILE,
        help="CSV r_over_R_sun,rho_kgm3 (por defecto: perfil AAG21 incluido).",
    )
    parser.add_argument("--rho-out-kgm3", type=float, default=rho_out)
    parser.add_argument("--observer-au", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    radii, densities = load_density_profile(args.profile)
    profile_name = (
        DEFAULT_SSM_PROFILE_NAME
        if args.profile.resolve() == DEFAULT_SSM_PROFILE
        else str(args.profile)
    )

    result = solve_radial_profile(
        args.g,
        args.lambda_value,
        args.m0_eV,
        radii,
        densities,
        kappa_over_lambda=args.kappa_over_lambda,
        rho_out_kgm3=args.rho_out_kgm3,
        observation_radius_m=args.observer_au * AU_METERS,
        profile_name=profile_name,
    )
    print(f"Perfil: {result['profile_name']}")
    print(f"N(0) = {result['center_field_GeV']:.8g} GeV")
    print(f"N(R_sun) = {result['surface_field_GeV']:.8g} GeV")
    print(
        f"Carga escalar superficial de la fuente = "
        f"{result['beta_scalar_surface']:.8g}"
    )
    print(
        f"Carga escalar normalizada en el empalme = "
        f"{result['beta_scalar_match']:.8g}"
    )
    print(
        "Razón no lineal exterior (superficie/empalme) = "
        f"{result['surface_exterior_nonlinearity_ratio']:.8g}/"
        f"{result['exterior_nonlinearity_ratio']:.8g}"
    )
    print(
        "Radio de empalme lineal = "
        f"{result['exterior_match_radius_m'] / result['body_radius_m']:.8g} "
        "radios del cuerpo"
    )
    print(
        "F_escalar/F_Newton (perfil radial aislado) en la superficie = "
        f"{result['surface_force_over_newtonian']:.8g}"
    )
    print(
        "F_escalar/F_Newton (perfil radial aislado) "
        f"a {args.observer_au:.6g} AU = "
        f"{result['force_over_newtonian_at_observation']:.8g}"
    )
    metric = result["weak_field_metric_at_observation"]
    print(
        f"Potenciales métricos débiles a {args.observer_au:.6g} AU: "
        f"Phi={metric['Phi']:.8g}, Psi={metric['Psi']:.8g}"
    )
    print(
        f"g_tt={metric['g_tt']:.12g}, "
        f"g_rr (areal)={metric['g_rr_areal']:.12g}"
    )
    print(
        "La métrica es un postproceso débil de un perfil escalar plano; "
        "no es una solución autoconsistente ni un ajuste PPN de Cassini."
    )
    print(
        "La fuerza radial es de un cuerpo aislado; la fórmula de dos cuerpos "
        "requiere regiones exteriores lineales no solapadas."
    )
    if args.output:
        print(f"Perfil radial guardado en {write_radial_profile(args.output, result)}")


if __name__ == "__main__":
    _main()