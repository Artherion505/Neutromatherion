from contextlib import redirect_stdout
from io import StringIO
from math import pi
import unittest

import numpy as np

from flrw_fiducial import run_fiducial_check
from flrw_slowroll import solve_slow_roll_history
from flrw_perturbations import (
    assemble_coupled_dust_scalar_mode,
    assemble_h_zero_dust_scalar_mode,
    canonical_scalar_stress_moments,
    clock_field_perturbation_rhs,
    clock_interaction_current_divergence_perturbation,
    collisionless_fourier_transport_rhs,
    collisionless_hamiltonian_force_multipoles,
    collisionless_multipole_streaming_rhs,
    coupled_massive_worldline_hilbert_stress_tensor,
    coupled_pressureless_matter_scalar_rhs,
    coupled_pressureless_matter_scalar_source_perturbation,
    coupled_pressureless_matter_stress_moments,
    finite_h_thomson_collision_velocity_projection,
    finite_h_thomson_cross_section_ratio,
    finite_h_thomson_opacity_per_conformal_time,
    free_streaming_bessel_closure,
    free_streaming_distance_history,
    free_streaming_multipole_streaming_rhs,
    integrate_free_streaming_multipoles,
    isotropic_fluid_scalar_source,
    isotropic_fluid_scalar_source_perturbation,
    modeled_matter_clock_current_divergence_perturbation,
    newtonian_gauge_metric_constraints,
    neutrino_background_dispersion_stress_moments,
    neutrino_energy_local,
    neutrino_dispersion_interaction_stress_multipoles,
    neutrino_hamiltonian_perturbation,
    neutrino_hamiltonian_mode_coefficients,
    neutrino_group_velocity_local,
    neutrino_kinetic_stress_multipoles,
    neutrino_scalar_mode_rhs,
    neutrino_spatial_metric_measure_response_moments,
    neutrino_scalar_source_from_distribution_multipoles,
    neutrino_scalar_source_per_particle,
    neutrino_scalar_source_perturbation,
    neutrino_scalar_source_spatial_metric_measure_response,
    neutrino_worldline_hilbert_background_response_moments,
    neutrino_worldline_hilbert_clock_frame_response_moments,
    neutrino_worldline_hilbert_contact_stress_multipoles,
    neutrino_worldline_hilbert_spatial_metric_response_moments,
    neutrino_worldline_hilbert_stress_multipoles,
    photon_hamiltonian_perturbation,
    photon_hamiltonian_mode_coefficients,
    photon_group_velocity_local,
    photon_background_dispersion_stress_moments,
    photon_clock_frame_stress_response_moments,
    photon_interaction_background_response_moments,
    photon_interaction_clock_frame_stress_response_moments,
    photon_interaction_hilbert_clock_frame_contact_moments,
    photon_interaction_stress_multipoles,
    photon_interaction_hilbert_stress_tensor,
    photon_maxwell_background_response_moments,
    photon_maxwell_clock_frame_stress_response_moments,
    photon_scalar_intensity_multipole_rhs,
    photon_action_stress_perturbation_moments,
    photon_maxwell_stress_multipoles,
    photon_scalar_mode_rhs,
    photon_scalar_source_background_response,
    photon_scalar_source_from_distribution_multipoles,
    photon_scalar_source_perturbation,
    photon_scalar_source_spatial_metric_measure_response,
    photon_spatial_metric_measure_response_moments,
    perfect_fluid_scalar_stress_moments,
    photon_kinetic_stress_multipoles,
    scalar_field_perturbation_rhs,
    scalar_mode_multipole_rhs,
    sum_scalar_source_perturbations,
    standard_thomson_scalar_collision_multipoles,
    standard_thomson_baryon_velocity_collision_source,
    standard_thomson_scalar_photon_mode_rhs,
    standard_thomson_photon_baryon_mode_rhs,
    standard_thomson_finite_h_photon_baryon_mode_rhs,
    standard_baryon_scalar_rhs,
    standard_thomson_opacity_per_conformal_time,
    sum_scalar_stress_moments,
    uncoupled_kinetic_stress_multipoles,
)
from flrw_coupled import (
    FLRWInitialConditions,
    FLRWParameters,
    T_NU0_GEV,
    background_quantities,
    photon_speed_log_derivative,
    photon_speed_ratio,
    potential_derivative,
    potential,
    relic_neutrino_components,
    relic_neutrino_energy_pressure,
    rhs_log_scale_factor,
    solve_background,
)


class CoupledFLRWTests(unittest.TestCase):
    def test_finite_h_thomson_opacity_has_stationary_electron_limit(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.2,
        )
        opacity = np.array([1.3, 0.8, 0.4])
        field = np.array([0.0, 0.3, -0.2])
        xi = parameters.h_GeV_inv * field

        cross_section_ratio = finite_h_thomson_cross_section_ratio(
            field, parameters
        )
        electron_inertial_mass_ratio = 1.0 + (
            parameters.g_GeV_inv + parameters.h_GeV_inv
        ) * field
        expected_cross_section_ratio = 1.0 / (
            (1.0 - xi) ** 2 * electron_inertial_mass_ratio**2
        )
        np.testing.assert_allclose(
            cross_section_ratio,
            expected_cross_section_ratio,
            rtol=1e-14,
            atol=0.0,
        )

        actual = finite_h_thomson_opacity_per_conformal_time(
            opacity, field, parameters
        )

        expected = (
            opacity
            * np.sqrt((1.0 - xi) / (1.0 + xi))
            * expected_cross_section_ratio
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
        self.assertEqual(actual[0], opacity[0])
        self.assertNotEqual(actual[1], opacity[1])
        with self.assertRaises(ValueError):
            finite_h_thomson_opacity_per_conformal_time(-1.0, 0.0, parameters)
        with self.assertRaises(ValueError):
            finite_h_thomson_opacity_per_conformal_time(1.0, 5.0, parameters)
        with self.assertRaises(ValueError):
            finite_h_thomson_cross_section_ratio(5.0, parameters)
        negative_inertia_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=-1.0,
            h_GeV_inv=0.0,
        )
        with self.assertRaises(ValueError):
            finite_h_thomson_cross_section_ratio(
                2.0, negative_inertia_parameters
            )

    def test_standard_thomson_opacity_uses_physical_electron_density(self):
        scale_factors = np.array([0.3, 0.8, 1.0])
        electron_densities = np.array([2e-8, 7e-9, 0.0])
        expected = (
            scale_factors
            * electron_densities
            * (8.0 * np.pi / 3.0)
            * (1.0 / 137.035999084 / 5.1099895e-4) ** 2
        )

        actual = standard_thomson_opacity_per_conformal_time(
            scale_factors, electron_densities
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)
        self.assertEqual(standard_thomson_opacity_per_conformal_time(0.5, 0.0), 0.0)
        with self.assertRaises(ValueError):
            standard_thomson_opacity_per_conformal_time(0.0, 1e-8)
        with self.assertRaises(ValueError):
            standard_thomson_opacity_per_conformal_time(0.5, -1e-8)
        with self.assertRaises(ValueError):
            standard_thomson_opacity_per_conformal_time(
                [0.5, 1.0], [1e-8, 2e-8, 3e-8]
            )

    def test_finite_h_collision_velocity_has_correct_frame_limits(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.2,
        )
        field = 0.3
        speed = photon_group_velocity_local(field, parameters)
        clock_velocity = 0.004 - 0.001j

        collision_velocity = finite_h_thomson_collision_velocity_projection(
            clock_velocity, clock_velocity, field, parameters
        )

        self.assertAlmostEqual(collision_velocity, speed * clock_velocity, places=15)
        momenta = np.array([0.2, 0.5, 0.9])
        distribution_derivative = np.array([-0.7, -0.3, -0.1])
        intensity = np.zeros((3, 4), dtype=complex)
        intensity[:, 1] = (
            -speed
            * momenta
            * distribution_derivative
            * clock_velocity
            / 3.0
        )
        opacity = np.array([1.0, 0.8, 0.5])
        collision_intensity, _ = standard_thomson_scalar_collision_multipoles(
            intensity,
            np.zeros_like(intensity),
            momenta,
            distribution_derivative,
            opacity,
            collision_velocity,
        )
        np.testing.assert_allclose(
            collision_intensity[:, 1], 0.0, rtol=0.0, atol=1e-15
        )

        standard_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.0,
        )
        standard_velocity = finite_h_thomson_collision_velocity_projection(
            0.006 + 0.002j,
            clock_velocity,
            field,
            standard_parameters,
        )
        self.assertAlmostEqual(standard_velocity, 0.006 + 0.002j, places=15)

    def test_stress_moment_sum_composes_components_for_metric_constraints(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.05,
            h_GeV_inv=0.02,
        )
        fluid = perfect_fluid_scalar_stress_moments(
            0.04,
            0.002,
            0.1 + 0.02j,
            0.03 - 0.01j,
            sound_speed_squared=0.02,
        )
        dust = coupled_pressureless_matter_stress_moments(
            0.01,
            -0.04 + 0.01j,
            0.01 + 0.005j,
            0.2,
            0.03 - 0.01j,
            parameters,
        )

        total = sum_scalar_stress_moments(fluid, dust)

        for key in total:
            self.assertAlmostEqual(total[key], fluid[key] + dust[key], places=15)
        metric = newtonian_gauge_metric_constraints(
            0.8,
            0.1,
            0.7,
            1.2,
            total["delta_rho_GeV4"],
            total["longitudinal_flux_GeV4"],
            total["longitudinal_anisotropic_stress_GeV4"],
        )
        self.assertAlmostEqual(
            metric["phi_prime_plus_hubble_psi_GeV"],
            1j
            * 0.7**2
            * total["longitudinal_flux_GeV4"]
            / (2.0 * 1.2**2 * 0.8),
            places=15,
        )
        with self.assertRaises(ValueError):
            sum_scalar_stress_moments()
        with self.assertRaises(ValueError):
            sum_scalar_stress_moments({"delta_rho_GeV4": 0.0})

    def test_coupled_dust_rhs_recovers_the_uncoupled_limit(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        density_contrast = 0.04 - 0.02j
        velocity = 0.007 - 0.004j
        wavenumber = 0.9
        conformal_hubble = 0.13
        psi = 0.02 + 0.003j
        phi_prime = -3e-6 + 1e-6j

        actual = coupled_pressureless_matter_scalar_rhs(
            density_contrast,
            velocity,
            wavenumber,
            conformal_hubble,
            psi,
            phi_prime,
            0.2,
            0.05,
            0.03 - 0.01j,
            parameters,
        )
        expected = standard_baryon_scalar_rhs(
            density_contrast,
            velocity,
            wavenumber,
            conformal_hubble,
            psi,
            phi_prime,
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_coupled_dust_rhs_includes_mass_variation_and_scalar_force(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.05,
            h_GeV_inv=0.02,
        )
        density_contrast = 0.04 - 0.02j
        velocity = 0.007 - 0.004j
        wavenumber = 0.9
        conformal_hubble = 0.13
        psi = 0.02 + 0.003j
        phi_prime = -3e-6 + 1e-6j
        background_N = 0.2
        background_N_prime = 0.05
        delta_N = 0.03 - 0.01j
        rest_mass_factor = 1.0 + parameters.eta_GeV_inv * background_N
        inertial_coupling = parameters.g_GeV_inv + parameters.h_GeV_inv
        inertial_mass_factor = 1.0 + inertial_coupling * background_N

        density_rhs, velocity_rhs = coupled_pressureless_matter_scalar_rhs(
            density_contrast,
            velocity,
            wavenumber,
            conformal_hubble,
            psi,
            phi_prime,
            background_N,
            background_N_prime,
            delta_N,
            parameters,
        )

        self.assertAlmostEqual(
            density_rhs,
            -1j * wavenumber * velocity + 3.0 * phi_prime,
            places=15,
        )
        self.assertAlmostEqual(
            velocity_rhs,
            -(
                conformal_hubble
                + inertial_coupling
                * background_N_prime
                / inertial_mass_factor
            )
            * velocity
            - 1j
            * wavenumber
            * (
                rest_mass_factor / inertial_mass_factor * psi
                + parameters.eta_GeV_inv
                * delta_N
                / inertial_mass_factor
            ),
            places=15,
        )

    def test_coupled_dust_euler_includes_clock_frame_motion(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.05,
            h_GeV_inv=0.02,
        )
        background_N = 0.2
        background_N_prime = 0.05
        conformal_hubble = 0.07
        clock_velocity = 0.006 - 0.002j
        clock_velocity_prime = 0.004 + 0.001j

        density_rhs, velocity_rhs = coupled_pressureless_matter_scalar_rhs(
            0.0,
            0.0,
            0.0,
            conformal_hubble,
            0.0,
            0.0,
            background_N,
            background_N_prime,
            0.0,
            parameters,
            clock_velocity_projection=clock_velocity,
            clock_velocity_prime_GeV=clock_velocity_prime,
        )

        inertial_mass_factor = 1.0 + (
            parameters.g_GeV_inv + parameters.h_GeV_inv
        ) * background_N
        expected_clock_source = (
            2.0
            * parameters.h_GeV_inv
            / inertial_mass_factor
            * (
                background_N * clock_velocity_prime
                + (
                    background_N_prime
                    + conformal_hubble * background_N
                )
                * clock_velocity
            )
        )
        self.assertEqual(density_rhs, 0.0)
        self.assertAlmostEqual(velocity_rhs, expected_clock_source, places=15)

    def test_coupled_dust_euler_accepts_collision_acceleration(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.05,
            h_GeV_inv=0.02,
        )
        collision_source = 0.003 - 0.001j

        density_rhs, velocity_rhs = coupled_pressureless_matter_scalar_rhs(
            0.0,
            0.0,
            0.0,
            0.07,
            0.0,
            0.0,
            0.2,
            0.05,
            0.0,
            parameters,
            collision_velocity_source=collision_source,
        )

        self.assertEqual(density_rhs, 0.0)
        self.assertEqual(velocity_rhs, collision_source)

    def test_worldline_hilbert_stress_matches_metric_shift_variation(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
        )
        number_density = 0.7
        particle_mass = 0.4
        field = 0.2
        particle_velocity = np.array([0.18, -0.07, 0.03])
        clock_velocity = np.array([-0.025, 0.016, 0.008])
        particle_gamma = 1.0 / np.sqrt(1.0 - particle_velocity @ particle_velocity)
        clock_gamma = 1.0 / np.sqrt(1.0 - clock_velocity @ clock_velocity)
        particle_four_velocity = np.concatenate(
            ([particle_gamma], particle_gamma * particle_velocity)
        )
        clock_four_velocity = np.concatenate(
            ([clock_gamma], clock_gamma * clock_velocity)
        )
        stress = coupled_massive_worldline_hilbert_stress_tensor(
            number_density,
            particle_mass,
            field,
            clock_four_velocity,
            particle_four_velocity,
            parameters,
        )

        minkowski_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        clock_gradient = -(minkowski_metric @ clock_four_velocity)
        worldline_tangent = np.concatenate(([1.0], particle_velocity))

        def point_particle_lagrangian(metric_row, metric_column, perturbation):
            metric = minkowski_metric.copy()
            metric[metric_row, metric_column] += perturbation
            if metric_row != metric_column:
                metric[metric_column, metric_row] += perturbation
            inverse_metric = np.linalg.inv(metric)
            proper_time_rate = np.sqrt(
                -(worldline_tangent @ metric @ worldline_tangent)
            )
            clock_norm = np.sqrt(
                -(clock_gradient @ inverse_metric @ clock_gradient)
            )
            clock_covector = -clock_gradient / clock_norm
            particle_four_velocity_in_metric = (
                worldline_tangent / proper_time_rate
            )
            relative_clock_factor = (
                clock_covector @ particle_four_velocity_in_metric
            )
            return -number_density * particle_mass * proper_time_rate * (
                1.0
                + parameters.g_GeV_inv * field
                - parameters.h_GeV_inv
                * field
                * relative_clock_factor**2
            )

        step = 1e-6
        for spatial_axis in range(1, 4):
            action_derivative = (
                point_particle_lagrangian(0, spatial_axis, step)
                - point_particle_lagrangian(0, spatial_axis, -step)
            ) / (2.0 * step)
            np.testing.assert_allclose(
                action_derivative,
                stress["total_stress_tensor_GeV4"][0, spatial_axis],
                rtol=1e-8,
                atol=1e-10,
            )

        for diagonal_axis in range(4):
            action_derivative = (
                point_particle_lagrangian(diagonal_axis, diagonal_axis, step)
                - point_particle_lagrangian(diagonal_axis, diagonal_axis, -step)
            ) / (2.0 * step)
            np.testing.assert_allclose(
                2.0 * action_derivative,
                stress["total_stress_tensor_GeV4"][diagonal_axis, diagonal_axis],
                rtol=1e-8,
                atol=1e-10,
            )

    def test_neutrino_hilbert_contact_closes_stress_multipoles(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
        )
        comoving_momentum = np.array([0.37])
        quadrature_weight = np.array([0.61])
        mass = 0.4
        scale_factor = 1.2
        background_N = 0.2
        distribution = np.array([[0.7, 0.025, -0.015]], dtype=complex)
        degeneracy = 1.3

        kinetic_moments = neutrino_kinetic_stress_multipoles(
            comoving_momentum,
            quadrature_weight,
            mass,
            scale_factor,
            background_N,
            distribution,
            parameters,
            degeneracy,
        )
        contact_moments = neutrino_worldline_hilbert_contact_stress_multipoles(
            comoving_momentum,
            quadrature_weight,
            mass,
            scale_factor,
            background_N,
            distribution,
            parameters,
            degeneracy,
        )
        hilbert_moments = neutrino_worldline_hilbert_stress_multipoles(
            comoving_momentum,
            quadrature_weight,
            mass,
            scale_factor,
            background_N,
            distribution,
            parameters,
            degeneracy,
        )
        physical_momentum = comoving_momentum[0] / scale_factor
        energy = np.hypot(physical_momentum, mass)
        phase_space_density = (
            degeneracy
            / (2.0 * np.pi**2)
            * quadrature_weight[0]
            * comoving_momentum[0] ** 2
            / scale_factor**3
        )
        clock_four_velocity = np.array([1.0, 0.0, 0.0, 0.0])
        action_stress = np.zeros((4, 4))
        angular_nodes, angular_weights = np.polynomial.legendre.leggauss(8)
        monopole, dipole, quadrupole = distribution[0].real
        for cosine, angular_weight in zip(angular_nodes, angular_weights):
            angular_distribution = (
                monopole
                + 3.0 * dipole * cosine
                + 5.0 * quadrupole * (3.0 * cosine**2 - 1.0) / 2.0
            )
            particle_four_velocity = np.array(
                [
                    energy / mass,
                    physical_momentum * np.sqrt(1.0 - cosine**2) / mass,
                    0.0,
                    physical_momentum * cosine / mass,
                ]
            )
            action_stress += coupled_massive_worldline_hilbert_stress_tensor(
                phase_space_density
                * angular_weight
                * angular_distribution
                / 2.0,
                mass,
                background_N,
                clock_four_velocity,
                particle_four_velocity,
                parameters,
            )["total_stress_tensor_GeV4"]

        for key in kinetic_moments:
            np.testing.assert_allclose(
                hilbert_moments[key],
                kinetic_moments[key] + contact_moments[key],
                rtol=1e-13,
                atol=1e-13,
            )
        np.testing.assert_allclose(
            hilbert_moments["delta_rho_GeV4"].real,
            action_stress[0, 0],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            hilbert_moments["delta_pressure_GeV4"].real,
            np.trace(action_stress[1:, 1:]) / 3.0,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            hilbert_moments["longitudinal_flux_GeV4"].real,
            action_stress[0, 3],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            hilbert_moments["longitudinal_anisotropic_stress_GeV4"].real,
            action_stress[3, 3]
            - np.trace(action_stress[1:, 1:]) / 3.0,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_coupled_dust_stress_includes_effective_mass_perturbation(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.05,
            h_GeV_inv=0.02,
        )
        bare_density = 0.02
        density_contrast = 0.1 + 0.02j
        velocity = 0.007 - 0.004j
        background_N = 0.2
        delta_N = 0.01 - 0.005j
        mass_factor = 1.0 + parameters.eta_GeV_inv * background_N
        inertial_mass_factor = 1.0 + (
            parameters.g_GeV_inv + parameters.h_GeV_inv
        ) * background_N

        moments = coupled_pressureless_matter_stress_moments(
            bare_density,
            density_contrast,
            velocity,
            background_N,
            delta_N,
            parameters,
        )
        scalar_source = isotropic_fluid_scalar_source_perturbation(
            bare_density * density_contrast,
            0.0,
            parameters,
        )

        self.assertAlmostEqual(
            moments["delta_rho_GeV4"],
            mass_factor * bare_density * density_contrast
            + parameters.eta_GeV_inv * bare_density * delta_N,
            places=15,
        )
        self.assertAlmostEqual(
            moments["longitudinal_flux_GeV4"],
            inertial_mass_factor * bare_density * velocity,
            places=15,
        )
        self.assertAlmostEqual(
            scalar_source,
            parameters.eta_GeV_inv * bare_density * density_contrast,
            places=15,
        )
        self.assertEqual(moments["delta_pressure_GeV4"], 0.0j)
        self.assertEqual(
            moments["longitudinal_anisotropic_stress_GeV4"], 0.0j
        )

    def test_canonical_scalar_stress_moments_include_lapse_and_potential(self):
        background_field_prime = 0.12
        field_perturbation = 0.03 - 0.02j
        field_perturbation_prime = -0.004 + 0.006j
        scale_factor = 0.7
        psi = 0.02 + 0.003j
        wavenumber = 0.9
        kinetic_normalization = 1.4
        potential_derivative = 0.5

        moments = canonical_scalar_stress_moments(
            background_field_prime,
            field_perturbation,
            field_perturbation_prime,
            scale_factor,
            psi,
            wavenumber,
            kinetic_normalization,
            potential_derivative,
        )
        kinetic_perturbation = (
            kinetic_normalization
            / scale_factor**2
            * (
                background_field_prime * field_perturbation_prime
                - psi * background_field_prime**2
            )
        )

        self.assertAlmostEqual(
            moments["delta_rho_GeV4"],
            kinetic_perturbation + potential_derivative * field_perturbation,
            places=15,
        )
        self.assertAlmostEqual(
            moments["delta_pressure_GeV4"],
            kinetic_perturbation - potential_derivative * field_perturbation,
            places=15,
        )
        self.assertAlmostEqual(
            moments["longitudinal_flux_GeV4"],
            -1j
            * wavenumber
            * kinetic_normalization
            * background_field_prime
            * field_perturbation
            / scale_factor**2,
            places=15,
        )
        self.assertEqual(
            moments["longitudinal_anisotropic_stress_GeV4"], 0.0j
        )

    def test_newtonian_gauge_metric_constraints_satisfy_einstein_equations(self):
        wavenumber = 0.7
        conformal_hubble = 0.2
        scale_factor = 0.8
        reduced_planck_mass = 1.3
        delta_rho = 0.03 - 0.01j
        longitudinal_flux = 0.004 + 0.003j
        anisotropic_stress = 0.002 - 0.001j

        metric = newtonian_gauge_metric_constraints(
            wavenumber,
            conformal_hubble,
            scale_factor,
            reduced_planck_mass,
            delta_rho,
            longitudinal_flux,
            anisotropic_stress,
        )
        expected_momentum_constraint = (
            1j
            * scale_factor**2
            * longitudinal_flux
            / (2.0 * reduced_planck_mass**2 * wavenumber)
        )

        self.assertAlmostEqual(
            metric["phi_prime_plus_hubble_psi_GeV"],
            expected_momentum_constraint,
            places=15,
        )
        self.assertAlmostEqual(
            wavenumber**2 * metric["phi"]
            + 3.0 * conformal_hubble * expected_momentum_constraint,
            -scale_factor**2 * delta_rho / (2.0 * reduced_planck_mass**2),
            places=15,
        )
        self.assertAlmostEqual(
            wavenumber**2 * (metric["phi"] - metric["psi"]),
            -3.0
            * scale_factor**2
            * anisotropic_stress
            / (2.0 * reduced_planck_mass**2),
            places=15,
        )
        self.assertAlmostEqual(
            metric["phi_prime_GeV"]
            + conformal_hubble * metric["psi"],
            expected_momentum_constraint,
            places=15,
        )

    def test_newtonian_gauge_metric_constraints_reject_zero_wavenumber(self):
        with self.assertRaises(ValueError):
            newtonian_gauge_metric_constraints(
                0.0,
                0.2,
                0.8,
                1.3,
                0.03,
                0.0,
                0.0,
            )

    def test_clock_current_divergence_vanishes_for_comoving_fluid(self):
        rho_plus_pressure = 0.7
        fluid_velocity = 0.025 - 0.012j

        divergence = clock_interaction_current_divergence_perturbation(
            0.9,
            0.04,
            0.3,
            0.12,
            rho_plus_pressure,
            rho_plus_pressure * fluid_velocity,
            fluid_velocity,
        )

        self.assertEqual(divergence, 0.0j)

    def test_clock_current_divergence_tracks_relative_fluid_velocity(self):
        wavenumber = 0.8
        h_coupling = 0.05
        N_background = 0.2
        Theta_background_prime = 0.1
        rho_plus_pressure = 0.6
        fluid_velocity = -0.03 + 0.02j
        clock_velocity = 0.01 - 0.015j

        divergence = clock_interaction_current_divergence_perturbation(
            wavenumber,
            h_coupling,
            N_background,
            Theta_background_prime,
            rho_plus_pressure,
            rho_plus_pressure * fluid_velocity,
            clock_velocity,
        )

        expected = (
            -1j
            * wavenumber
            * h_coupling
            * N_background
            / Theta_background_prime
            * rho_plus_pressure
            * (fluid_velocity - clock_velocity)
        )
        self.assertAlmostEqual(divergence, expected, places=15)

    def test_clock_field_rhs_recovers_canonical_free_limit(self):
        delta_Theta = 0.02 - 0.01j
        delta_Theta_prime = -0.003 + 0.002j
        wavenumber = 0.6
        conformal_hubble = 0.1
        scale_factor = 0.75
        Theta_background_prime = 0.08
        Theta_background_second_prime = (
            -2.0 * conformal_hubble * Theta_background_prime
        )
        psi_prime = 0.004 - 0.001j
        phi_prime = -0.002 + 0.003j

        actual_delta_Theta_prime, actual_delta_Theta_second_prime = (
            clock_field_perturbation_rhs(
                delta_Theta,
                delta_Theta_prime,
                wavenumber,
                conformal_hubble,
                scale_factor,
                Theta_background_prime,
                Theta_background_second_prime,
                1.7,
                0.03 + 0.01j,
                psi_prime,
                phi_prime,
                0.0,
            )
        )

        self.assertEqual(actual_delta_Theta_prime, delta_Theta_prime)
        self.assertAlmostEqual(
            actual_delta_Theta_second_prime,
            Theta_background_prime * (psi_prime + 3.0 * phi_prime)
            - 2.0 * conformal_hubble * delta_Theta_prime
            - wavenumber**2 * delta_Theta,
            places=15,
        )

    def test_clock_field_rhs_includes_metric_and_interaction_current(self):
        scale_factor = 0.72
        conformal_hubble = 0.13
        Theta_background_prime = 0.04
        Theta_background_second_prime = 0.006
        Z_theta = 1.4
        psi = 0.02 - 0.005j
        psi_prime = 0.003 + 0.001j
        phi_prime = -0.001 + 0.002j
        current_divergence = 0.0008 - 0.0003j

        _, delta_Theta_second_prime = clock_field_perturbation_rhs(
            0.0j,
            0.0j,
            0.9,
            conformal_hubble,
            scale_factor,
            Theta_background_prime,
            Theta_background_second_prime,
            Z_theta,
            psi,
            psi_prime,
            phi_prime,
            current_divergence,
        )

        self.assertAlmostEqual(
            delta_Theta_second_prime,
            Theta_background_prime * (psi_prime + 3.0 * phi_prime)
            + 2.0
            * psi
            * (
                Theta_background_second_prime
                + 2.0 * conformal_hubble * Theta_background_prime
            )
            + 2.0 * scale_factor**2 * current_divergence / Z_theta,
            places=15,
        )

    def test_scalar_field_perturbation_rhs_recovers_free_klein_gordon_limit(self):
        delta_N = 0.03 - 0.02j
        delta_N_prime = -0.004 + 0.001j
        wavenumber = 0.7
        conformal_hubble = 0.12
        scale_factor = 0.8
        potential_second_derivative = 0.5

        actual_delta_N_prime, actual_delta_N_second_prime = (
            scalar_field_perturbation_rhs(
                delta_N,
                delta_N_prime,
                wavenumber,
                conformal_hubble,
                scale_factor,
                potential_second_derivative,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )

        self.assertEqual(actual_delta_N_prime, delta_N_prime)
        self.assertAlmostEqual(
            actual_delta_N_second_prime,
            -2.0 * conformal_hubble * delta_N_prime
            - (
                wavenumber**2
                + scale_factor**2 * potential_second_derivative
            )
            * delta_N,
            places=15,
        )

    def test_scalar_field_perturbation_rhs_includes_metric_and_source_terms(self):
        scale_factor = 0.73
        N_background_prime = 0.06
        N_background_second_prime = -0.01
        conformal_hubble = 0.11
        psi = 0.02 + 0.003j
        psi_prime = 0.005 - 0.001j
        phi_prime = -0.002 + 0.004j
        scalar_source = 0.0007 - 0.0002j

        _, delta_N_second_prime = scalar_field_perturbation_rhs(
            0.0j,
            0.0j,
            0.8,
            conformal_hubble,
            scale_factor,
            0.4,
            N_background_prime,
            N_background_second_prime,
            psi,
            psi_prime,
            phi_prime,
            scalar_source,
        )
        expected = (
            N_background_prime * (psi_prime + 3.0 * phi_prime)
            + 2.0
            * psi
            * (
                N_background_second_prime
                + 2.0 * conformal_hubble * N_background_prime
            )
            - scale_factor**2 * scalar_source
        )

        self.assertAlmostEqual(delta_N_second_prime, expected, places=15)

    def test_photon_hamiltonian_perturbation_recovers_uncoupled_metric_limit(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        momentum = 0.7
        psi = 1e-5
        phi = -2e-5

        perturbation = photon_hamiltonian_perturbation(
            momentum,
            0.2,
            0.3,
            psi,
            phi,
            0.04,
            parameters,
        )

        self.assertAlmostEqual(perturbation, momentum * (psi + phi), places=15)

    def test_photon_hamiltonian_field_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.02,
        )
        momentum = 0.7
        field_value = 0.1
        step = 1e-6
        finite_difference = momentum * (
            photon_speed_ratio(field_value + step, parameters)
            - photon_speed_ratio(field_value - step, parameters)
        ) / (2.0 * step)

        perturbation = photon_hamiltonian_perturbation(
            momentum,
            field_value,
            step,
            0.0,
            0.0,
            0.0,
            parameters,
        )

        self.assertAlmostEqual(perturbation / step, finite_difference, places=9)

    def test_photon_hamiltonian_mode_coefficients_reconstruct_angular_response(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.02,
        )
        momentum = 0.7
        field_value = 0.1
        field_perturbation = 0.02 - 0.01j
        psi = 0.001 + 0.002j
        phi = -0.003 + 0.001j
        clock_velocity = 0.005 - 0.003j

        monopole, dipole = photon_hamiltonian_mode_coefficients(
            momentum,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            parameters,
        )

        for cosine in (-1.0, 0.0, 1.0):
            direct_hamiltonian = photon_hamiltonian_perturbation(
                momentum,
                field_value,
                field_perturbation,
                psi,
                phi,
                cosine * clock_velocity,
                parameters,
            )
            self.assertAlmostEqual(
                direct_hamiltonian, monopole + cosine * dipole, places=15
            )

        canonical_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        canonical_monopole, canonical_dipole = photon_hamiltonian_mode_coefficients(
            momentum,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            canonical_parameters,
        )
        self.assertAlmostEqual(
            canonical_monopole, momentum * (psi + phi), places=15
        )
        self.assertAlmostEqual(canonical_dipole, 0.0j, places=15)

    def test_photon_intensity_rhs_uses_modified_speed_and_supplied_collisions(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.02,
        )
        multipoles = np.array(
            [
                [0.1 + 0.02j, -0.03j, 0.04 - 0.01j],
                [0.08 - 0.01j, 0.02 + 0.01j, -0.02j],
            ]
        )
        momenta = np.array([0.4, 0.8])
        radial_gradients = np.array([-0.3, -0.12])
        next_multipoles = np.array([0.01j, -0.015 + 0.005j])
        collisions = np.array(
            [
                [0.01j, -0.02, 0.005],
                [-0.004j, 0.006 + 0.002j, -0.003],
            ]
        )
        field_value = 0.1
        field_perturbation = 0.02 - 0.01j
        psi = 0.001 + 0.002j
        phi = -0.003 + 0.001j
        clock_velocity = 0.005 - 0.003j
        wavenumber = 0.8

        actual = photon_scalar_intensity_multipole_rhs(
            multipoles,
            wavenumber,
            momenta,
            radial_gradients,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            next_multipoles,
            parameters,
            collisions,
        )
        monopole, dipole = photon_hamiltonian_mode_coefficients(
            momenta,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            parameters,
        )
        expected = scalar_mode_multipole_rhs(
            multipoles,
            wavenumber,
            photon_group_velocity_local(field_value, parameters),
            radial_gradients,
            monopole,
            dipole,
            next_multipoles,
            collisions,
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_photon_mode_rhs_streams_both_sectors_with_external_collisions(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.02,
        )
        intensity = np.array(
            [[0.1, 0.02j, -0.03, 0.01], [0.06j, -0.02, 0.01j, 0.005]]
        )
        polarization = np.array(
            [[0.01, -0.004j, 0.003, 0.002j], [-0.005j, 0.002, 0.001j, -0.001]]
        )
        intensity_collisions = np.array(
            [[0.01, -0.005j, 0.003, 0.0], [0.002j, -0.004, 0.001, 0.0]]
        )
        polarization_collisions = np.array(
            [[-0.002, 0.001j, 0.003, -0.001], [0.001j, 0.002, -0.001, 0.0]]
        )
        momenta = np.array([0.4, 0.8])
        radial_gradient = np.array([-0.3, -0.12])
        next_intensity = np.array([0.01j, -0.015 + 0.005j])
        next_polarization = np.array([0.002, -0.001j])
        field_value = 0.1
        field_perturbation = 0.02 - 0.01j
        psi = 0.001 + 0.002j
        phi = -0.003 + 0.001j
        clock_velocity = 0.005 - 0.003j
        wavenumber = 0.8

        actual_intensity, actual_polarization = photon_scalar_mode_rhs(
            intensity,
            polarization,
            wavenumber,
            momenta,
            radial_gradient,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            next_intensity,
            next_polarization,
            parameters,
            intensity_collisions,
            polarization_collisions,
        )
        expected_intensity = photon_scalar_intensity_multipole_rhs(
            intensity,
            wavenumber,
            momenta,
            radial_gradient,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            next_intensity,
            parameters,
            intensity_collisions,
        )
        expected_polarization = collisionless_multipole_streaming_rhs(
            polarization,
            wavenumber,
            photon_group_velocity_local(field_value, parameters),
            polarization_collisions,
            next_polarization,
        )

        np.testing.assert_allclose(
            actual_intensity, expected_intensity, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            actual_polarization, expected_polarization, rtol=0.0, atol=1e-15
        )

    def test_photon_mode_rhs_recovers_standard_thomson_limit(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        intensity = np.array(
            [[0.1, 0.02j, -0.03, 0.01], [0.06j, -0.02, 0.01j, 0.005]]
        )
        polarization = np.array(
            [[0.01, -0.004j, 0.003, 0.002j], [-0.005j, 0.002, 0.001j, -0.001]]
        )
        momenta = np.array([0.4, 0.8])
        radial_gradient = np.array([-0.3, -0.12])
        distribution_derivative = np.array([-0.5, -0.2])
        opacity = np.array([1.2, 0.7])
        baryon_velocity = 0.004 - 0.002j
        next_intensity = np.array([0.01j, -0.015 + 0.005j])
        next_polarization = np.array([0.002, -0.001j])
        field_value = 0.1
        psi = 0.001 + 0.002j
        phi = -0.003 + 0.001j

        intensity_collisions, polarization_collisions = (
            standard_thomson_scalar_collision_multipoles(
                intensity,
                polarization,
                momenta,
                distribution_derivative,
                opacity,
                baryon_velocity,
            )
        )
        actual = photon_scalar_mode_rhs(
            intensity,
            polarization,
            0.8,
            momenta,
            radial_gradient,
            field_value,
            0.03 - 0.01j,
            psi,
            phi,
            baryon_velocity,
            next_intensity,
            next_polarization,
            parameters,
            intensity_collisions,
            polarization_collisions,
        )
        expected = standard_thomson_scalar_photon_mode_rhs(
            intensity,
            polarization,
            0.8,
            np.ones_like(momenta),
            radial_gradient,
            momenta * (psi + phi),
            np.zeros_like(momenta),
            next_intensity,
            next_polarization,
            momenta,
            distribution_derivative,
            opacity,
            baryon_velocity,
        )

        for actual_sector, expected_sector in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_sector, expected_sector, rtol=0.0, atol=1e-15
            )

    def test_neutrino_hamiltonian_field_and_clock_frame_responses(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momentum = 0.3
        mass = 0.4
        scale_factor = 0.5
        field_value = 0.1
        physical_momentum = momentum / scale_factor
        step = 1e-6
        field_response = scale_factor * (
            neutrino_energy_local(
                physical_momentum, mass, field_value + step, parameters
            )
            - neutrino_energy_local(
                physical_momentum, mass, field_value, parameters
            )
        )
        field_perturbation = neutrino_hamiltonian_perturbation(
            momentum,
            mass,
            scale_factor,
            field_value,
            step,
            0.0,
            0.0,
            0.0,
            parameters,
        )
        frame_velocity = 0.02
        frame_perturbation = neutrino_hamiltonian_perturbation(
            momentum,
            mass,
            scale_factor,
            field_value,
            0.0,
            0.0,
            0.0,
            frame_velocity,
            parameters,
        )
        expected_frame_response = (
            scale_factor
            * 2.0
            * parameters.h_GeV_inv
            * field_value
            * physical_momentum
            * frame_velocity
        )

        self.assertAlmostEqual(field_perturbation, field_response, places=14)
        self.assertAlmostEqual(
            frame_perturbation, expected_frame_response, places=14
        )

    def test_species_hamiltonians_vectorize_scalar_mode_frame_angle(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        cosines = np.array([-0.7, 0.0, 0.45])
        clock_velocity = 0.025
        frame_projection = clock_velocity * cosines
        field_value = 0.1
        field_perturbation = 0.002
        psi = 1e-5
        phi = -4e-6

        photon_momentum = 0.7
        photon_speed = photon_speed_ratio(field_value, parameters)
        photon_speed_derivative = photon_speed * photon_speed_log_derivative(
            field_value, parameters
        )
        photon_monopole = photon_momentum * (
            photon_speed * (psi + phi)
            + photon_speed_derivative * field_perturbation
        )
        photon_dipole = (
            photon_momentum * (1.0 - photon_speed**2) * clock_velocity
        )
        photon_hamiltonian = photon_hamiltonian_perturbation(
            photon_momentum,
            field_value,
            field_perturbation,
            psi,
            phi,
            frame_projection,
            parameters,
        )

        neutrino_momentum = 0.3
        neutrino_mass = 0.4
        scale_factor = 0.5
        physical_momentum = neutrino_momentum / scale_factor
        neutrino_energy = neutrino_energy_local(
            physical_momentum, neutrino_mass, field_value, parameters
        )
        neutrino_velocity = neutrino_group_velocity_local(
            physical_momentum, neutrino_mass, field_value, parameters
        )
        neutrino_scalar_response = neutrino_scalar_source_per_particle(
            physical_momentum, neutrino_mass, parameters
        )
        neutrino_monopole = scale_factor * (
            psi * neutrino_energy
            + phi * physical_momentum * neutrino_velocity
            + field_perturbation * neutrino_scalar_response
        )
        neutrino_dipole = (
            scale_factor
            * 2.0
            * parameters.h_GeV_inv
            * field_value
            * physical_momentum
            * clock_velocity
        )
        neutrino_hamiltonian = neutrino_hamiltonian_perturbation(
            neutrino_momentum,
            neutrino_mass,
            scale_factor,
            field_value,
            field_perturbation,
            psi,
            phi,
            frame_projection,
            parameters,
        )

        np.testing.assert_allclose(
            photon_hamiltonian,
            photon_monopole + photon_dipole * cosines,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            neutrino_hamiltonian,
            neutrino_monopole + neutrino_dipole * cosines,
            rtol=0.0,
            atol=1e-15,
        )

    def test_neutrino_group_velocity_matches_energy_derivative(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momentum = 0.3
        mass = 0.4
        field_value = 0.1
        step = 1e-6
        finite_difference = (
            neutrino_energy_local(
                momentum + step, mass, field_value, parameters
            )
            - neutrino_energy_local(
                momentum - step, mass, field_value, parameters
            )
        ) / (2.0 * step)

        self.assertAlmostEqual(
            neutrino_group_velocity_local(
                momentum, mass, field_value, parameters
            ),
            finite_difference,
            places=10,
        )

    def test_neutrino_hamiltonian_mode_coefficients_reconstruct_angular_response(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.3, 0.7])
        mass = 0.4
        scale_factor = 0.5
        field_value = 0.1
        field_perturbation = 0.002 - 0.0003j
        psi = 1e-5 + 0.2e-5j
        phi = -0.7e-5 + 0.1e-5j
        clock_velocity = 0.003 - 0.001j

        monopole, dipole = neutrino_hamiltonian_mode_coefficients(
            momenta,
            mass,
            scale_factor,
            field_value,
            field_perturbation,
            psi,
            phi,
            clock_velocity,
            parameters,
        )

        for cosine in (-1.0, 0.0, 1.0):
            direct = np.array(
                [
                    neutrino_hamiltonian_perturbation(
                        momentum,
                        mass,
                        scale_factor,
                        field_value,
                        field_perturbation,
                        psi,
                        phi,
                        cosine * clock_velocity,
                        parameters,
                    )
                    for momentum in momenta
                ]
            )
            np.testing.assert_allclose(
                direct, monopole + cosine * dipole, rtol=0.0, atol=1e-15
            )

    def test_neutrino_scalar_mode_rhs_recovers_canonical_hierarchy(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        momenta = np.array([0.2, 0.45, 0.8])
        mass = 0.35
        scale_factor = 0.6
        physical_momenta = momenta / scale_factor
        energies = np.hypot(physical_momenta, mass)
        multipoles = np.array(
            [
                [0.03, 0.02j, -0.01, 0.004],
                [0.02j, -0.01, 0.006j, -0.002],
                [0.01, 0.004 - 0.002j, 0.003, 0.001j],
            ],
            dtype=complex,
        )
        radial_gradient = np.array([-0.4, -0.2, -0.08])
        next_multipole = np.array([0.001, -0.0005j, 0.0003])
        wavenumber = 1.1
        psi = 1.2e-5 - 0.2e-5j
        phi = -0.8e-5 + 0.1e-5j

        actual = neutrino_scalar_mode_rhs(
            multipoles,
            wavenumber,
            momenta,
            mass,
            scale_factor,
            radial_gradient,
            0.1,
            0.003 - 0.001j,
            psi,
            phi,
            0.004 + 0.002j,
            next_multipole,
            parameters,
        )
        expected = scalar_mode_multipole_rhs(
            multipoles,
            wavenumber,
            physical_momenta / energies,
            radial_gradient,
            scale_factor * (psi * energies + phi * physical_momenta**2 / energies),
            np.zeros_like(momenta, dtype=complex),
            next_multipole,
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_uncoupled_total_mode_stress_obeys_energy_conservation(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        wavenumber = 1.1
        momenta = np.array([0.2, 0.45, 0.8])
        weights = np.array([0.05, 0.12, 0.07])
        neutrino_multipoles = np.array(
            [
                [0.03 + 0.01j, 0.02 - 0.01j, -0.004j],
                [-0.02j, -0.01 + 0.005j, 0.003],
                [0.01 - 0.002j, 0.004j, 0.002 - 0.001j],
            ]
        )
        photon_intensity = np.array(
            [
                [0.02j, -0.01 + 0.004j, 0.003],
                [0.015 - 0.003j, 0.006j, -0.002],
                [-0.01j, 0.003 + 0.002j, 0.001j],
            ]
        )
        photon_polarization = np.zeros_like(photon_intensity)
        next_multipole = np.array([0.002j, -0.001, 0.003 + 0.001j])
        zero_gradient = np.zeros_like(momenta)

        neutrino_rhs = neutrino_scalar_mode_rhs(
            neutrino_multipoles,
            wavenumber,
            momenta,
            0.0,
            1.0,
            zero_gradient,
            0.0,
            0.0,
            0.0j,
            0.0j,
            0.0j,
            next_multipole,
            parameters,
        )
        photon_rhs, _ = photon_scalar_mode_rhs(
            photon_intensity,
            photon_polarization,
            wavenumber,
            momenta,
            zero_gradient,
            0.0,
            0.0,
            0.0j,
            0.0j,
            0.0j,
            next_multipole,
            np.zeros_like(momenta),
            parameters,
            np.zeros_like(photon_intensity),
            np.zeros_like(photon_polarization),
        )

        neutrino_stress = uncoupled_kinetic_stress_multipoles(
            momenta, weights, 0.0, 1.0, neutrino_multipoles
        )
        neutrino_stress_rhs = uncoupled_kinetic_stress_multipoles(
            momenta, weights, 0.0, 1.0, neutrino_rhs
        )
        photon_stress = photon_maxwell_stress_multipoles(
            momenta, weights, 1.0, 0.0, photon_intensity, parameters
        )
        photon_stress_rhs = photon_maxwell_stress_multipoles(
            momenta, weights, 1.0, 0.0, photon_rhs, parameters
        )

        dust_density = 0.7
        dust_contrast = 0.04 - 0.01j
        dust_velocity = -0.03 + 0.02j
        dust_contrast_rhs, dust_velocity_rhs = coupled_pressureless_matter_scalar_rhs(
            dust_contrast,
            dust_velocity,
            wavenumber,
            0.0,
            0.0j,
            0.0j,
            0.0,
            0.0,
            0.0j,
            parameters,
        )
        dust_stress = coupled_pressureless_matter_stress_moments(
            dust_density, dust_contrast, dust_velocity, 0.0, 0.0, parameters
        )
        dust_stress_rhs = coupled_pressureless_matter_stress_moments(
            dust_density,
            dust_contrast_rhs,
            dust_velocity_rhs,
            0.0,
            0.0,
            parameters,
        )

        total_stress = sum_scalar_stress_moments(
            dust_stress, photon_stress, neutrino_stress
        )
        total_stress_rhs = sum_scalar_stress_moments(
            dust_stress_rhs, photon_stress_rhs, neutrino_stress_rhs
        )
        np.testing.assert_allclose(
            total_stress_rhs["delta_rho_GeV4"]
            + 1j * wavenumber * total_stress["longitudinal_flux_GeV4"],
            0.0j,
            rtol=1e-13,
            atol=1e-14,
        )

    def test_finite_coupling_dust_scalar_mode_stress_obeys_energy_conservation(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.4,
            lambda_N=0.2,
            g_GeV_inv=0.08,
            h_GeV_inv=0.0,
        )
        wavenumber = 0.9
        bare_density = 0.6
        field = 0.2
        field_prime = 0.03
        delta_field = 0.012 - 0.003j
        delta_field_prime = -0.02 + 0.005j
        density_contrast = 0.04 - 0.01j
        velocity = -0.03 + 0.02j
        potential_prime = potential_derivative(field, parameters)
        potential_second = (
            parameters.m_phi_GeV**2
            + 3.0 * parameters.lambda_N * field**2
        )
        background_source = isotropic_fluid_scalar_source(
            bare_density, 0.0, parameters
        )
        field_second = -potential_prime - background_source
        scalar_source = coupled_pressureless_matter_scalar_source_perturbation(
            bare_density, density_contrast, parameters
        )

        density_contrast_rhs, _ = coupled_pressureless_matter_scalar_rhs(
            density_contrast,
            velocity,
            wavenumber,
            0.0,
            0.0j,
            0.0,
            field,
            field_prime,
            delta_field,
            parameters,
        )
        _, delta_field_second = scalar_field_perturbation_rhs(
            delta_field,
            delta_field_prime,
            wavenumber,
            0.0,
            1.0,
            potential_second,
            field_prime,
            field_second,
            0.0j,
            0.0j,
            0.0j,
            scalar_source,
        )
        dust_stress = coupled_pressureless_matter_stress_moments(
            bare_density,
            density_contrast,
            velocity,
            field,
            delta_field,
            parameters,
        )
        scalar_stress = canonical_scalar_stress_moments(
            field_prime,
            delta_field,
            delta_field_prime,
            1.0,
            0.0j,
            wavenumber,
            potential_derivative_GeV3=potential_prime,
        )
        total_stress = sum_scalar_stress_moments(
            dust_stress, scalar_stress
        )

        dust_density_rate = (
            parameters.eta_GeV_inv
            * field_prime
            * bare_density
            * density_contrast
            + (1.0 + parameters.eta_GeV_inv * field)
            * bare_density
            * density_contrast_rhs
            + parameters.eta_GeV_inv
            * bare_density
            * delta_field_prime
        )
        scalar_density_rate = (
            field_second * delta_field_prime
            + field_prime * delta_field_second
            + potential_second * field_prime * delta_field
            + potential_prime * delta_field_prime
        )
        np.testing.assert_allclose(
            dust_density_rate
            + scalar_density_rate
            + 1j
            * wavenumber
            * total_stress["longitudinal_flux_GeV4"],
            0.0j,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_finite_h_neutrino_clock_mode_obeys_energy_conservation(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.02,
        )
        wavenumber = 1.1
        field = 5e-4
        theta_prime = 0.12
        momenta = np.array([0.2, 0.45, 0.8])
        weights = np.array([0.05, 0.12, 0.07])
        neutrino_multipoles = np.array(
            [
                [0.03 + 0.01j, 0.02 - 0.01j, -0.004j],
                [-0.02j, -0.01 + 0.005j, 0.003],
                [0.01 - 0.002j, 0.004j, 0.002 - 0.001j],
            ]
        )
        zero_gradient = np.zeros_like(momenta)
        next_multipole = np.array([0.002j, -0.001, 0.003 + 0.001j])
        neutrino_rhs = neutrino_scalar_mode_rhs(
            neutrino_multipoles,
            wavenumber,
            momenta,
            0.0,
            1.0,
            zero_gradient,
            field,
            0.0,
            0.0j,
            0.0j,
            0.0j,
            next_multipole,
            parameters,
        )

        neutrino_stress = neutrino_kinetic_stress_multipoles(
            momenta,
            weights,
            0.0,
            1.0,
            field,
            neutrino_multipoles,
            parameters,
        )
        neutrino_stress_rhs = neutrino_kinetic_stress_multipoles(
            momenta,
            weights,
            0.0,
            1.0,
            field,
            neutrino_rhs,
            parameters,
        )
        uncoupled_stress = uncoupled_kinetic_stress_multipoles(
            momenta, weights, 0.0, 1.0, neutrino_multipoles
        )
        background_multipoles = np.zeros_like(neutrino_multipoles)
        background_multipoles[:, 0] = [0.8, 0.5, 0.2]
        background_stress = uncoupled_kinetic_stress_multipoles(
            momenta, weights, 0.0, 1.0, background_multipoles
        )
        current_divergence = modeled_matter_clock_current_divergence_perturbation(
            wavenumber,
            field,
            theta_prime,
            0.0,
            parameters,
            {
                "rho_plus_pressure_matter_uncoupled_GeV4": (
                    background_stress["delta_rho_GeV4"]
                    + background_stress["delta_pressure_GeV4"]
                )
            },
            uncoupled_stress,
        )
        _, theta_second = clock_field_perturbation_rhs(
            0.0j,
            0.0j,
            wavenumber,
            0.0,
            1.0,
            theta_prime,
            0.0,
            parameters.Z_theta,
            0.0j,
            0.0j,
            0.0j,
            current_divergence,
        )
        clock_stress = canonical_scalar_stress_moments(
            theta_prime,
            0.0j,
            0.0j,
            1.0,
            0.0j,
            wavenumber,
            kinetic_normalization=parameters.Z_theta,
        )
        clock_stress_rhs = canonical_scalar_stress_moments(
            theta_prime,
            0.0j,
            theta_second,
            1.0,
            0.0j,
            wavenumber,
            kinetic_normalization=parameters.Z_theta,
        )
        total_stress = sum_scalar_stress_moments(
            neutrino_stress, clock_stress
        )
        total_stress_rhs = sum_scalar_stress_moments(
            neutrino_stress_rhs, clock_stress_rhs
        )
        np.testing.assert_allclose(
            total_stress_rhs["delta_rho_GeV4"]
            + 1j * wavenumber * total_stress["longitudinal_flux_GeV4"],
            0.0j,
            rtol=0.0,
            atol=1e-14,
        )

    def test_finite_h_dust_clock_mode_stress_obeys_energy_conservation(self):
        h_coupling = 0.02
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=h_coupling,
            h_GeV_inv=h_coupling,
            Z_theta=1.4,
        )
        wavenumber = 0.9
        bare_density = 0.6
        field = 0.3
        theta_prime = 0.12
        density_contrast = 0.04 - 0.01j
        dust_velocity = -0.03 + 0.02j
        delta_theta = 0.015 + 0.002j
        delta_theta_prime = -0.01 + 0.004j
        clock_velocity = -1j * wavenumber * delta_theta / theta_prime
        clock_velocity_prime = (
            -1j * wavenumber * delta_theta_prime / theta_prime
        )

        density_rhs, _ = coupled_pressureless_matter_scalar_rhs(
            density_contrast,
            dust_velocity,
            wavenumber,
            0.0,
            0.0j,
            0.0,
            field,
            0.0,
            0.0j,
            parameters,
            clock_velocity_projection=clock_velocity,
            clock_velocity_prime_GeV=clock_velocity_prime,
        )
        uncoupled_dust_stress = perfect_fluid_scalar_stress_moments(
            bare_density,
            0.0,
            density_contrast,
            dust_velocity,
        )
        current_divergence = clock_interaction_current_divergence_perturbation(
            wavenumber,
            h_coupling,
            field,
            theta_prime,
            bare_density,
            uncoupled_dust_stress["longitudinal_flux_GeV4"],
            clock_velocity,
        )
        _, theta_second_prime = clock_field_perturbation_rhs(
            delta_theta,
            delta_theta_prime,
            wavenumber,
            0.0,
            1.0,
            theta_prime,
            0.0,
            parameters.Z_theta,
            0.0j,
            0.0j,
            0.0j,
            current_divergence,
        )
        dust_stress = coupled_pressureless_matter_stress_moments(
            bare_density,
            density_contrast,
            dust_velocity,
            field,
            0.0j,
            parameters,
            clock_velocity_projection=clock_velocity,
        )
        clock_stress = canonical_scalar_stress_moments(
            theta_prime,
            delta_theta,
            delta_theta_prime,
            1.0,
            0.0j,
            wavenumber,
            kinetic_normalization=parameters.Z_theta,
        )
        total_energy_rate = (
            bare_density * density_rhs
            + parameters.Z_theta * theta_prime * theta_second_prime
        )
        total_energy_flux = (
            dust_stress["longitudinal_flux_GeV4"]
            + clock_stress["longitudinal_flux_GeV4"]
        )
        np.testing.assert_allclose(
            total_energy_rate + 1j * wavenumber * total_energy_flux,
            0.0j,
            rtol=1e-13,
            atol=1e-14,
        )

    def test_finite_h_dust_scalar_mode_assembles_constraints_and_clock_current(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.4,
            lambda_N=0.2,
            g_GeV_inv=0.08,
            h_GeV_inv=0.025,
            Z_theta=1.4,
            Mpl_GeV=1.3,
        )
        scale_factor = 0.8
        bare_density = 0.6
        field = 0.2
        field_prime = 0.03
        theta_prime = 0.12
        rho_total = (
            (1.0 + parameters.eta_GeV_inv * field) * bare_density
            + 0.5 * (field_prime / scale_factor) ** 2
            + potential(field, parameters)
            + 0.5
            * parameters.Z_theta
            * (theta_prime / scale_factor) ** 2
        )
        conformal_hubble = scale_factor * np.sqrt(
            rho_total / (3.0 * parameters.Mpl_GeV**2)
        )
        background_source = isotropic_fluid_scalar_source(
            bare_density, 0.0, parameters
        )
        background = {
            "rho_m_bare_GeV4": bare_density,
            "N_GeV": field,
            "N_prime_GeV2": field_prime,
            "N_second_prime_GeV3": (
                -2.0 * conformal_hubble * field_prime
                - scale_factor**2
                * (potential_derivative(field, parameters) + background_source)
            ),
            "Theta_prime_GeV2": theta_prime,
            "Theta_second_prime_GeV3": -2.0 * conformal_hubble * theta_prime,
        }
        perturbations = {
            "delta_m": 0.04 - 0.01j,
            "velocity_m": -0.03 + 0.02j,
            "delta_N_GeV": 0.012 - 0.003j,
            "delta_N_prime_GeV2": -0.02 + 0.005j,
            "delta_Theta_GeV": 0.015 + 0.002j,
            "delta_Theta_prime_GeV2": -0.01 + 0.004j,
        }
        wavenumber = 0.9
        clock_velocity = (
            -1j
            * wavenumber
            * perturbations["delta_Theta_GeV"]
            / theta_prime
        )
        clock_velocity_prime = -1j * wavenumber * (
            perturbations["delta_Theta_prime_GeV2"] / theta_prime
            - perturbations["delta_Theta_GeV"]
            * background["Theta_second_prime_GeV3"]
            / theta_prime**2
        )

        actual = assemble_coupled_dust_scalar_mode(
            wavenumber,
            conformal_hubble,
            scale_factor,
            parameters,
            background,
            perturbations,
        )
        expected_current = clock_interaction_current_divergence_perturbation(
            wavenumber,
            parameters.h_GeV_inv,
            field,
            theta_prime,
            bare_density,
            bare_density * perturbations["velocity_m"],
            clock_velocity,
        )
        self.assertNotEqual(expected_current, 0.0j)
        np.testing.assert_allclose(
            actual["clock_current_divergence_perturbation_GeV3"],
            expected_current,
            rtol=1e-13,
            atol=1e-14,
        )

        metric = actual["metric"]
        expected_components = (
            coupled_pressureless_matter_stress_moments(
                bare_density,
                perturbations["delta_m"],
                perturbations["velocity_m"],
                field,
                perturbations["delta_N_GeV"],
                parameters,
                clock_velocity_projection=clock_velocity,
            ),
            canonical_scalar_stress_moments(
                field_prime,
                perturbations["delta_N_GeV"],
                perturbations["delta_N_prime_GeV2"],
                scale_factor,
                metric["psi"],
                wavenumber,
                potential_derivative_GeV3=potential_derivative(
                    field, parameters
                ),
            ),
            canonical_scalar_stress_moments(
                theta_prime,
                perturbations["delta_Theta_GeV"],
                perturbations["delta_Theta_prime_GeV2"],
                scale_factor,
                metric["psi"],
                wavenumber,
                kinetic_normalization=parameters.Z_theta,
            ),
        )
        expected_total = sum_scalar_stress_moments(*expected_components)
        for key, expected in expected_total.items():
            np.testing.assert_allclose(
                actual["stress_moments"][key], expected, rtol=1e-13, atol=1e-14
            )
        expected_metric = newtonian_gauge_metric_constraints(
            wavenumber,
            conformal_hubble,
            scale_factor,
            parameters.Mpl_GeV,
            expected_total["delta_rho_GeV4"],
            expected_total["longitudinal_flux_GeV4"],
            expected_total["longitudinal_anisotropic_stress_GeV4"],
        )
        for key, expected in expected_metric.items():
            np.testing.assert_allclose(
                metric[key], expected, rtol=1e-12, atol=1e-14
            )

        expected_dust_rhs = coupled_pressureless_matter_scalar_rhs(
            perturbations["delta_m"],
            perturbations["velocity_m"],
            wavenumber,
            conformal_hubble,
            metric["psi"],
            metric["phi_prime_GeV"],
            field,
            field_prime,
            perturbations["delta_N_GeV"],
            parameters,
            clock_velocity_projection=clock_velocity,
            clock_velocity_prime_GeV=clock_velocity_prime,
        )
        np.testing.assert_allclose(
            actual["rhs"]["dust"]["velocity_m_prime_GeV"],
            expected_dust_rhs[1],
            rtol=1e-13,
            atol=1e-14,
        )
        expected_clock_rhs = clock_field_perturbation_rhs(
            perturbations["delta_Theta_GeV"],
            perturbations["delta_Theta_prime_GeV2"],
            wavenumber,
            conformal_hubble,
            scale_factor,
            theta_prime,
            background["Theta_second_prime_GeV3"],
            parameters.Z_theta,
            metric["psi"],
            metric["phi_prime_GeV"],
            metric["phi_prime_GeV"],
            expected_current,
        )
        np.testing.assert_allclose(
            actual["rhs"]["Theta"]["delta_Theta_second_prime_GeV3"],
            expected_clock_rhs[1],
            rtol=1e-13,
            atol=1e-14,
        )

    def test_h_zero_dust_scalar_mode_assembles_constraints_and_rhs(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.4,
            lambda_N=0.2,
            g_GeV_inv=0.08,
            h_GeV_inv=0.0,
            Z_theta=1.4,
            Mpl_GeV=1.3,
        )
        scale_factor = 0.8
        bare_density = 0.6
        field = 0.2
        field_prime = 0.03
        theta_prime = 0.12
        rho_total = (
            (1.0 + parameters.eta_GeV_inv * field) * bare_density
            + 0.5 * (field_prime / scale_factor) ** 2
            + potential(field, parameters)
            + 0.5
            * parameters.Z_theta
            * (theta_prime / scale_factor) ** 2
        )
        conformal_hubble = scale_factor * np.sqrt(
            rho_total / (3.0 * parameters.Mpl_GeV**2)
        )
        background_source = isotropic_fluid_scalar_source(
            bare_density, 0.0, parameters
        )
        background = {
            "rho_m_bare_GeV4": bare_density,
            "N_GeV": field,
            "N_prime_GeV2": field_prime,
            "N_second_prime_GeV3": (
                -2.0 * conformal_hubble * field_prime
                - scale_factor**2
                * (potential_derivative(field, parameters) + background_source)
            ),
            "Theta_prime_GeV2": theta_prime,
            "Theta_second_prime_GeV3": -2.0 * conformal_hubble * theta_prime,
        }
        perturbations = {
            "delta_m": 0.04 - 0.01j,
            "velocity_m": -0.03 + 0.02j,
            "delta_N_GeV": 0.012 - 0.003j,
            "delta_N_prime_GeV2": -0.02 + 0.005j,
            "delta_Theta_GeV": 0.015 + 0.002j,
            "delta_Theta_prime_GeV2": -0.01 + 0.004j,
        }
        wavenumber = 0.9

        actual = assemble_h_zero_dust_scalar_mode(
            wavenumber,
            conformal_hubble,
            scale_factor,
            parameters,
            background,
            perturbations,
        )
        metric = actual["metric"]
        expected_components = (
            coupled_pressureless_matter_stress_moments(
                bare_density,
                perturbations["delta_m"],
                perturbations["velocity_m"],
                field,
                perturbations["delta_N_GeV"],
                parameters,
            ),
            canonical_scalar_stress_moments(
                field_prime,
                perturbations["delta_N_GeV"],
                perturbations["delta_N_prime_GeV2"],
                scale_factor,
                metric["psi"],
                wavenumber,
                potential_derivative_GeV3=potential_derivative(
                    field, parameters
                ),
            ),
            canonical_scalar_stress_moments(
                theta_prime,
                perturbations["delta_Theta_GeV"],
                perturbations["delta_Theta_prime_GeV2"],
                scale_factor,
                metric["psi"],
                wavenumber,
                kinetic_normalization=parameters.Z_theta,
            ),
        )
        expected_total = sum_scalar_stress_moments(*expected_components)
        for key, expected in expected_total.items():
            np.testing.assert_allclose(
                actual["stress_moments"][key], expected, rtol=1e-13, atol=1e-14
            )
        expected_metric = newtonian_gauge_metric_constraints(
            wavenumber,
            conformal_hubble,
            scale_factor,
            parameters.Mpl_GeV,
            expected_total["delta_rho_GeV4"],
            expected_total["longitudinal_flux_GeV4"],
            expected_total["longitudinal_anisotropic_stress_GeV4"],
        )
        for key, expected in expected_metric.items():
            np.testing.assert_allclose(
                metric[key], expected, rtol=1e-12, atol=1e-14
            )

        expected_dust_rhs = coupled_pressureless_matter_scalar_rhs(
            perturbations["delta_m"],
            perturbations["velocity_m"],
            wavenumber,
            conformal_hubble,
            metric["psi"],
            metric["phi_prime_GeV"],
            field,
            field_prime,
            perturbations["delta_N_GeV"],
            parameters,
        )
        np.testing.assert_allclose(
            actual["rhs"]["dust"]["delta_m_prime_GeV"],
            expected_dust_rhs[0],
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["rhs"]["dust"]["velocity_m_prime_GeV"],
            expected_dust_rhs[1],
            rtol=1e-13,
            atol=1e-14,
        )
        expected_scalar_source = coupled_pressureless_matter_scalar_source_perturbation(
            bare_density, perturbations["delta_m"], parameters
        )
        self.assertEqual(
            actual["scalar_source_perturbation_GeV3"],
            expected_scalar_source,
        )
        expected_field_rhs = scalar_field_perturbation_rhs(
            perturbations["delta_N_GeV"],
            perturbations["delta_N_prime_GeV2"],
            wavenumber,
            conformal_hubble,
            scale_factor,
            parameters.m_phi_GeV**2
            + 3.0 * parameters.lambda_N * field**2,
            field_prime,
            background["N_second_prime_GeV3"],
            metric["psi"],
            metric["phi_prime_GeV"],
            metric["phi_prime_GeV"],
            expected_scalar_source,
        )
        for key, expected in zip(
            ("delta_N_prime_GeV2", "delta_N_second_prime_GeV3"),
            expected_field_rhs,
        ):
            np.testing.assert_allclose(
                actual["rhs"]["N"][key], expected, rtol=1e-13, atol=1e-14
            )
        expected_clock_rhs = clock_field_perturbation_rhs(
            perturbations["delta_Theta_GeV"],
            perturbations["delta_Theta_prime_GeV2"],
            wavenumber,
            conformal_hubble,
            scale_factor,
            theta_prime,
            background["Theta_second_prime_GeV3"],
            parameters.Z_theta,
            metric["psi"],
            metric["phi_prime_GeV"],
            metric["phi_prime_GeV"],
            0.0j,
        )
        for key, expected in zip(
            ("delta_Theta_prime_GeV2", "delta_Theta_second_prime_GeV3"),
            expected_clock_rhs,
        ):
            np.testing.assert_allclose(
                actual["rhs"]["Theta"][key],
                expected,
                rtol=1e-13,
                atol=1e-14,
            )
        self.assertEqual(
            actual["clock_current_divergence_perturbation_GeV3"], 0.0j
        )

    def test_group_velocities_recover_photon_and_massless_limits(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        field_value = 0.1
        momentum = 0.7

        self.assertEqual(
            photon_group_velocity_local(field_value, parameters),
            photon_speed_ratio(field_value, parameters),
        )
        self.assertAlmostEqual(
            neutrino_group_velocity_local(
                momentum, 0.0, field_value, parameters
            ),
            1.0 - parameters.h_GeV_inv * field_value,
            places=12,
        )

    def test_group_velocities_broadcast_over_background_and_momentum(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        field_history = np.array([0.05, 0.1])
        momentum_grid = np.array([0.1, 0.3, 0.7])
        mass_grid = np.array([0.02, 0.15, 0.4])

        photon_velocities = photon_group_velocity_local(
            field_history, parameters
        )
        neutrino_velocities = neutrino_group_velocity_local(
            momentum_grid[np.newaxis, :],
            mass_grid,
            field_history[:, np.newaxis],
            parameters,
        )

        expected_photons = np.array(
            [photon_speed_ratio(field, parameters) for field in field_history]
        )
        expected_neutrinos = np.array(
            [
                [
                    neutrino_group_velocity_local(
                        momentum, mass, field, parameters
                    )
                    for momentum, mass in zip(momentum_grid, mass_grid)
                ]
                for field in field_history
            ]
        )
        np.testing.assert_allclose(photon_velocities, expected_photons)
        np.testing.assert_allclose(neutrino_velocities, expected_neutrinos)

    def test_collisionless_fourier_transport_has_streaming_force_and_collision(self):
        delta_distribution = 0.2 + 0.3j
        wavenumber = 2.5
        direction_cosine = -0.4
        group_velocity = 0.8
        delta_hamiltonian = 0.06 - 0.02j
        distribution_gradient = -0.7
        collision_term = 0.01 + 0.04j

        rhs = collisionless_fourier_transport_rhs(
            delta_distribution,
            wavenumber,
            direction_cosine,
            group_velocity,
            delta_hamiltonian,
            distribution_gradient,
            collision_term,
        )
        expected = (
            -1j
            * wavenumber
            * direction_cosine
            * group_velocity
            * delta_distribution
            + 1j
            * wavenumber
            * direction_cosine
            * delta_hamiltonian
            * distribution_gradient
            + collision_term
        )
        np.testing.assert_allclose(rhs, expected, rtol=0.0, atol=1e-15)

        homogeneous_rhs = collisionless_fourier_transport_rhs(
            delta_distribution,
            0.0,
            direction_cosine,
            group_velocity,
            delta_hamiltonian,
            distribution_gradient,
            collision_term,
        )
        np.testing.assert_allclose(
            homogeneous_rhs, collision_term, rtol=0.0, atol=1e-15
        )

    def test_standard_thomson_intensity_matches_unpolarized_angular_kernel(self):
        intensity = np.array(
            [0.2 + 0.03j, -0.04j, 0.05 - 0.02j, 0.01 + 0.02j]
        )
        polarization = np.zeros_like(intensity)
        opacity = 2.3
        collision_intensity, collision_polarization = (
            standard_thomson_scalar_collision_multipoles(
                intensity,
                polarization,
                0.7,
                -0.4,
                opacity,
                0.0,
            )
        )
        cosines, weights = np.polynomial.legendre.leggauss(48)
        intensity_coefficients = [
            (2 * ell + 1) * moment
            for ell, moment in enumerate(intensity)
        ]
        angular_intensity = np.polynomial.legendre.legval(
            cosines, intensity_coefficients
        )
        angular_gain = intensity[0] + 0.5 * intensity[2] * np.polynomial.legendre.legval(
            cosines, [0.0, 0.0, 1.0]
        )
        angular_collision = opacity * (-angular_intensity + angular_gain)
        projected = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * angular_collision
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                )
                for ell in range(len(intensity))
            ]
        )

        np.testing.assert_allclose(
            collision_intensity, projected, rtol=0.0, atol=1e-14
        )
        self.assertEqual(collision_intensity[0], 0.0)
        self.assertAlmostEqual(
            collision_polarization[0], opacity * intensity[2] / 2.0
        )
        self.assertAlmostEqual(
            collision_polarization[2], opacity * intensity[2] / 10.0
        )

    def test_standard_thomson_collision_vanishes_in_baryon_frame(self):
        momentum = np.array([0.2, 0.7])
        distribution_derivative = np.array([-0.4, -0.1])
        baryon_velocity = 0.03 - 0.02j
        intensity = np.zeros((2, 4), dtype=complex)
        intensity[:, 0] = [0.1 + 0.02j, -0.03j]
        intensity[:, 1] = (
            -momentum * distribution_derivative * baryon_velocity / 3.0
        )
        polarization = np.zeros_like(intensity)

        collision_intensity, collision_polarization = (
            standard_thomson_scalar_collision_multipoles(
                intensity,
                polarization,
                momentum,
                distribution_derivative,
                np.array([1.4, 0.8]),
                baryon_velocity,
            )
        )

        np.testing.assert_allclose(
            collision_intensity, 0.0, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            collision_polarization, 0.0, rtol=0.0, atol=1e-15
        )

    def test_standard_thomson_polarization_couples_through_total_quadrupole(self):
        intensity = np.array([0.1, 0.02j, 0.07 - 0.01j, -0.03j])
        polarization = np.array([0.03 - 0.02j, 0.01j, -0.015 + 0.04j, 0.02])
        opacity = 1.8
        quadrupole_source = intensity[2] + polarization[0] + polarization[2]

        collision_intensity, collision_polarization = (
            standard_thomson_scalar_collision_multipoles(
                intensity,
                polarization,
                0.6,
                -0.2,
                opacity,
                0.0,
            )
        )

        expected_intensity = -opacity * intensity
        expected_intensity[0] = 0.0
        expected_intensity[2] = -opacity * (
            intensity[2] - quadrupole_source / 10.0
        )
        expected_polarization = -opacity * polarization
        expected_polarization[0] = -opacity * (
            polarization[0] - quadrupole_source / 2.0
        )
        expected_polarization[2] = -opacity * (
            polarization[2] - quadrupole_source / 10.0
        )

        np.testing.assert_allclose(
            collision_intensity, expected_intensity, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            collision_polarization, expected_polarization, rtol=0.0, atol=1e-15
        )

    def test_standard_thomson_photon_rhs_composes_q_grid_blocks(self):
        intensity = np.array(
            [
                [0.04, 0.015, -0.008, 0.003],
                [0.02j, -0.01j, 0.006, 0.002j],
            ],
            dtype=complex,
        )
        polarization = np.array(
            [
                [0.002, 0.001j, -0.0005, 0.0002],
                [0.001j, -0.0007, 0.0003j, -0.0001],
            ],
            dtype=complex,
        )
        wavenumber = np.array([0.8, 1.1])
        group_velocity = np.array([0.99, 0.94])
        radial_gradient = np.array([-0.4, -0.2])
        hamiltonian_monopole = np.array([0.003j, -0.002])
        hamiltonian_dipole = np.array([0.001, -0.001j])
        next_intensity = np.array([0.001j, 0.0004])
        next_polarization = np.array([-0.0002, 0.0003j])
        momenta = np.array([0.3, 0.7])
        distribution_derivative = np.array([-0.6, -0.2])
        opacity = np.array([1.2, 0.7])
        baryon_velocity = 0.004 - 0.002j

        actual_intensity_rhs, actual_polarization_rhs = (
            standard_thomson_scalar_photon_mode_rhs(
                intensity,
                polarization,
                wavenumber,
                group_velocity,
                radial_gradient,
                hamiltonian_monopole,
                hamiltonian_dipole,
                next_intensity,
                next_polarization,
                momenta,
                distribution_derivative,
                opacity,
                baryon_velocity,
            )
        )
        intensity_collisions, polarization_collisions = (
            standard_thomson_scalar_collision_multipoles(
                intensity,
                polarization,
                momenta,
                distribution_derivative,
                opacity,
                baryon_velocity,
            )
        )
        expected_intensity_rhs = scalar_mode_multipole_rhs(
            intensity,
            wavenumber,
            group_velocity,
            radial_gradient,
            hamiltonian_monopole,
            hamiltonian_dipole,
            next_intensity,
            intensity_collisions,
        )
        expected_polarization_rhs = collisionless_multipole_streaming_rhs(
            polarization,
            wavenumber,
            group_velocity,
            polarization_collisions,
            next_polarization,
        )

        np.testing.assert_allclose(
            actual_intensity_rhs, expected_intensity_rhs, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            actual_polarization_rhs,
            expected_polarization_rhs,
            rtol=0.0,
            atol=1e-15,
        )

    def test_standard_thomson_photon_baryon_rhs_conserves_momentum(self):
        intensity = np.array(
            [
                [0.03, 0.04, 0.015, -0.004],
                [0.02, 0.025, 0.008, 0.002],
                [0.01, 0.012, -0.003, 0.001],
            ],
            dtype=complex,
        )
        polarization = np.zeros_like(intensity)
        momenta = np.array([0.25, 0.55, 0.9])
        weights = np.array([0.06, 0.09, 0.04])
        distribution_derivative = np.array([-0.5, -0.3, -0.12])
        opacity = np.array([1.3, 0.8, 0.4])
        scale_factor = 0.71
        baryon_density = 0.024
        baryon_velocity = 0.003 - 0.001j
        baryon_density_contrast = 0.02 + 0.01j
        wavenumber = 1.2
        conformal_hubble = 0.06
        psi = 2e-5 - 0.5e-5j
        phi_prime = -1e-6 + 0.3e-6j
        sound_speed_squared = 1e-4
        group_velocity = np.array([0.99, 0.97, 0.93])
        radial_gradient = np.array([-0.2, -0.1, -0.04])
        hamiltonian_monopole = np.array([1e-5, 0.8e-5, 0.5e-5])
        hamiltonian_dipole = np.array([0.3e-5j, 0.2e-5j, 0.1e-5j])
        next_intensity = np.array([0.001, 0.0008, 0.0004])
        next_polarization = np.zeros_like(momenta, dtype=complex)

        actual = standard_thomson_photon_baryon_mode_rhs(
            intensity,
            polarization,
            wavenumber,
            group_velocity,
            radial_gradient,
            hamiltonian_monopole,
            hamiltonian_dipole,
            next_intensity,
            next_polarization,
            momenta,
            weights,
            distribution_derivative,
            opacity,
            scale_factor,
            baryon_density,
            baryon_density_contrast,
            baryon_velocity,
            conformal_hubble,
            psi,
            phi_prime,
            sound_speed_squared=sound_speed_squared,
        )
        intensity_collisions, polarization_collisions = (
            standard_thomson_scalar_collision_multipoles(
                intensity,
                polarization,
                momenta,
                distribution_derivative,
                opacity,
                baryon_velocity,
            )
        )
        expected_photon_transfer = standard_thomson_baryon_velocity_collision_source(
            momenta,
            weights,
            intensity_collisions,
            scale_factor,
            baryon_density,
        )
        expected_baryon_density_rhs, expected_baryon_velocity_rhs = (
            standard_baryon_scalar_rhs(
                baryon_density_contrast,
                baryon_velocity,
                wavenumber,
                conformal_hubble,
                psi,
                phi_prime,
                expected_photon_transfer["baryon_velocity_source_GeV"],
                sound_speed_squared,
            )
        )

        np.testing.assert_allclose(
            actual["intensity_multipole_rhs"],
            scalar_mode_multipole_rhs(
                intensity,
                wavenumber,
                group_velocity,
                radial_gradient,
                hamiltonian_monopole,
                hamiltonian_dipole,
                next_intensity,
                intensity_collisions,
            ),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            actual["polarization_multipole_rhs"],
            collisionless_multipole_streaming_rhs(
                polarization,
                wavenumber,
                group_velocity,
                polarization_collisions,
                next_polarization,
            ),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(
            actual["baryon_density_contrast_rhs"],
            expected_baryon_density_rhs,
            places=15,
        )
        self.assertAlmostEqual(
            actual["baryon_velocity_rhs"], expected_baryon_velocity_rhs, places=15
        )
        self.assertAlmostEqual(
            actual["photon_momentum_transfer_GeV5"]
            + baryon_density * actual["baryon_velocity_collision_source_GeV"],
            0.0,
            places=15,
        )

    def test_thomson_baryon_source_conserves_photon_baryon_momentum(self):
        momenta = np.array([0.2, 0.5, 0.9])
        weights = np.array([0.04, 0.1, 0.08])
        intensity = np.zeros((3, 4), dtype=complex)
        intensity[:, 1] = [0.08, 0.05, 0.03]
        polarization = np.zeros_like(intensity)
        opacity = np.array([1.4, 0.9, 0.6])
        scale_factor = 0.73
        baryon_density = 0.021
        photon_degeneracy = 2.0
        collision_intensity, _ = standard_thomson_scalar_collision_multipoles(
            intensity,
            polarization,
            momenta,
            np.zeros_like(momenta),
            opacity,
            0.0,
        )
        momentum_sources = standard_thomson_baryon_velocity_collision_source(
            momenta,
            weights,
            collision_intensity,
            scale_factor,
            baryon_density,
            photon_degeneracy,
        )
        expected_photon_transfer = (
            photon_degeneracy
            / (2.0 * np.pi**2 * scale_factor**4)
            * np.sum(weights * momenta**3 * collision_intensity[:, 1])
        )

        self.assertGreater(momentum_sources["baryon_velocity_source_GeV"].real, 0.0)
        self.assertAlmostEqual(
            momentum_sources["photon_momentum_transfer_GeV5"],
            expected_photon_transfer,
            places=15,
        )
        self.assertAlmostEqual(
            expected_photon_transfer
            + baryon_density
            * momentum_sources["baryon_velocity_source_GeV"],
            0.0,
            places=15,
        )

    def test_finite_h_thomson_baryon_rhs_composes_standard_collisions(self):
        intensity = np.array(
            [
                [0.03, 0.04, 0.015, -0.004],
                [0.02, 0.025, 0.008, 0.002],
                [0.01, 0.012, -0.003, 0.001],
            ],
            dtype=complex,
        )
        polarization = np.array(
            [
                [0.004, -0.001, 0.002, 0.0005],
                [0.003, 0.0002, -0.001, 0.0003],
                [0.002, -0.0003, 0.0005, -0.0001],
            ],
            dtype=complex,
        )
        momenta = np.array([0.25, 0.55, 0.9])
        weights = np.array([0.06, 0.09, 0.04])
        radial_gradient = np.array([-0.2, -0.1, -0.04])
        distribution_derivative = np.array([-0.5, -0.3, -0.12])
        opacity = np.array([1.3, 0.8, 0.4])
        next_intensity = np.array([0.001, 0.0008, 0.0004])
        next_polarization = np.array([0.0002, -0.0001, 0.0003])
        N_background = 0.1
        delta_N = 0.002 - 0.0004j
        psi = 2e-5 - 0.5e-5j
        phi = 1.4e-5 + 0.3e-5j
        clock_velocity = 0.001 + 0.0003j
        scale_factor = 0.71
        baryon_density = 0.024
        baryon_density_contrast = 0.02 + 0.01j
        baryon_velocity = 0.003 - 0.001j
        wavenumber = 1.2
        conformal_hubble = 0.06
        phi_prime = -1e-6 + 0.3e-6j

        for h_value in (0.0, 0.02):
            parameters = FLRWParameters(
                m_phi_GeV=0.0,
                lambda_N=0.0,
                g_GeV_inv=0.03,
                h_GeV_inv=h_value,
            )
            use_coupled_dust = h_value != 0.0
            clock_velocity_prime = 0.004 + 0.001j
            coupled_dust_kwargs = (
                {
                    "N_background_prime_GeV2": 0.05,
                    "clock_velocity_prime_GeV": clock_velocity_prime,
                }
                if use_coupled_dust
                else {}
            )
            sound_speed_squared = 0.0 if use_coupled_dust else 1e-4
            group_velocity = photon_group_velocity_local(
                N_background, parameters
            )
            hamiltonian_monopole, hamiltonian_dipole = (
                photon_hamiltonian_mode_coefficients(
                    momenta,
                    N_background,
                    delta_N,
                    psi,
                    phi,
                    clock_velocity,
                    parameters,
                )
            )
            actual = standard_thomson_finite_h_photon_baryon_mode_rhs(
                intensity,
                polarization,
                wavenumber,
                radial_gradient,
                N_background,
                delta_N,
                psi,
                phi,
                clock_velocity,
                next_intensity,
                next_polarization,
                momenta,
                weights,
                distribution_derivative,
                opacity,
                scale_factor,
                baryon_density,
                baryon_density_contrast,
                baryon_velocity,
                conformal_hubble,
                phi_prime,
                parameters,
                sound_speed_squared=sound_speed_squared,
                **coupled_dust_kwargs,
            )
            expected = standard_thomson_photon_baryon_mode_rhs(
                intensity,
                polarization,
                wavenumber,
                group_velocity,
                radial_gradient,
                hamiltonian_monopole,
                hamiltonian_dipole,
                next_intensity,
                next_polarization,
                momenta,
                weights,
                distribution_derivative,
                finite_h_thomson_opacity_per_conformal_time(
                    opacity, N_background, parameters
                ),
                scale_factor,
                baryon_density,
                baryon_density_contrast,
                baryon_velocity,
                conformal_hubble,
                psi,
                phi_prime,
                sound_speed_squared=sound_speed_squared,
                collision_velocity_projection=(
                    finite_h_thomson_collision_velocity_projection(
                        baryon_velocity,
                        clock_velocity,
                        N_background,
                        parameters,
                    )
                ),
            )
            if use_coupled_dust:
                rest_mass_factor = (
                    1.0 + parameters.eta_GeV_inv * N_background
                )
                inertial_mass_factor = 1.0 + (
                    parameters.g_GeV_inv + parameters.h_GeV_inv
                ) * N_background
                collision_acceleration = (
                    rest_mass_factor
                    / inertial_mass_factor
                    * expected["baryon_velocity_collision_source_GeV"]
                )
                expected["baryon_density_contrast_rhs"], expected[
                    "baryon_velocity_rhs"
                ] = coupled_pressureless_matter_scalar_rhs(
                    baryon_density_contrast,
                    baryon_velocity,
                    wavenumber,
                    conformal_hubble,
                    psi,
                    phi_prime,
                    N_background,
                    0.05,
                    delta_N,
                    parameters,
                    clock_velocity_projection=clock_velocity,
                    clock_velocity_prime_GeV=clock_velocity_prime,
                    collision_velocity_source=collision_acceleration,
                )
                expected["baryon_velocity_collision_source_GeV"] = (
                    collision_acceleration
                )

            for key, expected_value in expected.items():
                np.testing.assert_allclose(
                    actual[key], expected_value, rtol=0.0, atol=1e-15
                )
            inertial_to_rest_ratio = (
                inertial_mass_factor / rest_mass_factor
                if use_coupled_dust
                else 1.0
            )
            self.assertAlmostEqual(
                actual["photon_momentum_transfer_GeV5"]
                + baryon_density
                * inertial_to_rest_ratio
                * actual["baryon_velocity_collision_source_GeV"],
                0.0,
                places=15,
            )
            if h_value == 0.0:
                np.testing.assert_allclose(group_velocity, 1.0, rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(
                    hamiltonian_monopole,
                    momenta * (psi + phi),
                    rtol=0.0,
                    atol=1e-15,
                )
                np.testing.assert_allclose(
                    hamiltonian_dipole, 0.0, rtol=0.0, atol=1e-15
                )

    def test_standard_baryon_scalar_rhs_matches_continuity_and_euler(self):
        density_contrast = 0.04 - 0.02j
        velocity = -0.003j
        wavenumber = 1.3
        conformal_hubble = 0.07
        psi = 2e-5 - 1e-5j
        phi_prime = -3e-6 + 1e-6j
        collision_source = 4e-5j
        sound_speed_squared = 2e-4

        density_rhs, velocity_rhs = standard_baryon_scalar_rhs(
            density_contrast,
            velocity,
            wavenumber,
            conformal_hubble,
            psi,
            phi_prime,
            collision_source,
            sound_speed_squared,
        )

        self.assertAlmostEqual(
            density_rhs,
            -1j * wavenumber * velocity + 3.0 * phi_prime,
            places=15,
        )
        self.assertAlmostEqual(
            velocity_rhs,
            -conformal_hubble * velocity
            - 1j * wavenumber * (psi + sound_speed_squared * density_contrast)
            + collision_source,
            places=15,
        )

        homogeneous_density_rhs, homogeneous_velocity_rhs = (
            standard_baryon_scalar_rhs(
                density_contrast,
                velocity,
                0.0,
                conformal_hubble,
                psi,
                phi_prime,
            )
        )
        self.assertAlmostEqual(homogeneous_density_rhs, 3.0 * phi_prime)
        self.assertAlmostEqual(
            homogeneous_velocity_rhs, -conformal_hubble * velocity
        )

    def test_perfect_fluid_stress_moments_feed_clock_current(self):
        energy_density = 0.021
        pressure = 0.0003
        density_contrast = 0.04 - 0.02j
        baryon_velocity = 0.007 - 0.004j
        clock_velocity = -0.002 + 0.003j
        sound_speed_squared = 0.015
        wavenumber = 0.9
        h_coupling = 0.05
        N_background = 0.4
        Theta_background_prime = 0.13

        stress_moments = perfect_fluid_scalar_stress_moments(
            energy_density,
            pressure,
            density_contrast,
            baryon_velocity,
            sound_speed_squared,
        )
        current_divergence = clock_interaction_current_divergence_perturbation(
            wavenumber,
            h_coupling,
            N_background,
            Theta_background_prime,
            energy_density + pressure,
            stress_moments["longitudinal_flux_GeV4"],
            clock_velocity,
        )

        self.assertAlmostEqual(
            stress_moments["delta_rho_GeV4"],
            energy_density * density_contrast,
            places=15,
        )
        self.assertAlmostEqual(
            stress_moments["delta_pressure_GeV4"],
            sound_speed_squared * energy_density * density_contrast,
            places=15,
        )
        self.assertAlmostEqual(
            stress_moments["longitudinal_flux_GeV4"],
            (energy_density + pressure) * baryon_velocity,
            places=15,
        )
        self.assertEqual(
            stress_moments["longitudinal_anisotropic_stress_GeV4"], 0.0j
        )
        self.assertAlmostEqual(
            current_divergence,
            -1j
            * wavenumber
            * h_coupling
            * N_background
            / Theta_background_prime
            * (energy_density + pressure)
            * (baryon_velocity - clock_velocity),
            places=15,
        )

    def test_multipole_streaming_matches_direct_angular_projection(self):
        multipoles = np.array([0.2 + 0.1j, -0.03j, 0.04 + 0.02j])
        next_multipole = -0.01 + 0.02j
        sources = np.array([0.01j, -0.03 + 0.01j, 0.02])
        wavenumber = 1.7
        group_velocity = 0.63
        maximum_ell = len(multipoles) - 1
        cosines, weights = np.polynomial.legendre.leggauss(48)
        distribution_coefficients = [
            (2 * ell + 1) * moment
            for ell, moment in enumerate(multipoles)
        ]
        distribution_coefficients.append(
            (2 * (maximum_ell + 1) + 1) * next_multipole
        )
        source_coefficients = [
            (2 * ell + 1) * source
            for ell, source in enumerate(sources)
        ]
        distribution = np.polynomial.legendre.legval(
            cosines, distribution_coefficients
        )
        source = np.polynomial.legendre.legval(cosines, source_coefficients)
        angular_rhs = collisionless_fourier_transport_rhs(
            distribution,
            wavenumber,
            cosines,
            group_velocity,
            0.0,
            0.0,
            source,
        )
        projected = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * angular_rhs
                    * np.polynomial.legendre.legval(
                        cosines,
                        [0.0] * ell + [1.0],
                    )
                )
                for ell in range(maximum_ell + 1)
            ]
        )
        hierarchy_rhs = collisionless_multipole_streaming_rhs(
            multipoles,
            wavenumber,
            group_velocity,
            sources,
            next_multipole,
        )

        np.testing.assert_allclose(
            hierarchy_rhs, projected, rtol=0.0, atol=1e-14
        )

    def test_free_streaming_bessel_closure_matches_plane_wave_projection(self):
        maximum_ell = 5
        wavenumber = 0.8
        free_streaming_phase = 14.0
        free_streaming_distance = free_streaming_phase / wavenumber
        cosines, weights = np.polynomial.legendre.leggauss(96)
        plane_wave = np.exp(-1j * free_streaming_phase * cosines)
        multipoles = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * plane_wave
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                )
                for ell in range(maximum_ell + 1)
            ]
        )
        expected_next_multipole = 0.5 * np.sum(
            weights
            * plane_wave
            * np.polynomial.legendre.legval(
                cosines, [0.0] * (maximum_ell + 1) + [1.0]
            )
        )

        actual_next_multipole = free_streaming_bessel_closure(
            multipoles,
            wavenumber,
            free_streaming_distance,
        )

        self.assertAlmostEqual(
            actual_next_multipole, expected_next_multipole, places=14
        )

    def test_free_streaming_bessel_closure_broadcasts_and_smoothly_activates(self):
        maximum_ell = 3
        wavenumber = 1.25
        phases = np.array([0.0, 3.0, 4.5, 6.0])
        multipoles = np.array(
            [
                [0.1, -0.02j, 0.03, 0.004j],
                [0.08j, 0.01, -0.02j, 0.003],
                [0.04, 0.005j, -0.01, 0.002j],
                [0.03j, -0.004, 0.008j, 0.001],
            ],
            dtype=complex,
        )

        actual = free_streaming_bessel_closure(
            multipoles,
            wavenumber,
            phases / wavenumber,
        )
        middle_recurrence = multipoles[2, -2] - (
            1j * (2 * maximum_ell + 1) * multipoles[2, -1] / phases[2]
        )
        final_recurrence = multipoles[3, -2] - (
            1j * (2 * maximum_ell + 1) * multipoles[3, -1] / phases[3]
        )

        np.testing.assert_allclose(
            actual,
            np.array(
                [0.0j, 0.0j, 0.5 * middle_recurrence, final_recurrence]
            ),
            rtol=0.0,
            atol=1e-15,
        )

    def test_free_streaming_distance_history_integrates_time_first_momenta(self):
        log_scale_factor = np.array([0.0, 0.2, 0.45, 0.8])
        conformal_hubble = np.full(log_scale_factor.shape, 2.5)
        group_velocity = np.array(
            [
                [0.9, 0.6],
                [0.9, 0.6],
                [0.9, 0.6],
                [0.9, 0.6],
            ]
        )

        distance = free_streaming_distance_history(
            log_scale_factor,
            conformal_hubble,
            group_velocity,
        )

        expected = (
            log_scale_factor[:, np.newaxis]
            * group_velocity[0]
            / conformal_hubble[0]
        )
        np.testing.assert_allclose(distance, expected, rtol=0.0, atol=1e-15)

    def test_bessel_closed_streaming_rhs_matches_plane_wave_derivative(self):
        maximum_ell = 5
        wavenumber = 0.8
        group_velocity = 0.63
        phase = 14.0
        free_streaming_distance = phase / wavenumber
        cosines, weights = np.polynomial.legendre.leggauss(96)
        plane_wave = np.exp(-1j * phase * cosines)
        multipoles = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * plane_wave
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                )
                for ell in range(maximum_ell + 1)
            ]
        )
        expected_rhs = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * (-1j * wavenumber * group_velocity * cosines)
                    * plane_wave
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                )
                for ell in range(maximum_ell + 1)
            ]
        )

        actual_rhs = free_streaming_multipole_streaming_rhs(
            multipoles,
            wavenumber,
            group_velocity,
            np.zeros_like(multipoles),
            free_streaming_distance,
        )

        np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=0.0, atol=1e-14)

    def test_integrate_free_streaming_multipoles_matches_plane_wave(self):
        log_scale_factor = np.linspace(0.0, 1.5, 9)
        maximum_ell = 9
        wavenumber = 0.7
        group_velocity = 0.8
        conformal_hubble_value = 2.0
        conformal_hubble = np.full(log_scale_factor.shape, conformal_hubble_value)
        velocities = np.full(log_scale_factor.shape, group_velocity)
        initial_multipoles = np.zeros(maximum_ell + 1, dtype=complex)
        initial_multipoles[0] = 1.0
        sources = np.zeros(
            (log_scale_factor.size, maximum_ell + 1), dtype=complex
        )

        result = integrate_free_streaming_multipoles(
            log_scale_factor,
            initial_multipoles,
            wavenumber,
            conformal_hubble,
            velocities,
            sources,
            rtol=1e-11,
            atol=1e-13,
        )

        phase = (
            wavenumber
            * group_velocity
            / conformal_hubble_value
            * (log_scale_factor[-1] - log_scale_factor[0])
        )
        cosines, weights = np.polynomial.legendre.leggauss(96)
        plane_wave = np.exp(-1j * phase * cosines)
        expected = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * plane_wave
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                )
                for ell in range(maximum_ell + 1)
            ]
        )

        self.assertTrue(result["solution"].success, result["solution"].message)
        self.assertEqual(
            result["multipoles"].shape,
            (log_scale_factor.size, maximum_ell + 1),
        )
        np.testing.assert_allclose(
            result["multipoles"][-1], expected, rtol=0.0, atol=2e-12
        )
        self.assertAlmostEqual(
            result["free_streaming_distance_GeVinv"][-1],
            group_velocity
            / conformal_hubble_value
            * (log_scale_factor[-1] - log_scale_factor[0]),
            places=15,
        )

    def test_integrate_free_streaming_multipoles_integrates_constant_source(self):
        log_scale_factor = np.linspace(0.0, 0.8, 5)
        conformal_hubble_value = 2.5
        conformal_hubble = np.full(log_scale_factor.shape, conformal_hubble_value)
        velocities = np.full(log_scale_factor.shape, 0.7)
        initial_multipoles = np.array([1.0 + 0.2j, 0.0j, 0.0j])
        source = 0.4 + 0.1j
        sources = np.zeros(
            (log_scale_factor.size, initial_multipoles.size), dtype=complex
        )
        sources[:, 0] = source

        result = integrate_free_streaming_multipoles(
            log_scale_factor,
            initial_multipoles,
            0.0,
            conformal_hubble,
            velocities,
            sources,
        )

        expected_monopole = initial_multipoles[0] + source * (
            log_scale_factor - log_scale_factor[0]
        ) / conformal_hubble_value
        self.assertTrue(result["solution"].success, result["solution"].message)
        np.testing.assert_allclose(
            result["multipoles"][:, 0], expected_monopole, rtol=0.0, atol=1e-13
        )

    def test_integrate_free_streaming_multipoles_tracks_momentum_grid(self):
        log_scale_factor = np.linspace(0.0, 0.5, 5)
        maximum_ell = 7
        wavenumber = 0.7
        conformal_hubble_value = 1.5
        velocities_at_momenta = np.array([0.3, 0.8])
        conformal_hubble = np.full(log_scale_factor.shape, conformal_hubble_value)
        velocity_history = np.broadcast_to(
            velocities_at_momenta,
            (log_scale_factor.size, velocities_at_momenta.size),
        )
        initial_multipoles = np.zeros(
            (velocities_at_momenta.size, maximum_ell + 1), dtype=complex
        )
        initial_multipoles[:, 0] = 1.0
        sources = np.zeros(
            (log_scale_factor.size,) + initial_multipoles.shape, dtype=complex
        )

        result = integrate_free_streaming_multipoles(
            log_scale_factor,
            initial_multipoles,
            wavenumber,
            conformal_hubble,
            velocity_history,
            sources,
            rtol=1e-11,
            atol=1e-13,
        )

        cosines, weights = np.polynomial.legendre.leggauss(96)
        expected = np.empty_like(initial_multipoles)
        for momentum_index, velocity in enumerate(velocities_at_momenta):
            phase = (
                wavenumber
                * velocity
                / conformal_hubble_value
                * (log_scale_factor[-1] - log_scale_factor[0])
            )
            plane_wave = np.exp(-1j * phase * cosines)
            expected[momentum_index] = [
                0.5
                * np.sum(
                    weights
                    * plane_wave
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                )
                for ell in range(maximum_ell + 1)
            ]

        self.assertTrue(result["solution"].success, result["solution"].message)
        np.testing.assert_allclose(
            result["multipoles"][-1], expected, rtol=0.0, atol=2e-12
        )

    def test_scalar_e_polarization_streaming_matches_spin_two_projection(self):
        polarization_multipoles = np.array(
            [0.08 + 0.02j, -0.03j, 0.04 - 0.01j]
        )
        next_polarization_multipole = -0.02 + 0.015j
        wavenumber = 1.4
        group_velocity = 0.82
        maximum_ell = len(polarization_multipoles) - 1
        cosines, weights = np.polynomial.legendre.leggauss(48)
        coefficients = [
            (2 * ell + 1) * moment
            for ell, moment in enumerate(polarization_multipoles)
        ]
        coefficients.append(
            (2 * (maximum_ell + 1) + 1) * next_polarization_multipole
        )
        scalar_e_polarization = (1.0 - cosines**2) * np.polynomial.legendre.legval(
            cosines, coefficients
        )
        angular_rhs = (
            -1j * wavenumber * group_velocity * cosines * scalar_e_polarization
        )
        projected = np.asarray(
            [
                0.5
                * np.sum(
                    weights
                    * np.polynomial.legendre.legval(
                        cosines, [0.0] * ell + [1.0]
                    )
                    * angular_rhs
                    / (1.0 - cosines**2)
                )
                for ell in range(maximum_ell + 1)
            ]
        )
        hierarchy_rhs = collisionless_multipole_streaming_rhs(
            polarization_multipoles,
            wavenumber,
            group_velocity,
            np.zeros_like(polarization_multipoles),
            next_polarization_multipole,
        )

        np.testing.assert_allclose(
            hierarchy_rhs, projected, rtol=0.0, atol=1e-14
        )

    def test_multipole_streaming_broadcasts_over_momentum_grid(self):
        multipoles = np.array(
            [
                [0.2 + 0.1j, -0.03j, 0.04 + 0.02j],
                [0.1 - 0.04j, 0.02 + 0.01j, -0.03j],
            ]
        )
        next_multipoles = np.array([-0.01 + 0.02j, 0.03 - 0.01j])
        sources = np.array(
            [
                [0.01j, -0.03 + 0.01j, 0.02],
                [0.02 - 0.01j, 0.01j, -0.04],
            ]
        )
        wavenumbers = np.array([1.7, 0.9])
        velocities = np.array([0.63, 0.81])
        maximum_ell = multipoles.shape[-1] - 1
        cosines, weights = np.polynomial.legendre.leggauss(48)
        projected = []
        for index in range(len(wavenumbers)):
            distribution_coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(multipoles[index])
            ]
            distribution_coefficients.append(
                (2 * (maximum_ell + 1) + 1) * next_multipoles[index]
            )
            source_coefficients = [
                (2 * ell + 1) * source
                for ell, source in enumerate(sources[index])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, distribution_coefficients
            )
            source = np.polynomial.legendre.legval(
                cosines, source_coefficients
            )
            angular_rhs = collisionless_fourier_transport_rhs(
                distribution,
                wavenumbers[index],
                cosines,
                velocities[index],
                0.0,
                0.0,
                source,
            )
            projected.append(
                [
                    0.5
                    * np.sum(
                        weights
                        * angular_rhs
                        * np.polynomial.legendre.legval(
                            cosines, [0.0] * ell + [1.0]
                        )
                    )
                    for ell in range(maximum_ell + 1)
                ]
            )
        hierarchy_rhs = collisionless_multipole_streaming_rhs(
            multipoles,
            wavenumbers,
            velocities,
            sources,
            next_multipoles,
        )

        np.testing.assert_allclose(
            hierarchy_rhs, projected, rtol=0.0, atol=1e-14
        )

    def test_hamiltonian_force_multipoles_match_direct_angular_projection(self):
        wavenumber = 1.7
        radial_gradient = np.array([-0.6, -0.2])
        hamiltonian_monopole = np.array([0.12 + 0.03j, -0.04 + 0.01j])
        hamiltonian_dipole = np.array([-0.02j, 0.07 + 0.02j])
        maximum_ell = 4
        cosines, weights = np.polynomial.legendre.leggauss(48)
        force_multipoles = collisionless_hamiltonian_force_multipoles(
            wavenumber,
            radial_gradient,
            hamiltonian_monopole,
            hamiltonian_dipole,
            maximum_ell,
        )
        projected = []
        for index in range(len(radial_gradient)):
            angular_source = (
                1j
                * wavenumber
                * cosines
                * radial_gradient[index]
                * (
                    hamiltonian_monopole[index]
                    + hamiltonian_dipole[index] * cosines
                )
            )
            projected.append(
                [
                    0.5
                    * np.sum(
                        weights
                        * angular_source
                        * np.polynomial.legendre.legval(
                            cosines, [0.0] * ell + [1.0]
                        )
                    )
                    for ell in range(maximum_ell + 1)
                ]
            )

        np.testing.assert_allclose(
            force_multipoles, projected, rtol=0.0, atol=1e-14
        )

    def test_scalar_mode_multipole_rhs_matches_direct_angular_transport(self):
        multipoles = np.array(
            [
                [0.2 + 0.1j, -0.03j, 0.04 + 0.02j],
                [0.1 - 0.04j, 0.02 + 0.01j, -0.03j],
            ]
        )
        next_multipoles = np.array([-0.01 + 0.02j, 0.03 - 0.01j])
        collision_multipoles = np.array(
            [
                [0.01j, -0.03 + 0.01j, 0.02],
                [0.02 - 0.01j, 0.01j, -0.04],
            ]
        )
        wavenumbers = np.array([1.7, 0.9])
        velocities = np.array([0.63, 0.81])
        radial_gradients = np.array([-0.6, -0.2])
        hamiltonian_monopoles = np.array(
            [0.12 + 0.03j, -0.04 + 0.01j]
        )
        hamiltonian_dipoles = np.array([-0.02j, 0.07 + 0.02j])
        maximum_ell = multipoles.shape[-1] - 1
        cosines, weights = np.polynomial.legendre.leggauss(48)
        projected = []
        for index in range(len(wavenumbers)):
            distribution_coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(multipoles[index])
            ]
            distribution_coefficients.append(
                (2 * (maximum_ell + 1) + 1) * next_multipoles[index]
            )
            collision_coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(collision_multipoles[index])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, distribution_coefficients
            )
            collision_term = np.polynomial.legendre.legval(
                cosines, collision_coefficients
            )
            hamiltonian = (
                hamiltonian_monopoles[index]
                + hamiltonian_dipoles[index] * cosines
            )
            angular_rhs = collisionless_fourier_transport_rhs(
                distribution,
                wavenumbers[index],
                cosines,
                velocities[index],
                hamiltonian,
                radial_gradients[index],
                collision_term,
            )
            projected.append(
                [
                    0.5
                    * np.sum(
                        weights
                        * angular_rhs
                        * np.polynomial.legendre.legval(
                            cosines, [0.0] * ell + [1.0]
                        )
                    )
                    for ell in range(maximum_ell + 1)
                ]
            )
        hierarchy_rhs = scalar_mode_multipole_rhs(
            multipoles,
            wavenumbers,
            velocities,
            radial_gradients,
            hamiltonian_monopoles,
            hamiltonian_dipoles,
            next_multipoles,
            collision_multipoles,
        )

        np.testing.assert_allclose(
            hierarchy_rhs, projected, rtol=0.0, atol=1e-14
        )

    def test_kinetic_stress_multipoles_match_direct_phase_space_projection(self):
        comoving_momenta = np.array([0.12, 0.37, 0.81])
        quadrature_weights = np.array([0.03, 0.11, 0.07])
        mass = 0.29
        scale_factor = 0.73
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        moment_integrals = uncoupled_kinetic_stress_multipoles(
            comoving_momenta,
            quadrature_weights,
            mass,
            scale_factor,
            moments,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        projected = {
            "delta_rho_GeV4": 0.0j,
            "delta_pressure_GeV4": 0.0j,
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for index, comoving_momentum in enumerate(comoving_momenta):
            distribution_coefficients = [
                (2 * ell + 1) * moments[index, ell]
                for ell in range(moments.shape[1])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, distribution_coefficients
            )
            physical_momentum = comoving_momentum / scale_factor
            energy = np.hypot(physical_momentum, mass)
            measure = (
                phase_space_factor
                * quadrature_weights[index]
                * comoving_momentum**2
                / scale_factor**3
            )
            angular_average = 0.5 * np.sum(angular_weights * distribution)
            angular_flux = 0.5 * np.sum(
                angular_weights * cosines * distribution
            )
            angular_shear = 0.5 * np.sum(
                angular_weights
                * (cosines**2 - 1.0 / 3.0)
                * distribution
            )
            projected["delta_rho_GeV4"] += (
                measure * energy * angular_average
            )
            projected["delta_pressure_GeV4"] += (
                measure
                * physical_momentum**2
                / (3.0 * energy)
                * angular_average
            )
            projected["longitudinal_flux_GeV4"] += (
                measure
                * physical_momentum
                * angular_flux
            )
            projected["longitudinal_anisotropic_stress_GeV4"] += (
                measure
                * physical_momentum**2
                / energy
                * angular_shear
            )

        for moment_name, expected in projected.items():
            self.assertAlmostEqual(
                moment_integrals[moment_name], expected, places=14
            )

    def test_coupled_neutrino_kinetic_moments_match_direct_projection(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        comoving_momenta = np.array([0.12, 0.37, 0.81])
        quadrature_weights = np.array([0.03, 0.11, 0.07])
        mass = 0.29
        scale_factor = 0.73
        field_value = 0.16
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        actual = neutrino_kinetic_stress_multipoles(
            comoving_momenta,
            quadrature_weights,
            mass,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        projected = {
            "delta_rho_GeV4": 0.0j,
            "delta_pressure_GeV4": 0.0j,
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for index, comoving_momentum in enumerate(comoving_momenta):
            distribution_coefficients = [
                (2 * ell + 1) * moments[index, ell]
                for ell in range(moments.shape[1])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, distribution_coefficients
            )
            physical_momentum = comoving_momentum / scale_factor
            energy = neutrino_energy_local(
                physical_momentum, mass, field_value, parameters
            )
            group_velocity = neutrino_group_velocity_local(
                physical_momentum, mass, field_value, parameters
            )
            measure = (
                phase_space_factor
                * quadrature_weights[index]
                * comoving_momentum**2
                / scale_factor**3
            )
            angular_average = 0.5 * np.sum(angular_weights * distribution)
            angular_flux = 0.5 * np.sum(
                angular_weights * cosines * distribution
            )
            angular_shear = 0.5 * np.sum(
                angular_weights
                * (cosines**2 - 1.0 / 3.0)
                * distribution
            )
            projected["delta_rho_GeV4"] += (
                measure * energy * angular_average
            )
            projected["delta_pressure_GeV4"] += (
                measure
                * physical_momentum
                * group_velocity
                * angular_average
                / 3.0
            )
            projected["longitudinal_flux_GeV4"] += (
                measure
                * physical_momentum
                * angular_flux
            )
            projected["longitudinal_anisotropic_stress_GeV4"] += (
                measure
                * physical_momentum
                * group_velocity
                * angular_shear
            )

        for moment_name, expected in projected.items():
            np.testing.assert_allclose(
                actual[moment_name], expected, rtol=0.0, atol=1e-14
            )

    def test_modified_photon_kinetic_moments_match_direct_projection(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        comoving_momenta = np.array([0.12, 0.37, 0.81])
        quadrature_weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.4
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        actual = photon_kinetic_stress_multipoles(
            comoving_momenta,
            quadrature_weights,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        photon_speed = photon_speed_ratio(field_value, parameters)
        projected = {
            "delta_rho_GeV4": 0.0j,
            "delta_pressure_GeV4": 0.0j,
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for index, comoving_momentum in enumerate(comoving_momenta):
            distribution_coefficients = [
                (2 * ell + 1) * moments[index, ell]
                for ell in range(moments.shape[1])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, distribution_coefficients
            )
            physical_momentum = comoving_momentum / scale_factor
            measure = (
                phase_space_factor
                * quadrature_weights[index]
                * comoving_momentum**2
                / scale_factor**3
            )
            angular_average = 0.5 * np.sum(angular_weights * distribution)
            angular_flux = 0.5 * np.sum(
                angular_weights * cosines * distribution
            )
            angular_shear = 0.5 * np.sum(
                angular_weights
                * (cosines**2 - 1.0 / 3.0)
                * distribution
            )
            projected["delta_rho_GeV4"] += (
                measure
                * photon_speed
                * physical_momentum
                * angular_average
            )
            projected["delta_pressure_GeV4"] += (
                measure
                * photon_speed
                * physical_momentum
                * angular_average
                / 3.0
            )
            projected["longitudinal_flux_GeV4"] += (
                measure
                * photon_speed**2
                * physical_momentum
                * angular_flux
            )
            projected["longitudinal_anisotropic_stress_GeV4"] += (
                measure
                * photon_speed
                * physical_momentum
                * angular_shear
            )

        for moment_name, expected in projected.items():
            np.testing.assert_allclose(
                actual[moment_name], expected, rtol=0.0, atol=1e-14
            )

    def test_photon_action_stress_composes_total_and_bare_scalar_source(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.4
        field_perturbation = 0.002 - 0.001j
        curvature = -0.001 + 0.0004j
        clock_velocity = 0.003 - 0.001j
        distribution = np.array([0.8, 0.4, 0.1])
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        degeneracy = 2.0

        actual = photon_action_stress_perturbation_moments(
            momenta,
            weights,
            scale_factor,
            field_value,
            field_perturbation,
            curvature,
            clock_velocity,
            moments,
            distribution,
            parameters,
            degeneracy,
        )
        physical_momenta = momenta / scale_factor
        physical_weights = weights / scale_factor
        metric_responses = photon_spatial_metric_measure_response_moments(
            momenta,
            weights,
            scale_factor,
            field_value,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )
        expected_total = sum_scalar_stress_moments(
            photon_kinetic_stress_multipoles(
                momenta,
                weights,
                scale_factor,
                field_value,
                moments,
                parameters,
                degeneracy,
            ),
            photon_background_dispersion_stress_moments(
                physical_momenta,
                physical_weights,
                field_value,
                field_perturbation,
                distribution,
                parameters,
                degeneracy,
            ),
            metric_responses["modified_action"],
            photon_clock_frame_stress_response_moments(
                physical_momenta,
                physical_weights,
                field_value,
                clock_velocity,
                distribution,
                parameters,
                degeneracy,
            ),
            photon_interaction_hilbert_clock_frame_contact_moments(
                physical_momenta,
                physical_weights,
                field_value,
                clock_velocity,
                distribution,
                parameters,
                degeneracy,
            ),
        )
        expected_maxwell_clock_frame = photon_maxwell_clock_frame_stress_response_moments(
            physical_momenta,
            physical_weights,
            field_value,
            clock_velocity,
            distribution,
            parameters,
            degeneracy,
        )
        expected_bare = sum_scalar_stress_moments(
            photon_maxwell_stress_multipoles(
                momenta,
                weights,
                scale_factor,
                field_value,
                moments,
                parameters,
                degeneracy,
            ),
            photon_maxwell_background_response_moments(
                physical_momenta,
                physical_weights,
                field_value,
                field_perturbation,
                distribution,
                parameters,
                degeneracy,
            ),
            metric_responses["maxwell"],
            expected_maxwell_clock_frame,
        )
        expected_interaction = sum_scalar_stress_moments(
            photon_interaction_stress_multipoles(
                momenta,
                weights,
                scale_factor,
                field_value,
                moments,
                parameters,
                degeneracy,
            ),
            photon_interaction_background_response_moments(
                physical_momenta,
                physical_weights,
                field_value,
                field_perturbation,
                distribution,
                parameters,
                degeneracy,
            ),
            metric_responses["interaction"],
            photon_interaction_clock_frame_stress_response_moments(
                physical_momenta,
                physical_weights,
                field_value,
                clock_velocity,
                distribution,
                parameters,
                degeneracy,
            ),
            photon_interaction_hilbert_clock_frame_contact_moments(
                physical_momenta,
                physical_weights,
                field_value,
                clock_velocity,
                distribution,
                parameters,
                degeneracy,
            ),
        )
        isotropic_multipoles = np.zeros_like(moments)
        isotropic_multipoles[:, 0] = distribution
        bare_background_density = uncoupled_kinetic_stress_multipoles(
            momenta,
            weights,
            0.0,
            scale_factor,
            isotropic_multipoles,
            degeneracy,
        )["delta_rho_GeV4"]
        photon_speed = photon_group_velocity_local(field_value, parameters)
        xi = parameters.h_GeV_inv * field_value
        maxwell_energy_ratio = photon_speed / (1.0 - xi**2)
        maxwell_flux_ratio = photon_speed / np.sqrt(1.0 - xi**2)
        expected_maxwell_frame_flux = (
            (4.0 / 3.0)
            * (maxwell_energy_ratio - photon_speed * maxwell_flux_ratio)
            * bare_background_density
            * clock_velocity
        )
        np.testing.assert_allclose(
            expected_maxwell_clock_frame["longitudinal_flux_GeV4"],
            expected_maxwell_frame_flux,
            rtol=0.0,
            atol=1e-14,
        )
        for moment_name, expected_value in expected_total.items():
            np.testing.assert_allclose(
                actual["total_stress_moments"][moment_name],
                expected_value,
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                actual["maxwell_stress_moments"][moment_name]
                + actual["interaction_stress_moments"][moment_name],
                expected_value,
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                actual["maxwell_stress_moments"][moment_name],
                expected_bare[moment_name],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                actual["interaction_stress_moments"][moment_name],
                expected_interaction[moment_name],
                rtol=0.0,
                atol=1e-14,
            )
        self.assertAlmostEqual(
            actual["bare_delta_rho_GeV4"],
            expected_bare["delta_rho_GeV4"],
            places=14,
        )
        self.assertAlmostEqual(
            actual["bare_delta_pressure_GeV4"],
            expected_bare["delta_pressure_GeV4"],
            places=14,
        )
        self.assertAlmostEqual(
            actual["scalar_source_perturbation_GeV3"],
            -parameters.h_GeV_inv * expected_bare["delta_rho_GeV4"],
            places=14,
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled = photon_action_stress_perturbation_moments(
            momenta,
            weights,
            scale_factor,
            field_value,
            field_perturbation,
            curvature,
            clock_velocity,
            moments,
            distribution,
            uncoupled_parameters,
            degeneracy,
        )
        self.assertEqual(uncoupled["scalar_source_perturbation_GeV3"], 0.0j)
        for moment_name in uncoupled["interaction_stress_moments"]:
            np.testing.assert_allclose(
                uncoupled["interaction_stress_moments"][moment_name],
                0.0j,
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                uncoupled["total_stress_moments"][moment_name],
                uncoupled["maxwell_stress_moments"][moment_name],
                rtol=0.0,
                atol=1e-14,
            )
        with self.assertRaises(ValueError):
            photon_action_stress_perturbation_moments(
                momenta,
                weights,
                0.0,
                field_value,
                field_perturbation,
                curvature,
                clock_velocity,
                moments,
                distribution,
                parameters,
                degeneracy,
            )

    def test_photon_interaction_stress_matches_modified_maxwell_action(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.4
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        actual = photon_interaction_stress_multipoles(
            momenta,
            weights,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        xi = parameters.h_GeV_inv * field_value
        photon_speed = photon_speed_ratio(field_value, parameters)
        projected = {
            "delta_rho_GeV4": 0.0j,
            "delta_pressure_GeV4": 0.0j,
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for index, momentum in enumerate(momenta):
            coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(moments[index])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, coefficients
            )
            physical_momentum = momentum / scale_factor
            measure = (
                phase_space_factor
                * weights[index]
                * momentum**2
                / scale_factor**3
            )
            interaction_energy = (
                -xi**2
                * photon_speed
                * physical_momentum
                / (1.0 - xi**2)
            )
            interaction_flux_energy = (
                -xi * physical_momentum / (1.0 + xi)
            )
            angular_average = 0.5 * np.sum(angular_weights * distribution)
            angular_flux = 0.5 * np.sum(
                angular_weights * cosines * distribution
            )
            angular_shear = 0.5 * np.sum(
                angular_weights
                * (cosines**2 - 1.0 / 3.0)
                * distribution
            )
            projected["delta_rho_GeV4"] += (
                measure * interaction_energy * angular_average
            )
            projected["delta_pressure_GeV4"] += (
                measure * interaction_energy * angular_average / 3.0
            )
            projected["longitudinal_flux_GeV4"] += (
                measure * interaction_flux_energy * angular_flux
            )
            projected["longitudinal_anisotropic_stress_GeV4"] += (
                measure * interaction_energy * angular_shear
            )

        for moment_name, expected in projected.items():
            np.testing.assert_allclose(
                actual[moment_name], expected, rtol=0.0, atol=1e-14
            )

        maxwell_stress = photon_maxwell_stress_multipoles(
            momenta,
            weights,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )
        uncoupled_stress = uncoupled_kinetic_stress_multipoles(
            momenta,
            weights,
            0.0,
            scale_factor,
            moments,
            degeneracy,
        )
        total_stress = photon_kinetic_stress_multipoles(
            momenta,
            weights,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )
        for moment_name in total_stress:
            np.testing.assert_allclose(
                total_stress[moment_name],
                maxwell_stress[moment_name] + actual[moment_name],
                rtol=0.0,
                atol=1e-14,
            )
        np.testing.assert_allclose(
            maxwell_stress["longitudinal_flux_GeV4"]
            + actual["longitudinal_flux_GeV4"],
            photon_speed**2 * uncoupled_stress["longitudinal_flux_GeV4"],
            rtol=0.0,
            atol=1e-14,
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled = photon_interaction_stress_multipoles(
            momenta,
            weights,
            scale_factor,
            field_value,
            moments,
            uncoupled_parameters,
            degeneracy,
        )
        for moment_name in uncoupled:
            np.testing.assert_allclose(
                uncoupled[moment_name], 0.0j, rtol=0.0, atol=1e-14
            )

    def test_neutrino_scalar_source_multipoles_match_direct_projection(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        mass = 0.29
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        source = neutrino_scalar_source_from_distribution_multipoles(
            momenta,
            weights,
            mass,
            moments,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        projected = 0.0j
        for index, momentum in enumerate(momenta):
            distribution_coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(moments[index])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, distribution_coefficients
            )
            particle_source = neutrino_scalar_source_per_particle(
                momentum, mass, parameters
            )
            angular_average = 0.5 * np.sum(
                angular_weights * distribution
            )
            projected += (
                phase_space_factor
                * weights[index]
                * momentum**2
                * particle_source
                * angular_average
            )

        self.assertAlmostEqual(source, projected, places=14)

    def test_coupled_dust_scalar_source_uses_bare_density_perturbation(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        bare_density = 0.41
        density_contrast = 0.012 - 0.004j
        actual = coupled_pressureless_matter_scalar_source_perturbation(
            bare_density,
            density_contrast,
            parameters,
        )
        expected = isotropic_fluid_scalar_source_perturbation(
            bare_density * density_contrast,
            0.0j,
            parameters,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

        equal_couplings = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.03,
        )
        self.assertEqual(
            coupled_pressureless_matter_scalar_source_perturbation(
                bare_density,
                density_contrast,
                equal_couplings,
            ),
            0.0j,
        )

    def test_scalar_source_sum_composes_photon_mapping_and_species_scalars(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.2
        field_perturbation = 0.002 - 0.001j
        curvature = -0.001 + 0.0004j
        distribution = np.array([0.8, 0.4, 0.1])
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        degeneracy = 2.0
        photon_source = {
            "scalar_source_perturbation_GeV3": photon_scalar_source_perturbation(
                momenta,
                weights,
                scale_factor,
                field_value,
                field_perturbation,
                curvature,
                moments,
                distribution,
                parameters,
                degeneracy,
            ),
        }
        neutrino_source = neutrino_scalar_source_perturbation(
            momenta / scale_factor,
            weights / scale_factor,
            0.29,
            moments,
            distribution,
            curvature,
            parameters,
            degeneracy,
        )
        dust_contrast = 0.012 - 0.004j
        dust_source = coupled_pressureless_matter_scalar_source_perturbation(
            0.41,
            dust_contrast,
            parameters,
        )
        actual = sum_scalar_source_perturbations(
            photon_source,
            neutrino_source,
            dust_source,
        )
        expected = (
            photon_source["scalar_source_perturbation_GeV3"]
            + neutrino_source
            + dust_source
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        _, field_rhs = scalar_field_perturbation_rhs(
            0.0j,
            0.0j,
            0.8,
            0.11,
            scale_factor,
            0.4,
            0.0,
            0.0,
            0.0j,
            0.0j,
            0.0j,
            actual,
        )
        np.testing.assert_allclose(
            field_rhs,
            -scale_factor**2 * actual,
            rtol=0.0,
            atol=1e-15,
        )
        with self.assertRaises(ValueError):
            sum_scalar_source_perturbations()
        with self.assertRaises(ValueError):
            sum_scalar_source_perturbations({})

    def test_neutrino_scalar_source_metric_measure_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        mass = 0.29
        curvature = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        metric_response = neutrino_scalar_source_spatial_metric_measure_response(
            momenta,
            weights,
            mass,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2
        )

        def integrated_source(metric_curvature):
            local_momenta = momenta / (1.0 - metric_curvature)
            local_measure = phase_space_weights / (1.0 - metric_curvature) ** 3
            particle_sources = np.array(
                [
                    neutrino_scalar_source_per_particle(
                        momentum, mass, parameters
                    )
                    for momentum in local_momenta
                ]
            )
            return np.sum(local_measure * distribution * particle_sources)

        finite_difference = curvature * (
            integrated_source(step) - integrated_source(-step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            metric_response,
            finite_difference,
            rtol=1e-8,
            atol=1e-14,
        )

        total_source = neutrino_scalar_source_perturbation(
            momenta,
            weights,
            mass,
            moments,
            distribution,
            curvature,
            parameters,
            degeneracy,
        )
        distribution_source = neutrino_scalar_source_from_distribution_multipoles(
            momenta,
            weights,
            mass,
            moments,
            parameters,
            degeneracy,
        )
        np.testing.assert_allclose(
            total_source,
            distribution_source + metric_response,
            rtol=0.0,
            atol=1e-14,
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled_source = neutrino_scalar_source_perturbation(
            momenta,
            weights,
            mass,
            moments,
            distribution,
            curvature,
            uncoupled_parameters,
            degeneracy,
        )
        self.assertEqual(uncoupled_source, 0.0j)

    def test_photon_interaction_hilbert_tensor_matches_action_variation(self):
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        field_strength = np.array(
            [
                [0.0, 0.2, -0.3, 0.1],
                [-0.2, 0.0, 0.6, -0.4],
                [0.3, -0.6, 0.0, 0.5],
                [-0.1, 0.4, -0.5, 0.0],
            ]
        )
        speed = 0.08
        lorentz_factor = 1.0 / np.sqrt(1.0 - speed**2)
        clock_velocity = np.array(
            [lorentz_factor, lorentz_factor * speed, 0.0, 0.0]
        )
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.14,
        )
        field_value = 0.3
        interaction_strength = parameters.h_GeV_inv * field_value
        actual = photon_interaction_hilbert_stress_tensor(
            field_strength,
            metric,
            clock_velocity,
            field_value,
            parameters,
        )
        inverse_metric = np.linalg.inv(metric)

        def interaction_action_density(varied_inverse_metric):
            varied_metric = np.linalg.inv(varied_inverse_metric)
            normalized_velocity = clock_velocity / np.sqrt(
                -(clock_velocity @ varied_metric @ clock_velocity)
            )
            field_squared = np.sum(
                field_strength
                * (
                    varied_inverse_metric
                    @ field_strength
                    @ varied_inverse_metric
                )
            )
            maxwell_stress = (
                field_strength
                @ varied_inverse_metric
                @ field_strength.T
                - 0.25 * varied_metric * field_squared
            )
            interaction_energy = (
                normalized_velocity
                @ maxwell_stress
                @ normalized_velocity
            )
            return (
                np.sqrt(-np.linalg.det(varied_metric))
                * interaction_strength
                * interaction_energy
            )

        step = 1e-6
        for row in range(4):
            for column in range(row, 4):
                perturbation = np.zeros((4, 4))
                perturbation[row, column] = step
                if row != column:
                    perturbation[column, row] = step
                finite_difference = (
                    interaction_action_density(inverse_metric + perturbation)
                    - interaction_action_density(inverse_metric - perturbation)
                ) / (2.0 * step)
                expected = (
                    -0.5 * actual[row, column]
                    if row == column
                    else -actual[row, column]
                )
                self.assertAlmostEqual(finite_difference, expected, places=9)

    def test_photon_hilbert_plane_wave_recovers_ray_stress_weights(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        field_value = 0.4
        interaction_strength = parameters.h_GeV_inv * field_value
        photon_speed = photon_group_velocity_local(field_value, parameters)
        momentum = 0.37
        amplitude = np.sqrt(
            photon_speed / ((1.0 - interaction_strength) * momentum)
        )
        electric = photon_speed * momentum * amplitude
        magnetic = momentum * amplitude
        field_strength = np.zeros((4, 4))
        field_strength[2, 0] = electric
        field_strength[0, 2] = -electric
        field_strength[1, 2] = magnetic
        field_strength[2, 1] = -magnetic
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        inverse_metric = metric.copy()
        field_squared = np.sum(
            field_strength
            * (inverse_metric @ field_strength @ inverse_metric)
        )
        maxwell_stress = (
            field_strength @ inverse_metric @ field_strength.T
            - 0.25 * metric * field_squared
        )
        interaction_stress = photon_interaction_hilbert_stress_tensor(
            field_strength,
            metric,
            np.array([1.0, 0.0, 0.0, 0.0]),
            field_value,
            parameters,
        )
        total_stress = maxwell_stress + interaction_stress

        self.assertAlmostEqual(
            total_stress[0, 0], photon_speed * momentum, places=14
        )
        self.assertAlmostEqual(
            maxwell_stress[0, 0],
            photon_speed * momentum / (1.0 - interaction_strength**2),
            places=14,
        )
        self.assertAlmostEqual(
            interaction_stress[0, 0],
            -interaction_strength**2
            * photon_speed
            * momentum
            / (1.0 - interaction_strength**2),
            places=14,
        )
        self.assertAlmostEqual(
            -maxwell_stress[0, 1], momentum / (1.0 + interaction_strength), places=14
        )
        self.assertAlmostEqual(
            -interaction_stress[0, 1],
            -interaction_strength * momentum / (1.0 + interaction_strength),
            places=14,
        )
        self.assertAlmostEqual(
            -total_stress[0, 1], photon_speed**2 * momentum, places=14
        )

    def test_photon_interaction_hilbert_angular_average_matches_multipoles(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        field_value = 0.4
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        degeneracy = 2.0
        photon_speed = photon_group_velocity_local(field_value, parameters)
        interaction_strength = parameters.h_GeV_inv * field_value
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        clock_velocity = np.array([1.0, 0.0, 0.0, 0.0])
        cosines, cosine_weights = np.polynomial.legendre.leggauss(8)
        azimuths = 2.0 * pi * np.arange(12) / 12.0
        angular_mean_tensors = np.zeros((momenta.size, 4, 4), dtype=complex)

        for momentum_index, momentum in enumerate(momenta):
            multipole_coefficients = (
                2 * np.arange(moments.shape[1]) + 1
            ) * moments[momentum_index]
            for cosine, cosine_weight in zip(cosines, cosine_weights):
                sine = np.sqrt(1.0 - cosine**2)
                for azimuth in azimuths:
                    direction = np.array(
                        [
                            sine * np.cos(azimuth),
                            sine * np.sin(azimuth),
                            cosine,
                        ]
                    )
                    theta_polarization = np.array(
                        [
                            cosine * np.cos(azimuth),
                            cosine * np.sin(azimuth),
                            -sine,
                        ]
                    )
                    phi_polarization = np.array(
                        [-np.sin(azimuth), np.cos(azimuth), 0.0]
                    )
                    distribution = np.polynomial.legendre.legval(
                        cosine, multipole_coefficients
                    )
                    angular_weight = (
                        0.5 * cosine_weight / azimuths.size
                    )
                    for polarization in (
                        theta_polarization,
                        phi_polarization,
                    ):
                        amplitude = np.sqrt(
                            photon_speed
                            / ((1.0 - interaction_strength) * momentum)
                        )
                        electric = photon_speed * momentum * amplitude
                        magnetic = momentum * amplitude
                        magnetic_direction = np.cross(
                            direction, polarization
                        )
                        field_strength = np.zeros((4, 4))
                        field_strength[1:, 0] = electric * polarization
                        field_strength[0, 1:] = -electric * polarization
                        field_strength[1:, 1:] = magnetic * np.array(
                            [
                                [
                                    0.0,
                                    magnetic_direction[2],
                                    -magnetic_direction[1],
                                ],
                                [
                                    -magnetic_direction[2],
                                    0.0,
                                    magnetic_direction[0],
                                ],
                                [
                                    magnetic_direction[1],
                                    -magnetic_direction[0],
                                    0.0,
                                ],
                            ]
                        )
                        covariant_stress = photon_interaction_hilbert_stress_tensor(
                            field_strength,
                            metric,
                            clock_velocity,
                            field_value,
                            parameters,
                        )
                        contravariant_stress = (
                            metric @ covariant_stress @ metric
                        )
                        angular_mean_tensors[momentum_index] += (
                            0.5
                            * angular_weight
                            * distribution
                            * contravariant_stress
                        )

        radial_measure = (
            degeneracy
            / (2.0 * pi**2)
            * weights
            * momenta**2
        )
        isotropic_pressure = np.trace(
            angular_mean_tensors[:, 1:, 1:], axis1=1, axis2=2
        ) / 3.0
        actual = {
            "delta_rho_GeV4": np.sum(
                radial_measure * angular_mean_tensors[:, 0, 0]
            ),
            "delta_pressure_GeV4": np.sum(
                radial_measure * isotropic_pressure
            ),
            "longitudinal_flux_GeV4": np.sum(
                radial_measure * angular_mean_tensors[:, 0, 3]
            ),
            "longitudinal_anisotropic_stress_GeV4": np.sum(
                radial_measure
                * (angular_mean_tensors[:, 3, 3] - isotropic_pressure)
            ),
        }
        expected = photon_interaction_stress_multipoles(
            momenta,
            weights,
            1.0,
            field_value,
            moments,
            parameters,
            degeneracy,
        )
        for moment_name, expected_value in expected.items():
            np.testing.assert_allclose(
                actual[moment_name],
                expected_value,
                rtol=1e-12,
                atol=1e-14,
            )

    def test_photon_interaction_hilbert_clock_contact_matches_angular_average(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        field_value = 0.4
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        degeneracy = 2.0
        frame_velocity = 0.002 - 0.001j
        step = 1e-5
        photon_speed = photon_group_velocity_local(field_value, parameters)
        interaction_strength = parameters.h_GeV_inv * field_value
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        cosines, cosine_weights = np.polynomial.legendre.leggauss(8)
        azimuths = 2.0 * pi * np.arange(12) / 12.0
        phase_space_weights = (
            degeneracy
            / (2.0 * pi**2)
            * weights
            * momenta**2
            * distribution
        )

        def integrated_hilbert_flux(clock_speed):
            lorentz_factor = 1.0 / np.sqrt(1.0 - clock_speed**2)
            clock_velocity = np.array(
                [lorentz_factor, 0.0, 0.0, lorentz_factor * clock_speed]
            )
            total_flux = 0.0
            for momentum_index, momentum in enumerate(momenta):
                amplitude = np.sqrt(
                    photon_speed
                    / ((1.0 - interaction_strength) * momentum)
                )
                electric_amplitude = photon_speed * momentum * amplitude
                magnetic_amplitude = momentum * amplitude
                for cosine, cosine_weight in zip(cosines, cosine_weights):
                    sine = np.sqrt(1.0 - cosine**2)
                    for azimuth in azimuths:
                        direction = np.array(
                            [
                                sine * np.cos(azimuth),
                                sine * np.sin(azimuth),
                                cosine,
                            ]
                        )
                        polarizations = (
                            np.array(
                                [
                                    cosine * np.cos(azimuth),
                                    cosine * np.sin(azimuth),
                                    -sine,
                                ]
                            ),
                            np.array(
                                [-np.sin(azimuth), np.cos(azimuth), 0.0]
                            ),
                        )
                        angular_weight = (
                            0.5 * cosine_weight / azimuths.size
                        )
                        for polarization in polarizations:
                            magnetic_direction = np.cross(
                                direction, polarization
                            )
                            field_strength = np.zeros((4, 4))
                            field_strength[1:, 0] = (
                                electric_amplitude * polarization
                            )
                            field_strength[0, 1:] = (
                                -electric_amplitude * polarization
                            )
                            field_strength[1:, 1:] = magnetic_amplitude * np.array(
                                [
                                    [
                                        0.0,
                                        magnetic_direction[2],
                                        -magnetic_direction[1],
                                    ],
                                    [
                                        -magnetic_direction[2],
                                        0.0,
                                        magnetic_direction[0],
                                    ],
                                    [
                                        magnetic_direction[1],
                                        -magnetic_direction[0],
                                        0.0,
                                    ],
                                ]
                            )
                            covariant_stress = photon_interaction_hilbert_stress_tensor(
                                field_strength,
                                metric,
                                clock_velocity,
                                field_value,
                                parameters,
                            )
                            contravariant_stress = (
                                metric @ covariant_stress @ metric
                            )
                            total_flux += (
                                phase_space_weights[momentum_index]
                                * angular_weight
                                * 0.5
                                * contravariant_stress[0, 3]
                            )
            return total_flux

        direct_contact = frame_velocity * (
            integrated_hilbert_flux(step)
            - integrated_hilbert_flux(-step)
        ) / (2.0 * step)
        actual_contact = photon_interaction_hilbert_clock_frame_contact_moments(
            momenta,
            weights,
            field_value,
            frame_velocity,
            distribution,
            parameters,
            degeneracy,
        )
        bare_density = np.sum(phase_space_weights * momenta)
        expected_contact = (
            (4.0 / 3.0)
            * interaction_strength
            * photon_speed
            / (1.0 + interaction_strength)
            * bare_density
            * frame_velocity
        )
        np.testing.assert_allclose(
            direct_contact,
            actual_contact["longitudinal_flux_GeV4"],
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual_contact["longitudinal_flux_GeV4"],
            expected_contact,
            rtol=1e-8,
            atol=1e-14,
        )

    def test_photon_clock_frame_response_matches_angular_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        field_value = 0.4
        frame_velocity = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_clock_frame_stress_response_moments(
            momenta,
            weights,
            field_value,
            frame_velocity,
            distribution,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        photon_speed = photon_speed_ratio(field_value, parameters)
        frame_coefficient = 1.0 - photon_speed**2

        def integrated_energy_current(frame_projection):
            total = 0.0
            for index, momentum in enumerate(momenta):
                energy = (
                    photon_speed * momentum
                    + frame_coefficient
                    * momentum
                    * cosines
                    * frame_projection
                )
                longitudinal_velocity = (
                    photon_speed * cosines
                    + frame_coefficient * frame_projection
                )
                angular_current = 0.5 * np.sum(
                    angular_weights * energy * longitudinal_velocity
                )
                total += (
                    phase_space_factor
                    * weights[index]
                    * momentum**2
                    * distribution[index]
                    * angular_current
                )
            return total

        finite_difference = frame_velocity * (
            integrated_energy_current(step)
            - integrated_energy_current(-step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual["longitudinal_flux_GeV4"],
            finite_difference,
            rtol=1e-8,
            atol=1e-14,
        )
        self.assertEqual(actual["delta_rho_GeV4"], 0.0j)
        self.assertEqual(actual["delta_pressure_GeV4"], 0.0j)
        self.assertEqual(
            actual["longitudinal_anisotropic_stress_GeV4"], 0.0j
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled = photon_clock_frame_stress_response_moments(
            momenta,
            weights,
            field_value,
            frame_velocity,
            distribution,
            uncoupled_parameters,
            degeneracy,
        )
        for moment_name in uncoupled:
            self.assertEqual(uncoupled[moment_name], 0.0j)

    def test_photon_spatial_metric_measure_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        scale_factor = 0.73
        field_value = 0.4
        curvature = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_spatial_metric_measure_response_moments(
            momenta,
            weights,
            scale_factor,
            field_value,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy
            / (2.0 * np.pi**2)
            * weights
            * momenta**2
            * distribution
        )
        photon_speed = photon_speed_ratio(field_value, parameters)
        xi = parameters.h_GeV_inv * field_value
        energy_ratios = {
            "modified_action": photon_speed,
            "maxwell": photon_speed / (1.0 - xi**2),
            "interaction": -xi**2 * photon_speed / (1.0 - xi**2),
        }

        def integrated_density(metric_curvature, energy_ratio):
            local_scale = scale_factor * (1.0 - metric_curvature)
            local_momenta = momenta / local_scale
            local_measure = (
                degeneracy
                / (2.0 * np.pi**2)
                * weights
                * momenta**2
                / local_scale**3
                * distribution
            )
            return np.sum(local_measure * energy_ratio * local_momenta)

        for sector, energy_ratio in energy_ratios.items():
            finite_difference = curvature * (
                integrated_density(step, energy_ratio)
                - integrated_density(-step, energy_ratio)
            ) / (2.0 * step)
            np.testing.assert_allclose(
                actual[sector]["delta_rho_GeV4"],
                finite_difference,
                rtol=1e-8,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                actual[sector]["delta_pressure_GeV4"],
                finite_difference / 3.0,
                rtol=1e-8,
                atol=1e-14,
            )

        def integrated_hilbert_interaction_stress(metric_curvature):
            spatial_metric = 1.0 - 2.0 * metric_curvature
            spatial_scale = np.sqrt(spatial_metric)
            metric = np.diag(
                [-1.0, spatial_metric, spatial_metric, spatial_metric]
            )
            coframe = np.diag(
                [1.0, spatial_scale, spatial_scale, spatial_scale]
            )
            orthonormal_frame = np.diag(
                [1.0, 1.0 / spatial_scale, 1.0 / spatial_scale, 1.0 / spatial_scale]
            )
            clock_velocity = np.array([1.0, 0.0, 0.0, 0.0])
            local_measure = (
                degeneracy
                / (2.0 * np.pi**2)
                * weights
                * momenta**2
                * distribution
                / (scale_factor**3 * spatial_scale**3)
            )
            local_momenta = momenta / (scale_factor * spatial_scale)
            density = 0.0
            pressure = 0.0
            for index, local_momentum in enumerate(local_momenta):
                amplitude = np.sqrt(
                    photon_speed
                    / ((1.0 - xi) * local_momentum)
                )
                electric = photon_speed * local_momentum * amplitude
                magnetic = local_momentum * amplitude
                local_field_strength = np.zeros((4, 4))
                local_field_strength[2, 0] = electric
                local_field_strength[0, 2] = -electric
                local_field_strength[1, 2] = magnetic
                local_field_strength[2, 1] = -magnetic
                coordinate_field_strength = (
                    coframe @ local_field_strength @ coframe
                )
                covariant_stress = photon_interaction_hilbert_stress_tensor(
                    coordinate_field_strength,
                    metric,
                    clock_velocity,
                    field_value,
                    parameters,
                )
                local_stress = (
                    orthonormal_frame
                    @ covariant_stress
                    @ orthonormal_frame.T
                )
                density += local_measure[index] * local_stress[0, 0]
                pressure += (
                    local_measure[index]
                    * np.trace(local_stress[1:, 1:])
                    / 3.0
                )
            return density, pressure

        lower_hilbert_stress = integrated_hilbert_interaction_stress(-step)
        upper_hilbert_stress = integrated_hilbert_interaction_stress(step)
        direct_metric_response = tuple(
            curvature
            * (upper_hilbert_stress[index] - lower_hilbert_stress[index])
            / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["interaction"]["delta_rho_GeV4"],
            direct_metric_response[0],
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["interaction"]["delta_pressure_GeV4"],
            direct_metric_response[1],
            rtol=1e-8,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            actual["maxwell"]["delta_rho_GeV4"]
            + actual["interaction"]["delta_rho_GeV4"],
            actual["modified_action"]["delta_rho_GeV4"],
            rtol=1e-14,
            atol=1e-14,
        )

    def test_photon_hilbert_lapse_response_vanishes_at_fixed_local_state(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        scale_factor = 0.73
        field_value = 0.4
        degeneracy = 2.0
        photon_speed = photon_group_velocity_local(field_value, parameters)
        interaction_strength = parameters.h_GeV_inv * field_value
        local_measure = (
            degeneracy
            / (2.0 * np.pi**2)
            * weights
            * momenta**2
            * distribution
            / scale_factor**3
        )
        local_momenta = momenta / scale_factor

        def integrated_local_stress(lapse_perturbation):
            lapse = np.sqrt(1.0 + 2.0 * lapse_perturbation)
            metric = np.diag([-lapse**2, 1.0, 1.0, 1.0])
            coframe = np.diag([lapse, 1.0, 1.0, 1.0])
            orthonormal_frame = np.diag([1.0 / lapse, 1.0, 1.0, 1.0])
            clock_velocity = np.array([1.0 / lapse, 0.0, 0.0, 0.0])
            density = 0.0
            pressure = 0.0
            for index, local_momentum in enumerate(local_momenta):
                amplitude = np.sqrt(
                    photon_speed
                    / ((1.0 - interaction_strength) * local_momentum)
                )
                electric = photon_speed * local_momentum * amplitude
                magnetic = local_momentum * amplitude
                local_field = np.zeros((4, 4))
                local_field[2, 0] = electric
                local_field[0, 2] = -electric
                local_field[1, 2] = magnetic
                local_field[2, 1] = -magnetic
                coordinate_field = coframe @ local_field @ coframe
                covariant_stress = photon_interaction_hilbert_stress_tensor(
                    coordinate_field,
                    metric,
                    clock_velocity,
                    field_value,
                    parameters,
                )
                local_stress = (
                    orthonormal_frame
                    @ covariant_stress
                    @ orthonormal_frame.T
                )
                density += local_measure[index] * local_stress[0, 0]
                pressure += (
                    local_measure[index]
                    * np.trace(local_stress[1:, 1:])
                    / 3.0
                )
            return density, pressure

        step = 1e-5
        lower = integrated_local_stress(-step)
        upper = integrated_local_stress(step)
        lapse_response = tuple(
            (upper[index] - lower[index]) / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            lapse_response,
            (0.0, 0.0),
            rtol=0.0,
            atol=1e-10,
        )

    def test_photon_maxwell_stress_matches_wave_projection_and_scalar_source(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.12,
        )
        comoving_momenta = np.array([0.12, 0.37, 0.81])
        quadrature_weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.4
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        actual = photon_maxwell_stress_multipoles(
            comoving_momenta,
            quadrature_weights,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        xi = parameters.h_GeV_inv * field_value
        photon_speed = photon_speed_ratio(field_value, parameters)
        projected = {
            "delta_rho_GeV4": 0.0j,
            "delta_pressure_GeV4": 0.0j,
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for index, comoving_momentum in enumerate(comoving_momenta):
            coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(moments[index])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, coefficients
            )
            physical_momentum = comoving_momentum / scale_factor
            maxwell_energy = (
                photon_speed * physical_momentum / (1.0 - xi**2)
            )
            maxwell_flux_energy = (
                photon_speed * physical_momentum / np.sqrt(1.0 - xi**2)
            )
            measure = (
                phase_space_factor
                * quadrature_weights[index]
                * comoving_momentum**2
                / scale_factor**3
            )
            angular_average = 0.5 * np.sum(angular_weights * distribution)
            angular_flux = 0.5 * np.sum(
                angular_weights * cosines * distribution
            )
            angular_shear = 0.5 * np.sum(
                angular_weights
                * (cosines**2 - 1.0 / 3.0)
                * distribution
            )
            projected["delta_rho_GeV4"] += (
                measure * maxwell_energy * angular_average
            )
            projected["delta_pressure_GeV4"] += (
                measure * maxwell_energy * angular_average / 3.0
            )
            projected["longitudinal_flux_GeV4"] += (
                measure * maxwell_flux_energy * angular_flux
            )
            projected["longitudinal_anisotropic_stress_GeV4"] += (
                measure * maxwell_energy * angular_shear
            )

        for moment_name, expected in projected.items():
            np.testing.assert_allclose(
                actual[moment_name], expected, rtol=0.0, atol=1e-14
            )

        source = photon_scalar_source_from_distribution_multipoles(
            comoving_momenta / scale_factor,
            quadrature_weights / scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )
        np.testing.assert_allclose(
            source,
            -parameters.h_GeV_inv * actual["delta_rho_GeV4"],
            rtol=1e-14,
            atol=1e-14,
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled = photon_maxwell_stress_multipoles(
            comoving_momenta,
            quadrature_weights,
            scale_factor,
            field_value,
            moments,
            uncoupled_parameters,
            degeneracy,
        )
        standard = uncoupled_kinetic_stress_multipoles(
            comoving_momenta,
            quadrature_weights,
            0.0,
            scale_factor,
            moments,
            degeneracy,
        )
        for moment_name, expected in standard.items():
            np.testing.assert_allclose(
                uncoupled[moment_name], expected, rtol=0.0, atol=1e-14
            )

    def test_photon_distribution_source_matches_energy_derivative(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        field_value = 0.2
        step = 1e-6
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j],
                [0.1 - 0.02j, 0.03 + 0.01j],
                [-0.05j, 0.02 - 0.03j],
            ]
        )

        source = photon_scalar_source_from_distribution_multipoles(
            momenta,
            weights,
            field_value,
            moments,
            parameters,
            degeneracy,
        )

        def integrated_energy(field):
            speed = photon_speed_ratio(field, parameters)
            return (
                degeneracy
                / (2.0 * np.pi**2)
                * np.sum(weights * momenta**3 * speed * moments[:, 0])
            )

        finite_difference = (
            integrated_energy(field_value + step)
            - integrated_energy(field_value - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            source, finite_difference, rtol=1e-8, atol=1e-14
        )

    def test_photon_scalar_source_background_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        field_value = 0.2
        field_perturbation = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_scalar_source_background_response(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )

        def integrated_source(field):
            speed = photon_group_velocity_local(field, parameters)
            xi = parameters.h_GeV_inv * field
            particle_source = (
                -parameters.h_GeV_inv * speed * momenta / (1.0 - xi**2)
            )
            return np.sum(phase_space_weights * particle_source)

        finite_difference = field_perturbation * (
            integrated_source(field_value + step)
            - integrated_source(field_value - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual, finite_difference, rtol=1e-8, atol=1e-14
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled = photon_scalar_source_background_response(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            uncoupled_parameters,
            degeneracy,
        )
        self.assertEqual(uncoupled, 0.0j)

    def test_photon_scalar_source_spatial_metric_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        field_value = 0.2
        curvature = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_scalar_source_spatial_metric_measure_response(
            momenta,
            weights,
            field_value,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )
        photon_speed = photon_group_velocity_local(field_value, parameters)
        xi = parameters.h_GeV_inv * field_value

        def integrated_source(metric_curvature):
            local_momenta = momenta / (1.0 - metric_curvature)
            local_measure = phase_space_weights / (1.0 - metric_curvature) ** 3
            source_per_mode = (
                -parameters.h_GeV_inv
                * photon_speed
                * local_momenta
                / (1.0 - xi**2)
            )
            return np.sum(local_measure * source_per_mode)

        finite_difference = curvature * (
            integrated_source(step) - integrated_source(-step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual,
            finite_difference,
            rtol=1e-8,
            atol=1e-14,
        )
        maxwell_metric_response = photon_spatial_metric_measure_response_moments(
            momenta,
            weights,
            1.0,
            field_value,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )["maxwell"]
        np.testing.assert_allclose(
            actual,
            -parameters.h_GeV_inv
            * maxwell_metric_response["delta_rho_GeV4"],
            rtol=1e-14,
            atol=1e-14,
        )

        uncoupled_parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
        )
        uncoupled = photon_scalar_source_spatial_metric_measure_response(
            momenta,
            weights,
            field_value,
            curvature,
            distribution,
            uncoupled_parameters,
            degeneracy,
        )
        self.assertEqual(uncoupled, 0.0j)

    def test_photon_scalar_source_composer_matches_stress_and_field_rhs(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.2
        field_perturbation = 0.002 - 0.001j
        curvature = -0.001 + 0.0004j
        distribution = np.array([0.8, 0.4, 0.1])
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        degeneracy = 2.0
        actual = photon_scalar_source_perturbation(
            momenta,
            weights,
            scale_factor,
            field_value,
            field_perturbation,
            curvature,
            moments,
            distribution,
            parameters,
            degeneracy,
        )
        photon_stress = photon_action_stress_perturbation_moments(
            momenta,
            weights,
            scale_factor,
            field_value,
            field_perturbation,
            curvature,
            0.0,
            moments,
            distribution,
            parameters,
            degeneracy,
        )
        np.testing.assert_allclose(
            actual,
            photon_stress["scalar_source_perturbation_GeV3"],
            rtol=1e-12,
            atol=1e-14,
        )
        _, field_rhs = scalar_field_perturbation_rhs(
            0.0j,
            0.0j,
            0.8,
            0.11,
            scale_factor,
            0.4,
            0.0,
            0.0,
            0.0j,
            0.0j,
            0.0j,
            actual,
        )
        np.testing.assert_allclose(
            field_rhs,
            -scale_factor**2 * actual,
            rtol=0.0,
            atol=1e-15,
        )

    def test_neutrino_spatial_metric_measure_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        mass = 0.29
        field_value = 0.16
        curvature = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = neutrino_spatial_metric_measure_response_moments(
            momenta,
            weights,
            mass,
            field_value,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )

        def integrated_stress(metric_curvature):
            local_momenta = momenta / (1.0 - metric_curvature)
            local_measure = phase_space_weights / (1.0 - metric_curvature) ** 3
            energies = np.array(
                [
                    neutrino_energy_local(momentum, mass, field_value, parameters)
                    for momentum in local_momenta
                ]
            )
            velocities = neutrino_group_velocity_local(
                local_momenta, mass, field_value, parameters
            )
            return (
                np.sum(local_measure * energies),
                np.sum(local_measure * local_momenta * velocities / 3.0),
            )

        lower = integrated_stress(-step)
        upper = integrated_stress(step)
        finite_difference = tuple(
            curvature * (upper[index] - lower[index]) / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"],
            finite_difference[0],
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference[1],
            rtol=1e-8,
            atol=1e-14,
        )

    def test_neutrino_hilbert_spatial_metric_response_matches_worldline_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        mass = 0.29
        field_value = 0.16
        curvature = 0.002 - 0.001j
        step = 1e-5
        degeneracy = 2.0
        actual = neutrino_worldline_hilbert_spatial_metric_response_moments(
            momenta,
            weights,
            mass,
            field_value,
            curvature,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )
        clock_four_velocity = np.array([1.0, 0.0, 0.0, 0.0])

        def integrated_action_stress(metric_curvature):
            local_momenta = momenta / (1.0 - metric_curvature)
            local_measure = phase_space_weights / (1.0 - metric_curvature) ** 3
            total_stress = np.zeros((4, 4))
            for momentum, number_density in zip(local_momenta, local_measure):
                energy = np.hypot(momentum, mass)
                particle_four_velocity = np.array(
                    [energy / mass, 0.0, 0.0, momentum / mass]
                )
                total_stress += coupled_massive_worldline_hilbert_stress_tensor(
                    number_density,
                    mass,
                    field_value,
                    clock_four_velocity,
                    particle_four_velocity,
                    parameters,
                )["total_stress_tensor_GeV4"]
            return (
                total_stress[0, 0],
                np.trace(total_stress[1:, 1:]) / 3.0,
            )

        lower = integrated_action_stress(-step)
        upper = integrated_action_stress(step)
        finite_difference = tuple(
            curvature * (upper[index] - lower[index]) / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"],
            finite_difference[0],
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference[1],
            rtol=1e-8,
            atol=1e-14,
        )

    def test_neutrino_hilbert_lapse_is_absent_in_local_moments(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        mass = 0.29
        field_value = 0.16
        degeneracy = 2.0
        metric_step = 1e-5
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )
        clock_four_velocity = np.array([1.0, 0.0, 0.0, 0.0])
        expected_density = 0.0
        expected_pressure = 0.0
        for momentum, number_density in zip(momenta, phase_space_weights):
            energy = np.hypot(momentum, mass)
            particle_four_velocity = np.array(
                [energy / mass, 0.0, 0.0, momentum / mass]
            )
            stress = coupled_massive_worldline_hilbert_stress_tensor(
                number_density,
                mass,
                field_value,
                clock_four_velocity,
                particle_four_velocity,
                parameters,
            )["total_stress_tensor_GeV4"]
            expected_density += stress[0, 0]
            expected_pressure += np.trace(stress[1:, 1:]) / 3.0

        def integrated_action_local_stress(lapse_perturbation):
            lapse = np.sqrt(1.0 + 2.0 * lapse_perturbation)
            background_metric = np.diag([-lapse**2, 1.0, 1.0, 1.0])
            clock_gradient = np.array([1.0, 0.0, 0.0, 0.0])
            density = 0.0
            pressure = 0.0
            for momentum, number_density in zip(momenta, phase_space_weights):
                energy = np.hypot(momentum, mass)
                worldline_tangent = np.array(
                    [1.0, 0.0, 0.0, lapse * momentum / energy]
                )

                def point_particle_lagrangian(metric):
                    inverse_metric = np.linalg.inv(metric)
                    proper_time_rate = np.sqrt(
                        -(worldline_tangent @ metric @ worldline_tangent)
                    )
                    clock_norm = np.sqrt(
                        -(clock_gradient @ inverse_metric @ clock_gradient)
                    )
                    clock_velocity = (
                        -inverse_metric @ clock_gradient / clock_norm
                    )
                    particle_velocity = worldline_tangent / proper_time_rate
                    relative_clock_factor = (
                        particle_velocity @ metric @ clock_velocity
                    )
                    return -number_density * mass * proper_time_rate * (
                        1.0
                        + parameters.g_GeV_inv * field_value
                        - parameters.h_GeV_inv
                        * field_value
                        * relative_clock_factor**2
                    )

                for diagonal_axis in range(4):
                    upper_metric = background_metric.copy()
                    lower_metric = background_metric.copy()
                    upper_metric[diagonal_axis, diagonal_axis] += metric_step
                    lower_metric[diagonal_axis, diagonal_axis] -= metric_step
                    lagrangian_derivative = (
                        point_particle_lagrangian(upper_metric)
                        - point_particle_lagrangian(lower_metric)
                    ) / (2.0 * metric_step)
                    local_stress_component = (
                        2.0 * lagrangian_derivative / lapse
                    )
                    if diagonal_axis == 0:
                        density += lapse**2 * local_stress_component
                    else:
                        pressure += local_stress_component / 3.0
            return density, pressure

        for lapse_perturbation in (-0.01, 0.01):
            actual = integrated_action_local_stress(lapse_perturbation)
            np.testing.assert_allclose(
                actual,
                (expected_density, expected_pressure),
                rtol=1e-7,
                atol=1e-11,
            )

    def test_neutrino_hilbert_clock_frame_response_matches_worldline_angular_average(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        mass = 0.29
        field_value = 0.16
        frame_velocity = 0.002
        degeneracy = 2.0
        actual = neutrino_worldline_hilbert_clock_frame_response_moments(
            momenta,
            weights,
            mass,
            field_value,
            frame_velocity,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )
        angular_nodes, angular_weights = np.polynomial.legendre.leggauss(8)

        def integrated_action_flux(clock_tilt):
            clock_gamma = 1.0 / np.sqrt(1.0 - clock_tilt**2)
            clock_velocity = np.array(
                [clock_gamma, 0.0, 0.0, clock_gamma * clock_tilt]
            )
            total_flux = 0.0
            for momentum, number_density in zip(momenta, phase_space_weights):
                energy = np.hypot(momentum, mass)
                for cosine, angular_weight in zip(angular_nodes, angular_weights):
                    particle_velocity = np.array(
                        [
                            energy / mass,
                            momentum * np.sqrt(1.0 - cosine**2) / mass,
                            0.0,
                            momentum * cosine / mass,
                        ]
                    )
                    stress = coupled_massive_worldline_hilbert_stress_tensor(
                        number_density * angular_weight / 2.0,
                        mass,
                        field_value,
                        clock_velocity,
                        particle_velocity,
                        parameters,
                    )
                    total_flux += stress["total_stress_tensor_GeV4"][0, 3]
            return total_flux

        step = 1e-4
        direct_response = frame_velocity * (
            integrated_action_flux(step) - integrated_action_flux(-step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual["longitudinal_flux_GeV4"],
            direct_response,
            rtol=1e-6,
            atol=1e-14,
        )
        self.assertEqual(actual["delta_rho_GeV4"], 0.0j)
        self.assertEqual(actual["delta_pressure_GeV4"], 0.0j)
        self.assertEqual(actual["longitudinal_anisotropic_stress_GeV4"], 0.0j)

    def test_neutrino_background_dispersion_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        mass = 0.29
        field_value = 0.16
        field_perturbation = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = neutrino_background_dispersion_stress_moments(
            momenta,
            weights,
            mass,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )

        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )

        def integrated_stress(field):
            energies = np.array(
                [
                    neutrino_energy_local(momentum, mass, field, parameters)
                    for momentum in momenta
                ]
            )
            velocities = neutrino_group_velocity_local(
                momenta, mass, field, parameters
            )
            return (
                np.sum(phase_space_weights * energies),
                np.sum(phase_space_weights * momenta * velocities / 3.0),
            )

        lower = integrated_stress(field_value - step)
        upper = integrated_stress(field_value + step)
        finite_difference = tuple(
            field_perturbation * (upper[index] - lower[index]) / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"], finite_difference[0], rtol=1e-8, atol=1e-14
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference[1],
            rtol=1e-8,
            atol=1e-14,
        )

    def test_neutrino_hilbert_background_response_matches_worldline_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        mass = 0.29
        field_value = 0.16
        field_perturbation = 0.002
        step = 1e-3
        degeneracy = 2.0
        actual = neutrino_worldline_hilbert_background_response_moments(
            momenta,
            weights,
            mass,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )
        clock_four_velocity = np.array([1.0, 0.0, 0.0, 0.0])

        def integrated_action_stress(field):
            total_stress = np.zeros((4, 4))
            for momentum, number_density in zip(momenta, phase_space_weights):
                energy = np.hypot(momentum, mass)
                particle_four_velocity = np.array(
                    [energy / mass, 0.0, 0.0, momentum / mass]
                )
                total_stress += coupled_massive_worldline_hilbert_stress_tensor(
                    number_density,
                    mass,
                    field,
                    clock_four_velocity,
                    particle_four_velocity,
                    parameters,
                )["total_stress_tensor_GeV4"]
            return (
                total_stress[0, 0],
                np.trace(total_stress[1:, 1:]) / 3.0,
            )

        lower = integrated_action_stress(
            field_value - step * field_perturbation
        )
        upper = integrated_action_stress(
            field_value + step * field_perturbation
        )
        finite_difference = tuple(
            (upper[index] - lower[index]) / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"],
            finite_difference[0],
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference[1],
            rtol=1e-8,
            atol=1e-14,
        )

    def test_photon_maxwell_background_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        field_value = 0.2
        field_perturbation = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_maxwell_background_response_moments(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )

        def integrated_maxwell_density(field):
            speed = photon_group_velocity_local(field, parameters)
            xi = parameters.h_GeV_inv * field
            return np.sum(
                phase_space_weights * speed * momenta / (1.0 - xi**2)
            )

        finite_difference = field_perturbation * (
            integrated_maxwell_density(field_value + step)
            - integrated_maxwell_density(field_value - step)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"],
            finite_difference,
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference / 3.0,
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            photon_scalar_source_background_response(
                momenta,
                weights,
                field_value,
                field_perturbation,
                distribution,
                parameters,
                degeneracy,
            ),
            -parameters.h_GeV_inv * actual["delta_rho_GeV4"],
            rtol=1e-14,
            atol=1e-14,
        )

    def test_photon_interaction_background_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        field_value = 0.2
        field_perturbation = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_interaction_background_response_moments(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )

        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        clock_velocity = np.array([1.0, 0.0, 0.0, 0.0])

        def integrated_hilbert_interaction_stress(field):
            speed = photon_group_velocity_local(field, parameters)
            xi = parameters.h_GeV_inv * field
            density = 0.0
            pressure = 0.0
            for index, momentum in enumerate(momenta):
                amplitude = np.sqrt(speed / ((1.0 - xi) * momentum))
                electric = speed * momentum * amplitude
                magnetic = momentum * amplitude
                field_strength = np.zeros((4, 4))
                field_strength[2, 0] = electric
                field_strength[0, 2] = -electric
                field_strength[1, 2] = magnetic
                field_strength[2, 1] = -magnetic
                interaction_stress = photon_interaction_hilbert_stress_tensor(
                    field_strength,
                    metric,
                    clock_velocity,
                    field,
                    parameters,
                )
                density += phase_space_weights[index] * interaction_stress[0, 0]
                pressure += (
                    phase_space_weights[index]
                    * np.trace(interaction_stress[1:, 1:])
                    / 3.0
                )
            return density, pressure

        lower_stress = integrated_hilbert_interaction_stress(
            field_value - step
        )
        upper_stress = integrated_hilbert_interaction_stress(
            field_value + step
        )
        finite_difference = tuple(
            field_perturbation
            * (upper_stress[index] - lower_stress[index])
            / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"],
            finite_difference[0],
            rtol=1e-8,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference[1],
            rtol=1e-8,
            atol=1e-14,
        )

        total_response = photon_background_dispersion_stress_moments(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        maxwell_response = photon_maxwell_background_response_moments(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        for moment_name, expected in total_response.items():
            np.testing.assert_allclose(
                actual[moment_name] + maxwell_response[moment_name],
                expected,
                rtol=1e-14,
                atol=1e-14,
            )

    def test_photon_background_dispersion_response_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.04,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        distribution = np.array([0.8, 0.4, 0.1])
        field_value = 0.2
        field_perturbation = 0.002 - 0.001j
        step = 1e-6
        degeneracy = 2.0
        actual = photon_background_dispersion_stress_moments(
            momenta,
            weights,
            field_value,
            field_perturbation,
            distribution,
            parameters,
            degeneracy,
        )
        phase_space_weights = (
            degeneracy / (2.0 * np.pi**2) * weights * momenta**2 * distribution
        )

        def integrated_stress(field):
            speed = photon_speed_ratio(field, parameters)
            density = np.sum(phase_space_weights * speed * momenta)
            return density, density / 3.0

        lower = integrated_stress(field_value - step)
        upper = integrated_stress(field_value + step)
        finite_difference = tuple(
            field_perturbation * (upper[index] - lower[index]) / (2.0 * step)
            for index in range(2)
        )
        np.testing.assert_allclose(
            actual["delta_rho_GeV4"], finite_difference[0], rtol=1e-8, atol=1e-14
        )
        np.testing.assert_allclose(
            actual["delta_pressure_GeV4"],
            finite_difference[1],
            rtol=1e-8,
            atol=1e-14,
        )

    def test_neutrino_energy_first_order_massless_and_rest_limits(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        field_value = 0.1
        mass = 0.5
        momentum = 0.7

        rest_energy = neutrino_energy_local(
            0.0, mass, field_value, parameters
        )
        massless_energy = neutrino_energy_local(
            momentum, 0.0, field_value, parameters
        )

        self.assertAlmostEqual(
            rest_energy,
            mass * (1.0 + parameters.eta_GeV_inv * field_value),
            places=12,
        )
        self.assertAlmostEqual(
            massless_energy,
            momentum * (1.0 - parameters.h_GeV_inv * field_value),
            places=12,
        )

    def test_neutrino_particle_source_matches_isotropic_fluid_source(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momentum = 0.3
        mass = 0.4
        number_density = 2.3
        energy = np.hypot(momentum, mass)
        rho = number_density * energy
        pressure = number_density * momentum**2 / (3.0 * energy)

        fluid_source = isotropic_fluid_scalar_source(
            rho, pressure, parameters
        )
        particle_source = number_density * neutrino_scalar_source_per_particle(
            momentum, mass, parameters
        )

        self.assertAlmostEqual(fluid_source, particle_source, places=14)

    def test_neutrino_dispersion_interaction_stress_matches_action_moments(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        momenta = np.array([0.12, 0.37, 0.81])
        weights = np.array([0.03, 0.11, 0.07])
        scale_factor = 0.73
        field_value = 0.16
        mass = 0.29
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )
        actual = neutrino_dispersion_interaction_stress_multipoles(
            momenta,
            weights,
            mass,
            scale_factor,
            field_value,
            moments,
            parameters,
            degeneracy,
        )

        cosines, angular_weights = np.polynomial.legendre.leggauss(48)
        phase_space_factor = degeneracy / (2.0 * np.pi**2)
        projected = {
            "delta_rho_GeV4": 0.0j,
            "delta_pressure_GeV4": 0.0j,
            "longitudinal_flux_GeV4": 0.0j,
            "longitudinal_anisotropic_stress_GeV4": 0.0j,
        }
        for index, momentum in enumerate(momenta):
            coefficients = [
                (2 * ell + 1) * moment
                for ell, moment in enumerate(moments[index])
            ]
            distribution = np.polynomial.legendre.legval(
                cosines, coefficients
            )
            physical_momentum = momentum / scale_factor
            uncoupled_energy = np.hypot(physical_momentum, mass)
            coupled_energy = neutrino_energy_local(
                physical_momentum, mass, field_value, parameters
            )
            uncoupled_velocity = physical_momentum / uncoupled_energy
            coupled_velocity = neutrino_group_velocity_local(
                physical_momentum, mass, field_value, parameters
            )
            energy_correction = coupled_energy - uncoupled_energy
            pressure_weight_correction = physical_momentum * (
                coupled_velocity - uncoupled_velocity
            )
            measure = (
                phase_space_factor
                * weights[index]
                * momentum**2
                / scale_factor**3
            )
            angular_average = 0.5 * np.sum(angular_weights * distribution)
            angular_shear = 0.5 * np.sum(
                angular_weights
                * (cosines**2 - 1.0 / 3.0)
                * distribution
            )
            projected["delta_rho_GeV4"] += (
                measure * energy_correction * angular_average
            )
            projected["delta_pressure_GeV4"] += (
                measure
                * pressure_weight_correction
                * angular_average
                / 3.0
            )
            projected["longitudinal_anisotropic_stress_GeV4"] += (
                measure * pressure_weight_correction * angular_shear
            )

        for moment_name, expected in projected.items():
            np.testing.assert_allclose(
                actual[moment_name], expected, rtol=1e-8, atol=1e-14
            )

    def test_neutrino_distribution_source_matches_stress_perturbations(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        scale_factor = 0.73
        comoving_momenta = np.array([0.12, 0.37, 0.81])
        comoving_weights = np.array([0.03, 0.11, 0.07])
        mass = 0.29
        degeneracy = 2.0
        moments = np.array(
            [
                [0.2 + 0.03j, -0.04j, 0.02 + 0.01j],
                [0.1 - 0.02j, 0.03 + 0.01j, -0.01j],
                [-0.05j, 0.02 - 0.03j, 0.04 + 0.01j],
            ]
        )

        distribution_source = neutrino_scalar_source_from_distribution_multipoles(
            comoving_momenta / scale_factor,
            comoving_weights / scale_factor,
            mass,
            moments,
            parameters,
            degeneracy,
        )
        stress_moments = uncoupled_kinetic_stress_multipoles(
            comoving_momenta,
            comoving_weights,
            mass,
            scale_factor,
            moments,
            degeneracy,
        )
        stress_source = isotropic_fluid_scalar_source_perturbation(
            stress_moments["delta_rho_GeV4"],
            stress_moments["delta_pressure_GeV4"],
            parameters,
        )

        self.assertAlmostEqual(distribution_source, stress_source, places=14)

    def test_photon_speed_log_derivative_matches_finite_difference(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=2e-3,
        )
        field_value = 3.0
        step = 1e-5
        finite_difference = (
            np.log(photon_speed_ratio(field_value + step, parameters))
            - np.log(photon_speed_ratio(field_value - step, parameters))
        ) / (2.0 * step)

        self.assertAlmostEqual(
            photon_speed_log_derivative(field_value, parameters),
            finite_difference,
            places=10,
        )

    def test_massless_neutrinos_reproduce_standard_neff_radiation_density(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
            neutrino_masses_GeV=(0.0, 0.0, 0.0),
            N_eff=3.046,
        )

        rho_neutrino, pressure_neutrino = relic_neutrino_energy_pressure(
            1.0, parameters
        )
        temperature_gamma = T_NU0_GEV / (4.0 / 11.0) ** (1.0 / 3.0)
        rho_gamma = pi**2 * temperature_gamma**4 / 15.0
        expected_ratio = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * 3.046

        self.assertAlmostEqual(rho_neutrino / rho_gamma, expected_ratio, places=12)
        self.assertAlmostEqual(pressure_neutrino / rho_neutrino, 1.0 / 3.0, places=12)

    def test_massive_neutrinos_transition_from_radiation_to_matter(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
            neutrino_masses_GeV=(0.02e-9, 0.02e-9, 0.02e-9),
            N_eff=3.046,
        )

        rho_early, pressure_early = relic_neutrino_energy_pressure(
            1e-8, parameters
        )
        rho_relativistic, _ = relic_neutrino_energy_pressure(1e-4, parameters)
        rho_half, _ = relic_neutrino_energy_pressure(0.5, parameters)
        rho_today, pressure_today = relic_neutrino_energy_pressure(1.0, parameters)

        self.assertLess(abs(rho_early / rho_relativistic / 1e16 - 1.0), 2e-5)
        self.assertAlmostEqual(pressure_early / rho_early, 1.0 / 3.0, places=5)
        self.assertAlmostEqual(rho_half / rho_today, 8.0, delta=0.02)
        self.assertLess(pressure_today / rho_today, 1e-3)

    def test_neutrino_density_satisfies_continuity_across_transition(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
            neutrino_masses_GeV=(0.06e-9, 0.0, 0.0),
            N_eff=3.046,
        )
        step = 1e-5

        for scale_factor in (1e-8, 1e-4, 0.3, 1.0):
            rho, pressure = relic_neutrino_energy_pressure(
                scale_factor, parameters
            )
            rho_minus, _ = relic_neutrino_energy_pressure(
                scale_factor * np.exp(-step), parameters
            )
            rho_plus, _ = relic_neutrino_energy_pressure(
                scale_factor * np.exp(step), parameters
            )
            derivative = (rho_plus - rho_minus) / (2.0 * step)
            residual = derivative + 3.0 * (rho + pressure)
            scale = abs(derivative) + 3.0 * (rho + pressure)

            self.assertLess(abs(residual) / scale, 1e-8)

    def test_coupled_neutrino_background_closes_energy_exchange(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
            neutrino_masses_GeV=(0.06e-9, 0.0, 0.0),
            N_eff=3.046,
        )
        scale_factor = 0.4
        field_value = 0.2
        field_step = 1e-4
        log_scale_step = 1e-5
        components = relic_neutrino_components(
            scale_factor, parameters, field_value
        )
        lower_field = relic_neutrino_components(
            scale_factor, parameters, field_value - field_step
        )["rho_GeV4"]
        upper_field = relic_neutrino_components(
            scale_factor, parameters, field_value + field_step
        )["rho_GeV4"]
        field_derivative = (upper_field - lower_field) / (2.0 * field_step)

        self.assertAlmostEqual(
            field_derivative / components["scalar_source_GeV3"],
            1.0,
            places=8,
        )

        lower_scale = relic_neutrino_components(
            scale_factor * np.exp(-log_scale_step), parameters, field_value
        )
        upper_scale = relic_neutrino_components(
            scale_factor * np.exp(log_scale_step), parameters, field_value
        )
        scale_derivative = (
            upper_scale["rho_GeV4"] - lower_scale["rho_GeV4"]
        ) / (2.0 * log_scale_step)
        continuity_residual = scale_derivative + 3.0 * (
            components["rho_GeV4"] + components["pressure_GeV4"]
        )
        continuity_scale = abs(scale_derivative) + 3.0 * (
            components["rho_GeV4"] + components["pressure_GeV4"]
        )
        self.assertLess(abs(continuity_residual) / continuity_scale, 1e-8)

        initial = FLRWInitialConditions(
            N_GeV=field_value,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.0,
            rho_gamma_i_GeV4=0.0,
            Theta_dot_i_GeV2=1e-80,
            scale_factor_i=scale_factor,
        )
        state = np.array([field_value, 0.0, 0.0, 0.0])
        background = background_quantities(0.0, state, parameters, initial)
        derivative = rhs_log_scale_factor(0.0, state, parameters, initial)

        self.assertEqual(
            background["scalar_source_neutrino_GeV3"],
            components["scalar_source_GeV3"],
        )
        self.assertAlmostEqual(
            derivative[1],
            -components["scalar_source_GeV3"] / background["H_GeV"],
            delta=abs(derivative[1]) * 1e-12,
        )

    def test_neutrino_uncoupled_enthalpy_feeds_clock_current(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
            neutrino_masses_GeV=(0.06e-9, 0.02e-9, 0.0),
            N_eff=3.046,
        )
        scale_factor = 0.4
        field_value = 0.2
        clock_velocity = 0.003 - 0.002j
        coupled = relic_neutrino_components(
            scale_factor, parameters, field_value
        )
        uncoupled = relic_neutrino_components(scale_factor, parameters, 0.0)

        self.assertEqual(
            coupled["rho_plus_pressure_uncoupled_GeV4"],
            uncoupled["rho_GeV4"] + uncoupled["pressure_GeV4"],
        )

        initial = FLRWInitialConditions(
            N_GeV=field_value,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.0,
            rho_gamma_i_GeV4=0.0,
            Theta_dot_i_GeV2=1e-80,
            scale_factor_i=scale_factor,
        )
        background = background_quantities(
            0.0,
            np.array([field_value, 0.0, 0.0, 0.0]),
            parameters,
            initial,
        )
        enthalpy = background[
            "rho_plus_pressure_neutrino_uncoupled_GeV4"
        ]
        self.assertEqual(
            enthalpy, coupled["rho_plus_pressure_uncoupled_GeV4"]
        )

        divergence = clock_interaction_current_divergence_perturbation(
            0.9,
            parameters.h_GeV_inv,
            field_value,
            0.12,
            enthalpy,
            enthalpy * clock_velocity,
            clock_velocity,
        )
        self.assertEqual(divergence, 0.0j)

    def test_neutrino_multipole_flux_feeds_clock_current_with_frame_slip(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
            neutrino_masses_GeV=(0.06e-9, 0.02e-9, 0.0),
            N_eff=3.046,
        )
        scale_factor = 0.4
        field_value = 0.2
        clock_velocity = 0.003 - 0.002j
        neutrino_velocity = -0.002 + 0.001j
        wavenumber = 0.9
        theta_prime = 0.12
        initial = FLRWInitialConditions(
            N_GeV=field_value,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.0,
            rho_gamma_i_GeV4=0.0,
            Theta_dot_i_GeV2=1e-80,
            scale_factor_i=scale_factor,
        )
        background = background_quantities(
            0.0,
            np.array([field_value, 0.0, 0.0, 0.0]),
            parameters,
            initial,
        )
        enthalpy = background[
            "rho_plus_pressure_neutrino_uncoupled_GeV4"
        ]
        comoving_momenta = T_NU0_GEV * np.array([0.5, 1.5, 3.0])
        comoving_weights = T_NU0_GEV * np.array([0.4, 0.8, 1.2])
        distribution_multipoles = np.zeros((3, 3), dtype=complex)
        distribution_multipoles[:, 1] = 1.0
        unit_flux = uncoupled_kinetic_stress_multipoles(
            comoving_momenta,
            comoving_weights,
            0.02e-9,
            scale_factor,
            distribution_multipoles,
        )["longitudinal_flux_GeV4"]
        distribution_multipoles[:, 1] *= (
            enthalpy * neutrino_velocity / unit_flux
        )
        neutrino_stress = uncoupled_kinetic_stress_multipoles(
            comoving_momenta,
            comoving_weights,
            0.02e-9,
            scale_factor,
            distribution_multipoles,
        )

        actual = modeled_matter_clock_current_divergence_perturbation(
            wavenumber,
            field_value,
            theta_prime,
            clock_velocity,
            parameters,
            background,
            neutrino_stress,
        )
        expected = (
            1j
            * wavenumber
            * parameters.h_GeV_inv
            * field_value
            / theta_prime
            * enthalpy
            * (clock_velocity - neutrino_velocity)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-80)

    def test_clock_current_combines_dust_photon_and_neutrino_fluxes(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
            neutrino_masses_GeV=(0.06e-9, 0.02e-9, 0.0),
            N_eff=3.046,
        )
        scale_factor = 0.5
        field_value = 0.2
        clock_velocity = 0.003 - 0.002j
        dust_velocity = -0.001 + 0.0005j
        photon_velocity = 0.002 + 0.001j
        neutrino_velocity = -0.002 + 0.001j
        wavenumber = 0.9
        theta_prime = 0.12
        initial = FLRWInitialConditions(
            N_GeV=0.1,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=1e-51,
            rho_gamma_i_GeV4=2e-51,
            Theta_dot_i_GeV2=1e-80,
            scale_factor_i=scale_factor,
        )
        background = background_quantities(
            0.0,
            np.array([field_value, 0.0, 0.0, 0.0]),
            parameters,
            initial,
        )
        dust_enthalpy = background["rho_m_bare_GeV4"]
        photon_enthalpy = (
            background["rho_gamma_bare_GeV4"]
            + background["pressure_gamma_bare_GeV4"]
        )
        neutrino_enthalpy = background[
            "rho_plus_pressure_neutrino_uncoupled_GeV4"
        ]

        dust_stress = perfect_fluid_scalar_stress_moments(
            dust_enthalpy, 0.0, 0.0, dust_velocity, 0.0
        )
        photon_momenta = T_NU0_GEV * np.array([0.5, 1.5, 3.0])
        photon_weights = T_NU0_GEV * np.array([0.4, 0.8, 1.2])
        photon_multipoles = np.zeros((3, 3), dtype=complex)
        photon_multipoles[:, 1] = 1.0
        unit_photon_flux = photon_maxwell_stress_multipoles(
            photon_momenta,
            photon_weights,
            scale_factor,
            field_value,
            photon_multipoles,
            parameters,
            degeneracy=2.0,
        )["longitudinal_flux_GeV4"]
        photon_multipoles[:, 1] *= (
            photon_enthalpy * photon_velocity / unit_photon_flux
        )
        photon_stress = photon_maxwell_stress_multipoles(
            photon_momenta,
            photon_weights,
            scale_factor,
            field_value,
            photon_multipoles,
            parameters,
            degeneracy=2.0,
        )

        neutrino_momenta = T_NU0_GEV * np.array([0.5, 1.5, 3.0])
        neutrino_weights = T_NU0_GEV * np.array([0.4, 0.8, 1.2])
        neutrino_multipoles = np.zeros((3, 3), dtype=complex)
        neutrino_multipoles[:, 1] = 1.0
        unit_neutrino_flux = uncoupled_kinetic_stress_multipoles(
            neutrino_momenta,
            neutrino_weights,
            0.02e-9,
            scale_factor,
            neutrino_multipoles,
        )["longitudinal_flux_GeV4"]
        neutrino_multipoles[:, 1] *= (
            neutrino_enthalpy * neutrino_velocity / unit_neutrino_flux
        )
        neutrino_stress = uncoupled_kinetic_stress_multipoles(
            neutrino_momenta,
            neutrino_weights,
            0.02e-9,
            scale_factor,
            neutrino_multipoles,
        )
        total_stress = sum_scalar_stress_moments(
            dust_stress, photon_stress, neutrino_stress
        )

        actual = modeled_matter_clock_current_divergence_perturbation(
            wavenumber,
            field_value,
            theta_prime,
            clock_velocity,
            parameters,
            background,
            total_stress,
        )
        expected = (
            1j
            * wavenumber
            * parameters.h_GeV_inv
            * field_value
            / theta_prime
            * (
                dust_enthalpy * (clock_velocity - dust_velocity)
                + photon_enthalpy * (clock_velocity - photon_velocity)
                + neutrino_enthalpy * (clock_velocity - neutrino_velocity)
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-80)

    def test_background_assembles_uncoupled_matter_enthalpy(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
            neutrino_masses_GeV=(0.06e-9, 0.02e-9, 0.0),
            N_eff=3.046,
        )
        initial = FLRWInitialConditions(
            N_GeV=0.1,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.03,
            rho_gamma_i_GeV4=0.04,
            Theta_dot_i_GeV2=1e-80,
        )
        state = np.array([0.2, 0.0, 0.0, 0.0])
        background = background_quantities(
            np.log(0.8), state, parameters, initial
        )
        expected = (
            background["rho_m_bare_GeV4"]
            + 4.0 / 3.0 * background["rho_gamma_bare_GeV4"]
            + background["rho_plus_pressure_neutrino_uncoupled_GeV4"]
        )

        self.assertAlmostEqual(
            background["pressure_gamma_bare_GeV4"],
            background["rho_gamma_bare_GeV4"] / 3.0,
            places=15,
        )
        self.assertAlmostEqual(
            background["rho_plus_pressure_matter_uncoupled_GeV4"],
            expected,
            places=15,
        )

    def test_modeled_matter_clock_current_composes_background_and_flux(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
        )
        field_value = 0.2
        clock_velocity = 0.003 - 0.002j
        matter_velocity = -0.004 + 0.001j
        wavenumber = 0.9
        theta_prime = 0.12
        initial = FLRWInitialConditions(
            N_GeV=0.1,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.03,
            rho_gamma_i_GeV4=0.04,
            Theta_dot_i_GeV2=1e-80,
        )
        background = background_quantities(
            0.0,
            np.array([field_value, 0.0, 0.0, 0.0]),
            parameters,
            initial,
        )
        enthalpy = background["rho_plus_pressure_matter_uncoupled_GeV4"]
        total_stress = sum_scalar_stress_moments(
            {
                "delta_rho_GeV4": 0.0j,
                "delta_pressure_GeV4": 0.0j,
                "longitudinal_flux_GeV4": 0.35 * enthalpy * matter_velocity,
                "longitudinal_anisotropic_stress_GeV4": 0.0j,
            },
            {
                "delta_rho_GeV4": 0.0j,
                "delta_pressure_GeV4": 0.0j,
                "longitudinal_flux_GeV4": 0.65 * enthalpy * matter_velocity,
                "longitudinal_anisotropic_stress_GeV4": 0.0j,
            },
        )

        actual = modeled_matter_clock_current_divergence_perturbation(
            wavenumber,
            field_value,
            theta_prime,
            clock_velocity,
            parameters,
            background,
            total_stress,
        )
        expected = (
            1j
            * wavenumber
            * parameters.h_GeV_inv
            * field_value
            / theta_prime
            * enthalpy
            * (clock_velocity - matter_velocity)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

    def test_clock_current_uses_bare_dust_flux_not_effective_mass_flux(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.08,
            h_GeV_inv=0.03,
        )
        bare_density = 0.06
        field_value = 0.2
        delta_field = 0.01 - 0.002j
        dust_velocity = -0.003 + 0.002j
        wavenumber = 0.9
        theta_prime = 0.12
        background = {
            "rho_plus_pressure_matter_uncoupled_GeV4": bare_density
        }
        bare_dust = perfect_fluid_scalar_stress_moments(
            bare_density,
            0.0,
            0.04 - 0.01j,
            dust_velocity,
            0.0,
        )
        effective_dust = coupled_pressureless_matter_stress_moments(
            bare_density,
            0.04 - 0.01j,
            dust_velocity,
            field_value,
            delta_field,
            parameters,
            clock_velocity_projection=dust_velocity,
        )

        bare_divergence = modeled_matter_clock_current_divergence_perturbation(
            wavenumber,
            field_value,
            theta_prime,
            dust_velocity,
            parameters,
            background,
            bare_dust,
        )
        effective_divergence = modeled_matter_clock_current_divergence_perturbation(
            wavenumber,
            field_value,
            theta_prime,
            dust_velocity,
            parameters,
            background,
            effective_dust,
        )
        expected_effective_divergence = (
            -1j
            * wavenumber
            * parameters.h_GeV_inv
            * field_value
            / theta_prime
            * parameters.eta_GeV_inv
            * field_value
            * bare_density
            * dust_velocity
        )

        self.assertEqual(bare_divergence, 0.0j)
        np.testing.assert_allclose(
            effective_divergence,
            expected_effective_divergence,
            rtol=1e-14,
            atol=0.0,
        )

    def test_extra_neff_does_not_rescale_massive_states(self):
        common_parameters = {
            "m_phi_GeV": 0.0,
            "lambda_N": 0.0,
            "g_GeV_inv": 0.0,
            "h_GeV_inv": 0.0,
            "neutrino_masses_GeV": (0.06e-9, 0.0, 0.0),
        }
        standard = FLRWParameters(**common_parameters, N_eff=3.0)
        extra_radiation = FLRWParameters(**common_parameters, N_eff=3.046)

        standard_components = relic_neutrino_components(1.0, standard)
        extra_components = relic_neutrino_components(1.0, extra_radiation)

        self.assertAlmostEqual(
            standard_components["rho_massive_GeV4"],
            extra_components["rho_massive_GeV4"],
            places=14,
        )
        self.assertGreater(
            extra_components["rho_massless_GeV4"],
            standard_components["rho_massless_GeV4"],
        )

    def test_neff_cannot_be_less_than_explicit_species_count(self):
        with self.assertRaisesRegex(ValueError, "N_eff must be at least three"):
            FLRWParameters(
                m_phi_GeV=0.0,
                lambda_N=0.0,
                g_GeV_inv=0.0,
                h_GeV_inv=0.0,
                neutrino_masses_GeV=(0.06e-9, 0.0, 0.0),
                N_eff=2.99,
            )

    def test_slow_roll_no_lambda_history_reaches_present_normalization(self):
        result = solve_slow_roll_history(0.2, sample_count=31)
        today = result["today"]

        self.assertEqual(result["parameters"].rho_lambda_GeV4, 0.0)
        self.assertAlmostEqual(today["H_over_H0"], 1.0, places=7)
        self.assertAlmostEqual(today["omega_m_total"], 0.315, places=6)
        self.assertAlmostEqual(today["omega_nu_massive"], 0.0014042, places=6)
        self.assertGreater(today["omega_nu_massless"], 0.0)
        self.assertAlmostEqual(
            today["omega_nu"],
            today["omega_nu_massive"] + today["omega_nu_massless"],
            places=12,
        )
        self.assertAlmostEqual(
            today["omega_cb"] + today["omega_nu_massive"], 0.315, places=6
        )
        self.assertAlmostEqual(today["m_eff_over_H0"], 0.2, places=12)
        self.assertLess(abs(today["w_N"] + 1.0), 0.02)
        self.assertLess(today["max_abs_xi"], 1.0)
        self.assertLess(today["delta_c_gamma"], 0.0)
        self.assertAlmostEqual(
            today["delta_c_gamma"] / today["xi"], -1.0, places=10
        )
        self.assertLess(result["max_fractional_H_difference"], 0.02)
        self.assertLess(result["max_fractional_distance_difference"], 0.01)
        recombination_index = int(
            np.argmin(np.abs(result["history"]["a"] - 1.0 / 1101.0))
        )
        self.assertLess(
            abs(
                result["history"]["H_over_H0"][recombination_index]
                / result["history"]["H_reference_over_H0"][recombination_index]
                - 1.0
            ),
            1e-8,
        )
        self.assertTrue(np.any(np.isclose(result["history"]["a"], 0.5)))
        self.assertAlmostEqual(result["history"]["a"][-1], 1.0, places=12)

    def test_fiducial_branch_density_fractions_are_distinct(self):
        with redirect_stdout(StringIO()):
            diagnostics = run_fiducial_check()

        self.assertLess(diagnostics["omega_N_sourced_minimum"], 1e-18)
        self.assertAlmostEqual(diagnostics["omega_lambda"], 0.684946, places=6)
        self.assertAlmostEqual(diagnostics["omega_N_no_lambda"], 0.684946, places=6)
        self.assertAlmostEqual(diagnostics["h_ratio_lambda"], 1.0, places=12)
        self.assertAlmostEqual(diagnostics["h_ratio_no_lambda"], 1.0, places=12)

    def test_vacuum_density_normalizes_hubble_rate(self):
        target_H = 0.5
        rho_matter = 0.3
        rho_radiation = 0.2
        rho_clock = 0.5 * 0.1**2
        rho_lambda = 3.0 * target_H**2 - rho_matter - rho_radiation - rho_clock
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
            rho_lambda_GeV4=rho_lambda,
            Mpl_GeV=1.0,
        )
        initial = FLRWInitialConditions(
            N_GeV=0.0,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=rho_matter,
            rho_gamma_i_GeV4=rho_radiation,
            Theta_dot_i_GeV2=0.1,
        )

        background = background_quantities(0.0, (0.0, 0.0, 0.0, 0.0), parameters, initial)

        self.assertAlmostEqual(background["H_GeV"], target_H, places=14)
        self.assertEqual(background["pressure_lambda_GeV4"], -rho_lambda)

    def test_initial_conditions_are_anchored_at_log_a_zero(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
            Mpl_GeV=1.0,
        )
        initial = FLRWInitialConditions(
            N_GeV=0.0,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.3,
            rho_gamma_i_GeV4=0.2,
            Theta_dot_i_GeV2=0.1,
        )

        with self.assertRaisesRegex(ValueError, "log_a = 0"):
            solve_background(parameters, initial, (0.1, 1.0))

    def test_uncoupled_components_have_standard_dilution(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.0,
            h_GeV_inv=0.0,
            Mpl_GeV=1.0,
        )
        initial = FLRWInitialConditions(
            N_GeV=0.0,
            Ndot_GeV2=0.0,
            rho_m_bare_i_GeV4=0.3,
            rho_gamma_i_GeV4=0.2,
            Theta_dot_i_GeV2=0.1,
        )

        solution = solve_background(
            parameters,
            initial,
            (0.0, 1.0),
            atol=(1e-12, 1e-12, 1e-12, 1e-12),
        )
        self.assertTrue(solution.success, solution.message)
        x = solution.t[-1]
        state = solution.y[:, -1]
        background = background_quantities(x, state, parameters, initial)

        self.assertAlmostEqual(state[0], 0.0, places=12)
        self.assertAlmostEqual(state[1], 0.0, places=12)
        self.assertGreater(state[2], initial.Theta_i_GeV)
        self.assertGreater(state[3], initial.t_i_GeVinv)
        self.assertAlmostEqual(
            background["rho_m_GeV4"], 0.3 * np.exp(-3.0 * x), places=12
        )
        self.assertAlmostEqual(
            background["rho_gamma_GeV4"], 0.2 * np.exp(-4.0 * x), places=12
        )
        self.assertAlmostEqual(
            background["rho_theta_GeV4"], 0.005 * np.exp(-6.0 * x), places=12
        )
        self.assertAlmostEqual(
            3.0 * background["H_GeV"] ** 2,
            background["rho_total_GeV4"],
            places=12,
        )

    def test_coupled_energy_exchange_cancels(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.4,
            lambda_N=0.3,
            g_GeV_inv=0.3,
            h_GeV_inv=0.2,
            Z_theta=1.2,
            Mpl_GeV=4.0,
        )
        initial = FLRWInitialConditions(
            N_GeV=0.2,
            Ndot_GeV2=0.07,
            rho_m_bare_i_GeV4=0.4,
            rho_gamma_i_GeV4=0.15,
            Theta_dot_i_GeV2=0.11,
        )
        x = 0.37
        state = np.array((0.23, 0.05, initial.Theta_i_GeV, initial.t_i_GeVinv))
        background = background_quantities(x, state, parameters, initial)
        H = background["H_GeV"]
        Ndot = state[1]
        Nddot = H * rhs_log_scale_factor(x, state, parameters, initial)[1]

        rho_N_dot = Ndot * (
            Nddot + potential_derivative(state[0], parameters)
        )
        rho_m_dot = (
            -3.0 * H * background["rho_m_GeV4"]
            + parameters.eta_GeV_inv
            * Ndot
            * background["rho_m_bare_GeV4"]
        )
        rho_gamma_dot = (
            -4.0 * H * background["rho_gamma_GeV4"]
            - parameters.h_GeV_inv
            * Ndot
            * background["rho_gamma_bare_GeV4"]
        )
        rho_theta_dot = -6.0 * H * background["rho_theta_GeV4"]
        rho_total_dot = rho_N_dot + rho_m_dot + rho_gamma_dot + rho_theta_dot
        rho_plus_pressure = (
            state[1] ** 2
            + background["rho_m_GeV4"]
            + 4.0 * background["rho_gamma_GeV4"] / 3.0
            + 2.0 * background["rho_theta_GeV4"]
        )
        residual = rho_total_dot + 3.0 * H * rho_plus_pressure

        self.assertAlmostEqual(residual, 0.0, delta=1e-12)

    def test_coupled_background_integrates_inside_photon_domain(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.5,
            lambda_N=0.1,
            g_GeV_inv=0.05,
            h_GeV_inv=0.04,
            Mpl_GeV=1.0,
        )
        initial = FLRWInitialConditions(
            N_GeV=0.1,
            Ndot_GeV2=0.01,
            rho_m_bare_i_GeV4=0.2,
            rho_gamma_i_GeV4=0.1,
            Theta_dot_i_GeV2=0.1,
        )

        solution = solve_background(parameters, initial, (0.0, 0.3))
        self.assertTrue(solution.success, solution.message)
        for x, state in zip(solution.t, solution.y.T):
            background = background_quantities(x, state, parameters, initial)
            self.assertLess(abs(background["xi"]), 1.0)
            self.assertGreater(background["H_GeV"], 0.0)
            self.assertGreater(background["rho_total_GeV4"], 0.0)


if __name__ == "__main__":
    unittest.main()