import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import simpson

import solar_thinshell
import solar_radial_validation

from solartest import run_checks


class SolarThinShellTests(unittest.TestCase):
    def test_conversions_minima_and_screening_checks(self):
        with redirect_stdout(StringIO()):
            run_checks()

    def test_massless_scalar_tensor_gamma_matches_f_r_limit(self):
        beta = 1.0 / np.sqrt(6.0)

        self.assertAlmostEqual(
            solar_thinshell.massless_scalar_tensor_gamma(beta, beta),
            0.5,
        )

    def test_search_stops_at_first_candidate_and_resumes_from_csv(self):
        non_candidate = [
            1e-20,
            0.0,
            1e-20,
            1e-10,
            1e-6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2e-6,
            3e-5,
        ]
        candidate = [
            2e-20,
            0.0,
            2e-20,
            1e-10,
            1e-6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            5e-7,
            1e-5,
        ]

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "search.csv"
            with redirect_stdout(StringIO()):
                with patch.object(
                    solar_thinshell,
                    "_evaluate_parameter_point",
                    side_effect=[non_candidate, candidate],
                ) as evaluate:
                    result = solar_thinshell.search_first_candidate(
                        output_path=output_path,
                        g_values=[1e-20, 2e-20, 3e-20],
                        lambda_values=[1e-10],
                        m0_values_eV=[1e-6],
                        resume=False,
                    )
            self.assertEqual(result, candidate)
            self.assertEqual(evaluate.call_count, 2)

            with output_path.open(
                "r", newline="", encoding="utf-8"
            ) as saved:
                rows = list(csv.reader(saved))
            self.assertEqual(len(rows), 3)

            with redirect_stdout(StringIO()):
                with patch.object(
                    solar_thinshell,
                    "_evaluate_parameter_point",
                    side_effect=AssertionError("Resume should reuse the CSV."),
                ):
                    resumed = solar_thinshell.search_first_candidate(
                        output_path=output_path,
                        g_values=[1e-20, 2e-20, 3e-20],
                        lambda_values=[1e-10],
                        m0_values_eV=[1e-6],
                        resume=True,
                    )
            self.assertEqual(resumed, candidate)

    def test_cycle_search_continues_after_candidates_and_resumes(self):
        def candidate_for_point(g, lam, m0_eV, kappa_over_lambda):
            return [
                g,
                kappa_over_lambda,
                g - kappa_over_lambda,
                lam,
                m0_eV,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5e-7,
                1e-5,
            ]

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cycles.csv"
            with redirect_stdout(StringIO()):
                with patch.object(
                    solar_thinshell,
                    "_evaluate_parameter_point",
                    side_effect=candidate_for_point,
                ) as evaluate:
                    summary = solar_thinshell.search_parameter_cycles(
                        output_path=output_path,
                        g_values=[1e-20, 2e-20],
                        lambda_values=[1e-10],
                        m0_values_eV=[1e-6],
                        cycles=2,
                        batch_size=3,
                        random_seed=11,
                        cycle_pause_seconds=0.0,
                        resume=False,
                    )
            self.assertEqual(evaluate.call_count, 5)
            self.assertEqual(summary["cycles_completed"], 2)
            self.assertEqual(summary["points_evaluated"], 5)
            self.assertEqual(summary["candidate_points"], 5)

            with output_path.open(
                "r", newline="", encoding="utf-8"
            ) as saved:
                rows = list(csv.reader(saved))
            self.assertEqual(len(rows), 6)

            with redirect_stdout(StringIO()):
                with patch.object(
                    solar_thinshell,
                    "_evaluate_parameter_point",
                    side_effect=candidate_for_point,
                ):
                    resumed = solar_thinshell.search_parameter_cycles(
                        output_path=output_path,
                        g_values=[1e-20, 2e-20],
                        lambda_values=[1e-10],
                        m0_values_eV=[1e-6],
                        cycles=1,
                        batch_size=3,
                        random_seed=11,
                        cycle_pause_seconds=0.0,
                        resume=True,
                    )
            self.assertEqual(resumed["cycles_completed"], 1)
            self.assertEqual(resumed["points_evaluated"], 3)
            self.assertEqual(resumed["candidate_points"], 3)
            with output_path.open(
                "r", newline="", encoding="utf-8"
            ) as saved:
                rows = list(csv.reader(saved))
            self.assertEqual(len(rows), 9)

    def test_radial_solver_recovers_uniform_density_minimum(self):
        radii = np.linspace(0.0, 1.0, 21)
        densities = np.ones_like(radii)
        result = solar_radial_validation.solve_radial_profile(
            1e-40,
            1e-10,
            1e-10,
            radii,
            densities,
            rho_out_kgm3=1.0,
            observation_radius_m=solar_thinshell.R_sun,
        )
        expected = solar_thinshell.find_N0(
            solar_thinshell.rho_kgm3_to_GeV4(1.0),
            1e-40,
            1e-10,
            1e-19,
        )

        field_error = max(
            abs(value - expected) for value in result["field_GeV"]
        )
        self.assertLess(field_error, max(abs(expected) * 1e-5, 1e-25))
        self.assertLess(result["surface_force_over_newtonian"], 1e-20)

    def test_exterior_linearity_check_does_not_hide_term_cancellation(self):
        ratio = solar_radial_validation._exterior_nonlinearity_ratio(
            -3.0,
            1.0,
            1.0,
            1.0,
        )

        self.assertEqual(ratio, 18.0)

    def test_default_solar_profile_matches_solar_mass(self):
        radii, densities = solar_radial_validation.load_density_profile(
            solar_radial_validation.DEFAULT_SSM_PROFILE
        )
        radius_m = radii * solar_thinshell.R_sun
        profile_mass = simpson(
            4.0 * np.pi * radius_m**2 * densities,
            x=radius_m,
        )

        self.assertAlmostEqual(
            profile_mass / solar_thinshell.M_sun,
            1.0,
            delta=0.005,
        )

    def test_default_radial_force_matches_refined_aag21_solution(self):
        radii, densities = solar_radial_validation.load_density_profile(
            solar_radial_validation.DEFAULT_SSM_PROFILE
        )
        parameters = (
            3.497100734152803e-16,
            1.28592102767713e-63,
            1.9036787651069732e-29,
            radii,
            densities,
        )
        default_result = solar_radial_validation.solve_radial_profile(
            *parameters,
            rho_out_kgm3=1e-21,
            observation_radius_m=solar_radial_validation.AU_METERS,
        )
        refined_result = solar_radial_validation.solve_radial_profile(
            *parameters,
            rho_out_kgm3=1e-21,
            observation_radius_m=solar_radial_validation.AU_METERS,
            tolerance=1e-8,
            max_nodes=100000,
        )
        refined_force = refined_result["force_over_newtonian_at_observation"]
        relative_force_error = abs(
            default_result["force_over_newtonian_at_observation"]
            - refined_force
        ) / abs(refined_force)

        self.assertLess(relative_force_error, 1e-3)
        self.assertGreater(
            default_result["surface_exterior_nonlinearity_ratio"], 1e10
        )
        self.assertLessEqual(
            default_result["exterior_nonlinearity_ratio"],
            solar_radial_validation.MAX_EXTERIOR_NONLINEARITY_RATIO,
        )
        self.assertGreater(
            default_result["exterior_match_radius_m"],
            solar_radial_validation.AU_METERS,
        )
        self.assertNotAlmostEqual(
            default_result["beta_scalar_surface"],
            default_result["beta_scalar_match"],
            delta=1e-7,
        )

        probe_radii = np.linspace(0.0, 1.0, 41)
        probe = solar_radial_validation.solve_radial_profile(
            *parameters[:3],
            probe_radii,
            np.full_like(probe_radii, 1000.0),
            rho_out_kgm3=1e-21,
            observation_radius_m=solar_radial_validation.AU_METERS,
            body_radius_m=1.0,
        )
        with self.assertRaisesRegex(ValueError, "se solapan"):
            solar_radial_validation.finite_range_scalar_force_ratio(
                default_result,
                probe,
                solar_radial_validation.AU_METERS,
            )

    def test_independent_body_charges_recover_finite_range_force(self):
        radii = np.linspace(0.0, 1.0, 21)
        densities = np.full_like(radii, 1000.0)
        parameters = (1e-30, 0.0, 1e-8, radii, densities)
        source = solar_radial_validation.solve_radial_profile(
            *parameters,
            rho_out_kgm3=1e-21,
            observation_radius_m=10.0,
            body_radius_m=1.0,
        )
        probe = solar_radial_validation.solve_radial_profile(
            *parameters,
            rho_out_kgm3=1e-21,
            observation_radius_m=10.0,
            body_radius_m=0.1,
        )

        beta_bare = parameters[0] * solar_thinshell.Mpl_GeV
        source_mR = (
            source["outside_effective_mass_GeV"]
            * source["body_radius_m"]
            * solar_thinshell.m_to_GeVinv
        )
        probe_mR = (
            probe["outside_effective_mass_GeV"]
            * probe["body_radius_m"]
            * solar_thinshell.m_to_GeVinv
        )
        source_form_factor = (
            3.0
            * (source_mR * np.cosh(source_mR) - np.sinh(source_mR))
            * np.exp(-source_mR)
            / source_mR**3
        )
        probe_form_factor = (
            3.0
            * (probe_mR * np.cosh(probe_mR) - np.sinh(probe_mR))
            * np.exp(-probe_mR)
            / probe_mR**3
        )
        self.assertAlmostEqual(
            source["beta_scalar_surface"] / beta_bare,
            source_form_factor,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            probe["beta_scalar_surface"] / beta_bare,
            probe_form_factor,
            delta=1e-3,
        )

        separation_m = 10.0
        outside_mass = source["outside_effective_mass_GeV"]
        separation_GeVinv = separation_m * solar_thinshell.m_to_GeVinv
        match_gap_GeVinv = (
            separation_m
            - source["exterior_match_radius_m"]
            - probe["exterior_match_radius_m"]
        ) * solar_thinshell.m_to_GeVinv
        expected_ratio = (
            2.0
            * source["beta_scalar_match"]
            * probe["beta_scalar_match"]
            * (1.0 + outside_mass * separation_GeVinv)
            * np.exp(-outside_mass * match_gap_GeVinv)
        )
        force_ratio = solar_radial_validation.finite_range_scalar_force_ratio(
            source,
            probe,
            separation_m,
        )
        self.assertAlmostEqual(force_ratio / expected_ratio, 1.0, delta=0.01)
        with self.assertRaisesRegex(ValueError, "finita y positiva"):
            solar_radial_validation.finite_range_scalar_force_ratio(
                source,
                probe,
                0.0,
            )

    def test_small_body_charge_matches_refined_profile(self):
        radii = np.linspace(0.0, 1.0, 41)
        densities = np.full_like(radii, 1000.0)
        parameters = (
            3.497100734152803e-16,
            1.28592102767713e-63,
            1.9036787651069732e-29,
            radii,
            densities,
        )
        default_result = solar_radial_validation.solve_radial_profile(
            *parameters,
            rho_out_kgm3=1e-21,
            observation_radius_m=10.0,
            body_radius_m=1.0,
        )
        refined_result = solar_radial_validation.solve_radial_profile(
            *parameters,
            rho_out_kgm3=1e-21,
            observation_radius_m=10.0,
            tolerance=1e-8,
            max_nodes=100000,
            body_radius_m=1.0,
        )

        beta_bare = parameters[0] * solar_thinshell.Mpl_GeV
        self.assertAlmostEqual(
            default_result["beta_scalar_surface"] / beta_bare,
            1.0,
            delta=0.002,
        )
        self.assertAlmostEqual(
            default_result["beta_scalar_surface"]
            / refined_result["beta_scalar_surface"],
            1.0,
            delta=1e-4,
        )

    def test_weak_field_metric_recovers_gr_for_uncoupled_body(self):
        radii = np.linspace(0.0, 1.0, 101)
        densities = np.full_like(radii, 1000.0)
        observation_radius_m = 3.0
        result = solar_radial_validation.solve_radial_profile(
            0.0,
            0.0,
            1e-8,
            radii,
            densities,
            rho_out_kgm3=1.0,
            observation_radius_m=observation_radius_m,
            body_radius_m=1.0,
        )

        metric = result["weak_field_metric_at_observation"]
        radius_GeVinv = observation_radius_m * solar_thinshell.m_to_GeVinv
        expected_potential = -result["bare_mass_GeV"] / (
            8.0
            * np.pi
            * solar_thinshell.Mpl_GeV**2
            * radius_GeVinv
        )
        self.assertAlmostEqual(metric["Phi"] / expected_potential, 1.0, delta=1e-4)
        self.assertAlmostEqual(metric["Psi"] / expected_potential, 1.0, delta=1e-4)
        self.assertAlmostEqual(metric["g_rr_areal"], 1.0 - 2.0 * expected_potential, delta=1e-12)
        self.assertAlmostEqual(metric["g_tt"], -(1.0 + 2.0 * expected_potential), delta=1e-12)

    def test_candidate_radial_solution_on_n3_polytrope(self):
        radii, densities = solar_radial_validation.solar_n3_polytrope_profile(
            samples=201
        )
        result = solar_radial_validation.solve_radial_profile(
            7.6115117e-17,
            5.3696576e-32,
            8.6323473e-12,
            radii,
            densities,
        )
        refined_radii, refined_densities = (
            solar_radial_validation.solar_n3_polytrope_profile(samples=501)
        )
        refined_result = solar_radial_validation.solve_radial_profile(
            7.6115117e-17,
            5.3696576e-32,
            8.6323473e-12,
            refined_radii,
            refined_densities,
        )

        self.assertEqual(radii[0], 0.0)
        self.assertEqual(radii[-1], 1.0)
        self.assertGreater(densities[0], densities[-1])
        self.assertEqual(densities[-1], 0.0)
        self.assertLess(result["center_field_GeV"], 0.0)
        self.assertTrue(np.isfinite(result["surface_force_over_newtonian"]))
        self.assertLess(result["surface_force_over_newtonian"], 1e-8)
        relative_force_change = abs(
            result["surface_force_over_newtonian"]
            - refined_result["surface_force_over_newtonian"]
        ) / refined_result["surface_force_over_newtonian"]
        self.assertLess(relative_force_change, 1e-3)
        self.assertEqual(
            result["force_over_newtonian_at_observation"], 0.0
        )
        self.assertGreaterEqual(min(result["density_kgm3"]), 0.0)

    def test_cli_defaults_to_bundled_aag21_profile(self):
        profile_result = {
            "profile_name": solar_radial_validation.DEFAULT_SSM_PROFILE_NAME,
            "center_field_GeV": 0.0,
            "surface_field_GeV": 0.0,
            "surface_force_over_newtonian": 0.0,
            "force_over_newtonian_at_observation": 0.0,
            "beta_scalar_surface": 0.0,
            "beta_scalar_match": 0.0,
            "surface_exterior_nonlinearity_ratio": 0.0,
            "exterior_nonlinearity_ratio": 0.0,
            "exterior_match_radius_m": solar_thinshell.R_sun,
            "body_radius_m": solar_thinshell.R_sun,
            "weak_field_metric_at_observation": {
                "Phi": 0.0,
                "Psi": 0.0,
                "g_tt": -1.0,
                "g_rr_areal": 1.0,
            },
        }
        with patch.object(
            solar_radial_validation,
            "solve_radial_profile",
            return_value=profile_result,
        ) as solve, patch(
            "sys.argv",
            [
                "solar_radial_validation.py",
                "--g",
                "7.6115117e-17",
                "--lambda",
                "5.3696576e-32",
                "--m0-eV",
                "8.6323473e-12",
            ],
        ), redirect_stdout(StringIO()) as output:
            solar_radial_validation._main()

        radii, densities = solve.call_args.args[3:5]
        self.assertEqual(radii.size, 2001)
        self.assertEqual(radii[0], 0.0)
        self.assertEqual(radii[-1], 1.0)
        self.assertAlmostEqual(densities[0], 148487.54, delta=0.1)
        self.assertAlmostEqual(densities[-1], 0.00017092674, delta=1e-11)
        self.assertEqual(
            solve.call_args.kwargs["profile_name"],
            solar_radial_validation.DEFAULT_SSM_PROFILE_NAME,
        )
        self.assertIn(
            solar_radial_validation.DEFAULT_SSM_PROFILE_NAME,
            output.getvalue(),
        )
        self.assertIn("Potenciales métricos débiles", output.getvalue())
        self.assertIn("Carga escalar superficial", output.getvalue())
        self.assertIn("Carga escalar normalizada en el empalme", output.getvalue())