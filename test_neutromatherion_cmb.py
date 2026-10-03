import importlib.util
import unittest

import numpy as np

from flrw_perturbations import (
    finite_h_thomson_opacity_per_conformal_time,
    standard_thomson_opacity_per_conformal_time,
)
from flrw_coupled import FLRWParameters
from neutromatherion_cmb import (
    build_camb_parameters,
    finite_h_opacity_history_with_fixed_xe,
    max_background_relative_error,
    standard_recombination_history_from_camb,
)


class StandardRecombinationHistoryTests(unittest.TestCase):
    def test_adapter_preserves_camb_fields_and_units(self):
        class FakeCambResults:
            def get_background_redshift_evolution(self, redshifts, vars, format):
                self.requested_redshifts = redshifts.copy()
                self.requested_vars = vars
                self.requested_format = format
                return {
                    "x_e": np.array([1.16, 0.14]),
                    "opacity": np.array([4.5e-7, 0.068]),
                }

        fake_results = FakeCambResults()
        redshifts = np.array([0.0, 1100.0])

        history = standard_recombination_history_from_camb(
            fake_results, redshifts
        )

        np.testing.assert_array_equal(fake_results.requested_redshifts, redshifts)
        self.assertEqual(fake_results.requested_vars, ["x_e", "opacity"])
        self.assertEqual(fake_results.requested_format, "dict")
        np.testing.assert_allclose(history["scale_factor"], 1.0 / (1.0 + redshifts))
        np.testing.assert_array_equal(
            history["ionization_fraction_per_hydrogen"], [1.16, 0.14]
        )
        np.testing.assert_array_equal(history["opacity_Mpc_inv"], [4.5e-7, 0.068])
        np.testing.assert_allclose(
            history["opacity_GeV"],
            np.array([4.5e-7, 0.068])
            * 1.973269804e-16
            / 3.0856775814913673e22,
            rtol=0.0,
            atol=0.0,
        )

    def test_adapter_rejects_invalid_redshift(self):
        with self.assertRaises(ValueError):
            standard_recombination_history_from_camb(object(), [-1.0])

    def test_finite_h_rate_can_use_fixed_camb_ionization_history(self):
        parameters = FLRWParameters(
            m_phi_GeV=0.0,
            lambda_N=0.0,
            g_GeV_inv=0.03,
            h_GeV_inv=0.02,
        )
        standard_history = {
            "redshift": np.array([0.0, 1000.0]),
            "scale_factor": np.array([1.0, 1.0 / 1001.0]),
            "ionization_fraction_per_hydrogen": np.array([1.16, 0.14]),
            "opacity_GeV": np.array([2e-45, 3e-40]),
        }
        field = np.array([0.0, 0.2])

        actual = finite_h_opacity_history_with_fixed_xe(
            standard_history, field, parameters
        )
        expected = finite_h_thomson_opacity_per_conformal_time(
            standard_history["opacity_GeV"], field, parameters
        )

        np.testing.assert_array_equal(
            actual["ionization_fraction_per_hydrogen"],
            standard_history["ionization_fraction_per_hydrogen"],
        )
        np.testing.assert_allclose(
            actual["opacity_finite_h_fixed_xe_GeV"], expected
        )
        with self.assertRaises(ValueError):
            finite_h_opacity_history_with_fixed_xe({}, 0.0, parameters)


@unittest.skipUnless(
    importlib.util.find_spec("camb") is not None,
    "Install the optional CAMB dependency to run this integration test.",
)
class NeutromatherionCMBTests(unittest.TestCase):
    def test_camb_reproduces_canonical_scalar_background(self):
        import camb

        parameters, history = build_camb_parameters(lmax=300, history_points=401)
        results = camb.get_background(parameters)

        self.assertAlmostEqual(parameters.num_nu_massless, 2.046, places=12)
        self.assertAlmostEqual(parameters.nu_mass_degeneracies[0], 1.0, places=12)
        self.assertAlmostEqual(parameters.DarkEnergy.cs2, 1.0, places=12)
        self.assertLess(max_background_relative_error(results, history), 1e-5)
        recombination = standard_recombination_history_from_camb(
            results, [0.0, 1100.0, 2000.0]
        )
        self.assertEqual(
            recombination["ionization_fraction_per_hydrogen"].shape, (3,)
        )
        self.assertTrue(np.all(recombination["opacity_Mpc_inv"] >= 0.0))

        redshifts = recombination["redshift"]
        scale_factors = recombination["scale_factor"]
        hubble_GeV = (
            parameters.H0
            / 3.0856775814913673e19
            * 6.582119569e-25
        )
        critical_density_GeV4 = 3.0 * hubble_GeV**2 * (2.435e18) ** 2
        omega_b = parameters.ombh2 / (parameters.H0 / 100.0) ** 2
        baryon_density_GeV4 = (
            omega_b * critical_density_GeV4 / scale_factors**3
        )
        hydrogen_density_GeV3 = (
            (1.0 - parameters.YHe)
            * baryon_density_GeV4
            / 0.93827208816
        )
        electron_density_GeV3 = (
            recombination["ionization_fraction_per_hydrogen"]
            * hydrogen_density_GeV3
        )
        derived_opacity_GeV = standard_thomson_opacity_per_conformal_time(
            scale_factors, electron_density_GeV3
        )
        np.testing.assert_allclose(
            recombination["opacity_GeV"],
            derived_opacity_GeV,
            rtol=4e-4,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()