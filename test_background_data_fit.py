import unittest

import numpy as np

from background_data_fit import (
    BackgroundLikelihood,
    lcdm_expansion,
    load_background_data,
    scalar_clock_history,
    scalar_clock_expansion,
)
from flrw_slowroll import solve_slow_roll_history


class BackgroundDataFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_background_data()
        cls.redshift_grid = np.linspace(0.0, 2.33, 301)

    def test_official_data_selection_and_covariance_shapes(self):
        self.assertEqual(len(self.data.sn_z_hd), 1590)
        self.assertEqual(self.data.sn_cholesky.shape, (1590, 1590))
        self.assertEqual(self.data.bao_values.shape, (13,))
        self.assertEqual(self.data.bao_cholesky.shape, (13, 13))
        self.assertTrue(np.all(self.data.sn_z_hd > 0.01))

    def test_lcdm_expansion_is_flatly_normalized_today(self):
        expansion = lcdm_expansion(
            self.redshift_grid, h0_km_s_mpc=67.4, omega_m=0.315
        )
        self.assertAlmostEqual(expansion[0], 1.0, places=12)
        self.assertTrue(np.all(np.diff(expansion) > 0.0))

    def test_decoupled_scalar_history_is_canonical_and_normalized(self):
        history = scalar_clock_history(
            np.geomspace(1e-8, 1.0, 101),
            h0_km_s_mpc=67.4,
            omega_m=0.302962,
            mass_ratio=0.858489,
        )

        self.assertAlmostEqual(history["H_over_H0"][-1], 1.0, places=7)
        self.assertTrue(np.all(history["rho_N_GeV4"] > 0.0))
        self.assertTrue(np.all(np.isfinite(history["w_N"])))
        self.assertGreaterEqual(float(np.min(history["w_N"])), -1.0 - 1e-12)
        self.assertLessEqual(float(np.max(history["w_N"])), 1.0 + 1e-12)

    def test_scalar_branch_reproduces_existing_shooting_history(self):
        known = solve_slow_roll_history(0.2, sample_count=1001)
        expansion, _N_initial = scalar_clock_expansion(
            self.redshift_grid,
            h0_km_s_mpc=67.4,
            omega_m=0.315,
            mass_ratio=0.2,
        )
        target_redshifts = np.asarray((0.5, 1.0, 2.0, 2.33))
        current_values = np.interp(
            target_redshifts, self.redshift_grid, expansion
        )
        old_redshifts = 1.0 / known["history"]["a"] - 1.0
        old_values = np.interp(
            target_redshifts,
            old_redshifts[::-1],
            known["history"]["H_over_H0"][::-1],
        )
        np.testing.assert_allclose(current_values, old_values, rtol=2e-4)

    def test_profiled_likelihood_uses_both_data_sets(self):
        likelihood = BackgroundLikelihood(
            self.data, h0_pivot_km_s_mpc=67.4, grid_size=301
        )
        residuals = likelihood.residual_vector("lcdm", (0.315, 30.0))
        self.assertEqual(residuals.shape, (1603,))
        self.assertTrue(np.all(np.isfinite(residuals)))

    def test_scalar_mass_profile_crosses_standard_thresholds(self):
        likelihood = BackgroundLikelihood(self.data, grid_size=301)
        fit = likelihood.fit_model("neutromatherion")
        profile = likelihood.profile_scalar_mass(fit, profile_points=5)
        interval_68 = profile["intervals"]["68.3%"]
        interval_95 = profile["intervals"]["95%"]

        self.assertTrue(profile["optimizer_success"])
        self.assertLessEqual(profile["minimum_chi2"], fit["chi2"] + 1e-9)
        self.assertLess(interval_95[0], interval_68[0])
        self.assertGreater(interval_68[1], 1.0)
        self.assertGreater(interval_95[1], interval_68[1])


if __name__ == "__main__":
    unittest.main()