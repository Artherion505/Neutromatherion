"""Conditional CMB predictions for the decoupled canonical-scalar limit.

The scalar background is taken from the Pantheon+/DESI best-fit branch, with
g=h=0 for this calculation. CAMB evolves its linear perturbations as a fluid
with the exact tabulated w(a) and rest-frame sound speed squared equal to one.
This does not include the finite photon-clock interaction or its perturbations.
"""

from collections.abc import Mapping
import csv
from pathlib import Path

import numpy as np

from background_data_fit import (
    _neutrino_parameters,
    _present_densities,
    scalar_clock_history,
)
from flrw_perturbations import finite_h_thomson_opacity_per_conformal_time
from planck_tt_check import load_planck_tt


H0_KM_S_MPC = 67.4
OMEGA_M_BEST_FIT = 0.302962
SCALAR_MASS_RATIO_BEST_FIT = 0.858489
OMEGA_B_H2_REFERENCE = 0.02237
AMPLITUDE_SCALAR_REFERENCE = 2.1e-9
SCALAR_SPECTRAL_INDEX_REFERENCE = 0.9649
OPTICAL_DEPTH_REFERENCE = 0.054
N_EFF_TOTAL = 3.046
N_EFF_THERMAL_MASSIVE_STATES = 3.0
MASSIVE_NEUTRINO_EV = 0.06
MAX_BACKGROUND_RELATIVE_ERROR = 1e-5
_MPC_IN_METERS = 3.0856775814913673e22
_GEV_INVERSE_IN_METERS = 1.973269804e-16
_MPC_INVERSE_TO_GEV = _GEV_INVERSE_IN_METERS / _MPC_IN_METERS


def _camb_module():
    try:
        import camb
    except ImportError as error:
        raise RuntimeError(
            "Install the optional CAMB dependency with "
            "'python -m pip install -r requirements-cmb.txt'."
        ) from error
    return camb


def build_camb_parameters(lmax=3000, history_points=1201):
    """Build CAMB inputs and the matching exact scalar background history."""
    if lmax < 100 or history_points < 101:
        raise ValueError("Require lmax >= 100 and at least 101 history points.")

    camb = _camb_module()
    scale_factors = np.geomspace(1e-8, 1.0, history_points)
    history = scalar_clock_history(
        scale_factors,
        H0_KM_S_MPC,
        OMEGA_M_BEST_FIT,
        SCALAR_MASS_RATIO_BEST_FIT,
        g_GeV_inv=0.0,
        h_GeV_inv=0.0,
    )

    hubble_parameter = H0_KM_S_MPC / 100.0
    neutrino_parameters = _neutrino_parameters()
    present = _present_densities(H0_KM_S_MPC, neutrino_parameters)
    omega_nu_massive = (
        present["neutrinos"]["rho_massive_GeV4"]
        / present["rho_critical_GeV4"]
    )
    omega_cb = OMEGA_M_BEST_FIT - omega_nu_massive
    omega_c_h2 = omega_cb * hubble_parameter**2 - OMEGA_B_H2_REFERENCE
    if omega_c_h2 <= 0.0:
        raise ValueError("The selected baryon density exceeds the fitted matter density.")

    parameters = camb.CAMBparams()
    parameters.set_cosmology(
        H0=H0_KM_S_MPC,
        ombh2=OMEGA_B_H2_REFERENCE,
        omch2=omega_c_h2,
        mnu=MASSIVE_NEUTRINO_EV,
        nnu=N_EFF_TOTAL,
        standard_neutrino_neff=N_EFF_THERMAL_MASSIVE_STATES,
        num_massive_neutrinos=1,
        TCMB=2.7255,
        tau=OPTICAL_DEPTH_REFERENCE,
    )
    if not np.isclose(parameters.num_nu_massless, 2.046, rtol=0.0, atol=1e-12):
        raise RuntimeError("CAMB did not preserve the model's massless N_eff weight.")
    if not np.isclose(parameters.nu_mass_degeneracies[0], 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("CAMB did not preserve unit weight for the massive neutrino.")

    parameters.set_dark_energy_w_a(
        history["scale_factor"], history["w_N"], dark_energy_model="fluid"
    )
    parameters.DarkEnergy.cs2 = 1.0
    parameters.InitPower.set_params(
        As=AMPLITUDE_SCALAR_REFERENCE,
        ns=SCALAR_SPECTRAL_INDEX_REFERENCE,
    )
    parameters.WantTensors = False
    parameters.Want_CMB_lensing = True
    parameters.DoLensing = True
    parameters.set_for_lmax(
        lmax,
        lens_potential_accuracy=1,
        nonlinear=False,
    )
    return parameters, history


def max_background_relative_error(results, history):
    """Compare CAMB H(a) with the model's exact background solution."""
    redshifts = 1.0 / history["scale_factor"] - 1.0
    hubble_camb = np.asarray(results.hubble_parameter(redshifts), dtype=float)
    hubble_model = H0_KM_S_MPC * history["H_over_H0"]
    if hubble_camb.shape != hubble_model.shape or np.any(~np.isfinite(hubble_camb)):
        raise ValueError("CAMB returned an invalid H(a) history.")
    return float(np.max(np.abs(hubble_camb / hubble_model - 1.0)))


def standard_recombination_history_from_camb(camb_results, redshifts):
    """Return CAMB's standard ionization and opacity history.

    CAMB's x_e convention is free electrons per hydrogen nucleus and opacity
    is returned in Mpc^-1. The returned opacity_GeV is converted for use by
    the GeV-based perturbation helpers. This is a standard-recombination
    reference, not a finite-h recombination calculation.
    """
    redshifts = np.asarray(redshifts, dtype=float)
    if (
        redshifts.ndim != 1
        or redshifts.size == 0
        or np.any(~np.isfinite(redshifts))
        or np.any(redshifts < 0.0)
    ):
        raise ValueError("Redshifts must be a nonempty finite nonnegative vector.")

    try:
        evolution = camb_results.get_background_redshift_evolution(
            redshifts,
            vars=["x_e", "opacity"],
            format="dict",
        )
        ionization_fraction = np.asarray(evolution["x_e"], dtype=float)
        opacity_per_mpc = np.asarray(evolution["opacity"], dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "CAMB results must provide x_e and opacity background histories."
        ) from error
    if (
        ionization_fraction.shape != redshifts.shape
        or opacity_per_mpc.shape != redshifts.shape
        or np.any(~np.isfinite(ionization_fraction))
        or np.any(ionization_fraction < 0.0)
        or np.any(~np.isfinite(opacity_per_mpc))
        or np.any(opacity_per_mpc < 0.0)
    ):
        raise ValueError("CAMB returned invalid recombination history values.")

    return {
        "redshift": redshifts,
        "scale_factor": 1.0 / (1.0 + redshifts),
        "ionization_fraction_per_hydrogen": ionization_fraction,
        "opacity_Mpc_inv": opacity_per_mpc,
        "opacity_GeV": opacity_per_mpc * _MPC_INVERSE_TO_GEV,
    }


def finite_h_opacity_history_with_fixed_xe(
    standard_recombination_history,
    N_background_GeV,
    parameters,
):
    """Apply the finite-h Thomson rate factor while keeping CAMB x_e fixed.

    This is a frozen-ionization comparison path, not a self-consistent
    finite-h recombination calculation. The returned opacity is already
    modified and must not be passed through the finite-h factor again.
    """
    required_keys = (
        "redshift",
        "scale_factor",
        "ionization_fraction_per_hydrogen",
        "opacity_GeV",
    )
    if not isinstance(standard_recombination_history, Mapping):
        raise ValueError("Recombination history must be a mapping.")
    missing_keys = [
        key for key in required_keys if key not in standard_recombination_history
    ]
    if missing_keys:
        raise ValueError(
            "Recombination history is missing: " + ", ".join(missing_keys)
        )

    redshifts = np.asarray(standard_recombination_history["redshift"], dtype=float)
    scale_factors = np.asarray(
        standard_recombination_history["scale_factor"], dtype=float
    )
    ionization_fraction = np.asarray(
        standard_recombination_history["ionization_fraction_per_hydrogen"],
        dtype=float,
    )
    standard_opacity = np.asarray(
        standard_recombination_history["opacity_GeV"], dtype=float
    )
    if (
        redshifts.ndim != 1
        or redshifts.size == 0
        or scale_factors.shape != redshifts.shape
        or ionization_fraction.shape != redshifts.shape
        or standard_opacity.shape != redshifts.shape
        or np.any(~np.isfinite(redshifts))
        or np.any(redshifts < 0.0)
        or np.any(~np.isfinite(scale_factors))
        or np.any(scale_factors <= 0.0)
        or np.any(~np.isfinite(ionization_fraction))
        or np.any(ionization_fraction < 0.0)
        or np.any(~np.isfinite(standard_opacity))
        or np.any(standard_opacity < 0.0)
    ):
        raise ValueError("Recombination history values must be finite and physical.")
    try:
        field = np.broadcast_to(
            np.asarray(N_background_GeV, dtype=float), redshifts.shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The scalar-field history must match the recombination grid."
        ) from error

    effective_opacity = finite_h_thomson_opacity_per_conformal_time(
        standard_opacity, field, parameters
    )
    return {
        "redshift": redshifts,
        "scale_factor": scale_factors,
        "ionization_fraction_per_hydrogen": ionization_fraction,
        "opacity_standard_GeV": standard_opacity,
        "opacity_finite_h_fixed_xe_GeV": effective_opacity,
    }


def run_cmb_check(output_dir=None, lmax=3000, history_points=1201):
    """Calculate conditional scalar CMB spectra and CMB lensing from CAMB."""
    camb = _camb_module()
    planck_tt = load_planck_tt()
    lmax = max(lmax, int(np.ceil(np.max(planck_tt[:, 0]))) + 25)
    parameters, history = build_camb_parameters(lmax, history_points)
    results = camb.get_results(parameters)
    background_error = max_background_relative_error(results, history)
    if background_error > MAX_BACKGROUND_RELATIVE_ERROR:
        raise RuntimeError(
            "CAMB background does not reproduce the scalar solution: "
            f"max relative error={background_error:.3e}."
        )

    spectra = results.get_cmb_power_spectra(
        parameters,
        lmax=lmax,
        CMB_unit="muK",
        raw_cl=False,
    )
    lensed = spectra["lensed_scalar"]
    unlensed = spectra["unlensed_scalar"]
    lensing = results.get_lens_potential_cls(
        lmax=lmax,
        CMB_unit="muK",
        raw_cl=True,
    )
    if (
        lensed.shape[0] <= int(np.max(planck_tt[:, 0]))
        or np.any(~np.isfinite(lensed))
        or np.any(~np.isfinite(unlensed))
        or np.any(~np.isfinite(lensing))
    ):
        raise ValueError("CAMB returned incomplete or non-finite spectra.")

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "cmb_model_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ell = np.arange(lensed.shape[0])
    spectra_path = output_dir / "neutromatherion_canonical_scalar_cls.csv"
    with spectra_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "ell",
                "Dl_TT_lensed_uK2",
                "Dl_EE_lensed_uK2",
                "Dl_BB_lensed_uK2",
                "Dl_TE_lensed_uK2",
                "Dl_TT_unlensed_uK2",
            )
        )
        writer.writerows(
            zip(
                ell,
                lensed[:, 0],
                lensed[:, 1],
                lensed[:, 2],
                lensed[:, 3],
                unlensed[:, 0],
                strict=True,
            )
        )

    lensing_path = output_dir / "neutromatherion_canonical_scalar_lensing.csv"
    with lensing_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("L", "C_L_phiphi", "C_L_phiT_uK", "C_L_phiE_uK"))
        writer.writerows(
            zip(
                np.arange(lensing.shape[0]),
                lensing[:, 0],
                lensing[:, 1],
                lensing[:, 2],
                strict=True,
            )
        )

    planck_ell, planck_dl, err_low, err_high, planck_best_fit = planck_tt.T
    model_at_bins = np.interp(planck_ell, ell, lensed[:, 0])
    errors = np.where(model_at_bins >= planck_dl, err_high, err_low)
    pulls = (model_at_bins - planck_dl) / errors
    diagnostic_path = output_dir / "planck_tt_conditional_diagnostic.csv"
    with diagnostic_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "ell",
                "Dl_TT_model_uK2",
                "Dl_TT_data_uK2",
                "Dl_TT_planck_bestfit_uK2",
                "error_low_uK2",
                "error_high_uK2",
                "diagonal_pull",
            )
        )
        writer.writerows(
            zip(
                planck_ell,
                model_at_bins,
                planck_dl,
                planck_best_fit,
                err_low,
                err_high,
                pulls,
                strict=True,
            )
        )

    diagonal_chi2 = float(np.dot(pulls, pulls))
    print("Conditional Neutromatherion canonical-scalar CMB calculation")
    print("Branch: rho_Lambda=0, lambda_N=0, g=h=0; CAMB fluid cs2=1.")
    print(
        f"Fixed inputs: H0={H0_KM_S_MPC:.1f}, Omega_m={OMEGA_M_BEST_FIT:.6f}, "
        f"m_phi/H0={SCALAR_MASS_RATIO_BEST_FIT:.6f}, "
        f"omega_b h^2={OMEGA_B_H2_REFERENCE:.5f}."
    )
    print(f"Max relative background H(a) difference: {background_error:.3e}")
    print(
        f"Planck TT diagonal diagnostic: chi2/bin="
        f"{diagonal_chi2 / len(planck_ell):.6f}; not a likelihood "
        "(bandpower covariance and window functions are not included)."
    )
    print("No finite h photon-clock perturbation is included in this spectrum.")
    print(f"Saved {spectra_path}")
    print(f"Saved {lensing_path}")
    print(f"Saved {diagnostic_path}")
    return {
        "lmax": int(lmax),
        "planck_tt_bins": int(len(planck_ell)),
        "background_max_relative_error": background_error,
        "planck_tt_diagonal_chi2_per_bin": diagonal_chi2 / len(planck_ell),
        "spectra_path": spectra_path,
        "lensing_path": lensing_path,
        "diagnostic_path": diagnostic_path,
    }


if __name__ == "__main__":
    run_cmb_check()