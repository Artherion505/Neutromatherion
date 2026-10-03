from __future__ import annotations

import csv
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from scipy.optimize import least_squares


DATA_ARCHIVE = Path(__file__).resolve().parent / "data" / "Rotmod_LTG.zip"
G_KPC_KMS2_PER_MSUN = 4.30091e-6
C_KMS = 299792.458
G_REFERENCE_KMS2_PER_KPC = 1.0
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7


@dataclass(frozen=True)
class RotationCurve:
    name: str
    radius_kpc: np.ndarray
    velocity_observed: np.ndarray
    velocity_error: np.ndarray
    velocity_gas: np.ndarray
    velocity_disk: np.ndarray
    velocity_bulge: np.ndarray
    velocity_baryon_squared: np.ndarray


def load_sparc_sample(archive_path: Path = DATA_ARCHIVE) -> list[RotationCurve]:
    """Load the public SPARC Rotmod_LTG archive and calculate Vbar squared."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"SPARC archive not found: {archive_path}")

    curves = []
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.lower().endswith("_rotmod.dat")
        )
        for member in members:
            with archive.open(member) as stream:
                values = np.loadtxt(stream, comments="#")
            if values.ndim != 2 or values.shape[1] != 8:
                raise ValueError(f"Unexpected SPARC table format: {member}")

            radius, observed, error, gas, disk, bulge = values[:, :6].T
            valid = (
                np.isfinite(radius)
                & np.isfinite(observed)
                & np.isfinite(error)
                & (radius > 0)
                & (error > 0)
            )
            radius, observed, error = radius[valid], observed[valid], error[valid]
            gas, disk, bulge = gas[valid], disk[valid], bulge[valid]
            order = np.argsort(radius)
            radius, observed, error = radius[order], observed[order], error[order]
            gas, disk, bulge = gas[order], disk[order], bulge[order]

            baryon_squared = (
                gas * np.abs(gas)
                + UPSILON_DISK * disk**2
                + UPSILON_BULGE * bulge**2
            )
            baryon_squared = np.maximum(baryon_squared, 0.0)
            curves.append(
                RotationCurve(
                    name=Path(member).name.removesuffix("_rotmod.dat"),
                    radius_kpc=radius,
                    velocity_observed=observed,
                    velocity_error=error,
                    velocity_gas=gas,
                    velocity_disk=disk,
                    velocity_bulge=bulge,
                    velocity_baryon_squared=baryon_squared,
                )
            )

    if not curves:
        raise ValueError(f"No Rotmod_LTG tables found in {archive_path}")
    return curves


def model_velocity(radius_kpc, velocity_baryon_squared, alpha, mu_per_kpc):
    """Baryons plus the current Yukawa-like spherical-equivalent extension."""
    correction = (
        alpha
        * velocity_baryon_squared
        * (1.0 + mu_per_kpc * radius_kpc)
        * np.exp(-mu_per_kpc * radius_kpc)
    )
    return np.sqrt(np.maximum(velocity_baryon_squared + correction, 0.0))


def monotonic_response_velocity(
    radius_kpc, velocity_baryon_squared, amplitude, exponent
):
    """Exploratory positive response increasing with local baryonic acceleration.

    This is a falsifiable phenomenological proxy for the verbal Reactio rule,
    not a derivation from the scalar-field dynamics or a total-energy density.
    """
    baryonic_acceleration = velocity_baryon_squared / radius_kpc
    extra_acceleration = (
        amplitude
        * G_REFERENCE_KMS2_PER_KPC
        * (baryonic_acceleration / G_REFERENCE_KMS2_PER_KPC) ** exponent
    )
    return np.sqrt(
        np.maximum(
            (baryonic_acceleration + extra_acceleration) * radius_kpc,
            0.0,
        )
    )


def fit_monotonic_response(curves: list[RotationCurve]) -> dict[str, float | int | bool]:
    """Fit a shared monotonic source-response proxy using reported SPARC errV."""
    radius = np.concatenate([curve.radius_kpc for curve in curves])
    observed = np.concatenate([curve.velocity_observed for curve in curves])
    errors = np.concatenate([curve.velocity_error for curve in curves])
    baryon_squared = np.concatenate(
        [curve.velocity_baryon_squared for curve in curves]
    )

    def residuals(parameters):
        amplitude = 10.0**parameters[0]
        exponent = parameters[1]
        predicted = monotonic_response_velocity(
            radius, baryon_squared, amplitude, exponent
        )
        return (predicted - observed) / errors

    fit = least_squares(
        residuals,
        x0=np.array([1.8, 0.4]),
        bounds=(np.array([-12.0, 0.05]), np.array([8.0, 2.5])),
        method="trf",
        x_scale="jac",
        max_nfev=5000,
    )
    dof = int(observed.size - fit.x.size)
    return {
        "amplitude": float(10.0**fit.x[0]),
        "exponent": float(fit.x[1]),
        "g_reference_kms2_per_kpc": G_REFERENCE_KMS2_PER_KPC,
        "n_galaxies": len(curves),
        "n_points": int(observed.size),
        "dof": dof,
        "chi2": float(np.sum(fit.fun**2)),
        "chi2_per_dof": float(np.sum(fit.fun**2) / dof),
        "optimizer_success": bool(fit.success),
        "parameter_at_bound": bool(
            np.isclose(fit.x[0], -12.0, atol=1e-4)
            or np.isclose(fit.x[0], 8.0, atol=1e-4)
            or np.isclose(fit.x[1], 0.05, atol=1e-4)
            or np.isclose(fit.x[1], 2.5, atol=1e-4)
        ),
    }


def cross_validate_models(
    curves: list[RotationCurve], n_splits: int = 5, seed: int = 20261001
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """Hold out whole galaxies and compare three shared-parameter models."""
    if n_splits < 2 or n_splits > len(curves):
        raise ValueError("n_splits must be between 2 and the number of galaxies")

    order = np.random.default_rng(seed).permutation(len(curves))
    folds = np.array_split(order, n_splits)
    rows = []
    totals = {
        "n_points": 0,
        "chi2_baryon": 0.0,
        "chi2_yukawa": 0.0,
        "chi2_monotonic": 0.0,
    }

    for fold_index, test_indices in enumerate(folds, start=1):
        test_index_set = set(test_indices.tolist())
        train_curves = [
            curve for index, curve in enumerate(curves)
            if index not in test_index_set
        ]
        test_curves = [curves[index] for index in test_indices]
        yukawa_fit = fit_global_model(train_curves)
        monotonic_fit = fit_monotonic_response(train_curves)

        point_count = 0
        chi2_values = {key: 0.0 for key in totals if key != "n_points"}
        for curve in test_curves:
            radius = curve.radius_kpc
            observed = curve.velocity_observed
            errors = curve.velocity_error
            baryon_squared = curve.velocity_baryon_squared
            baryon_prediction = np.sqrt(baryon_squared)
            yukawa_prediction = model_velocity(
                radius,
                baryon_squared,
                yukawa_fit["alpha"],
                yukawa_fit["mu_per_kpc"],
            )
            monotonic_prediction = monotonic_response_velocity(
                radius,
                baryon_squared,
                monotonic_fit["amplitude"],
                monotonic_fit["exponent"],
            )
            predictions = {
                "chi2_baryon": baryon_prediction,
                "chi2_yukawa": yukawa_prediction,
                "chi2_monotonic": monotonic_prediction,
            }
            for key, prediction in predictions.items():
                chi2_values[key] += float(np.sum(((prediction - observed) / errors) ** 2))
            point_count += observed.size

        row = {
            "fold": fold_index,
            "train_galaxies": len(train_curves),
            "test_galaxies": len(test_curves),
            "test_points": point_count,
            "alpha_yukawa": yukawa_fit["alpha"],
            "mu_yukawa_per_kpc": yukawa_fit["mu_per_kpc"],
            "amplitude_monotonic": monotonic_fit["amplitude"],
            "exponent_monotonic": monotonic_fit["exponent"],
            "chi2_per_point_baryon": chi2_values["chi2_baryon"] / point_count,
            "chi2_per_point_yukawa": chi2_values["chi2_yukawa"] / point_count,
            "chi2_per_point_monotonic": chi2_values["chi2_monotonic"] / point_count,
        }
        rows.append(row)
        totals["n_points"] += point_count
        for key, value in chi2_values.items():
            totals[key] += value

    summary = {
        "n_splits": n_splits,
        "seed": seed,
        "n_points": totals["n_points"],
        "chi2_per_point_baryon": totals["chi2_baryon"] / totals["n_points"],
        "chi2_per_point_yukawa": totals["chi2_yukawa"] / totals["n_points"],
        "chi2_per_point_monotonic": totals["chi2_monotonic"] / totals["n_points"],
    }
    return rows, summary


def fit_global_model(curves: list[RotationCurve]) -> dict[str, float | int | bool]:
    """Fit common positive alpha and mu to all SPARC velocities using errV."""
    radius = np.concatenate([curve.radius_kpc for curve in curves])
    observed = np.concatenate([curve.velocity_observed for curve in curves])
    errors = np.concatenate([curve.velocity_error for curve in curves])
    baryon_squared = np.concatenate(
        [curve.velocity_baryon_squared for curve in curves]
    )

    def residuals(parameters):
        alpha = 10.0**parameters[0]
        mu = parameters[1]
        predicted = model_velocity(radius, baryon_squared, alpha, mu)
        return (predicted - observed) / errors

    fit = least_squares(
        residuals,
        x0=np.array([-2.0, 0.01]),
        bounds=(np.array([-8.0, 0.0]), np.array([2.0, 10.0])),
        method="trf",
        x_scale="jac",
        max_nfev=5000,
    )
    alpha = 10.0**fit.x[0]
    mu = fit.x[1]
    baryon_velocity = np.sqrt(baryon_squared)
    baseline_chi2 = float(np.sum(((baryon_velocity - observed) / errors) ** 2))
    fitted_chi2 = float(np.sum(fit.fun**2))
    dof = int(observed.size - fit.x.size)
    at_bound = bool(
        np.isclose(fit.x[0], -8.0, atol=1e-4)
        or np.isclose(fit.x[0], 2.0, atol=1e-4)
        or np.isclose(fit.x[1], 0.0, atol=1e-6)
        or np.isclose(fit.x[1], 10.0, atol=1e-4)
    )
    return {
        "alpha": float(alpha),
        "mu_per_kpc": float(mu),
        "n_galaxies": len(curves),
        "n_points": int(observed.size),
        "dof": dof,
        "chi2_baryon": baseline_chi2,
        "chi2_baryon_per_dof": baseline_chi2 / observed.size,
        "chi2_fit": fitted_chi2,
        "chi2_fit_per_dof": fitted_chi2 / dof,
        "optimizer_success": bool(fit.success),
        "parameter_at_bound": at_bound,
    }


def save_global_fit(
    curves: list[RotationCurve],
    result: dict,
    output_dir: Path,
    monotonic_result: dict | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = result["alpha"]
    mu = result["mu_per_kpc"]
    rows = []
    galaxy_rows = []

    for curve in curves:
        baryon_velocity = np.sqrt(curve.velocity_baryon_squared)
        predicted = model_velocity(
            curve.radius_kpc, curve.velocity_baryon_squared, alpha, mu
        )
        pull = (predicted - curve.velocity_observed) / curve.velocity_error
        baryon_pull = (baryon_velocity - curve.velocity_observed) / curve.velocity_error
        monotonic_predicted = None
        monotonic_pull = None
        if monotonic_result is not None:
            monotonic_predicted = monotonic_response_velocity(
                curve.radius_kpc,
                curve.velocity_baryon_squared,
                monotonic_result["amplitude"],
                monotonic_result["exponent"],
            )
            monotonic_pull = (
                monotonic_predicted - curve.velocity_observed
            ) / curve.velocity_error
        galaxy_rows.append(
            {
                "galaxy": curve.name,
                "n_points": curve.radius_kpc.size,
                "chi2_baryon": float(np.sum(baryon_pull**2)),
                "chi2_fit": float(np.sum(pull**2)),
                "chi2_monotonic": (
                    float(np.sum(monotonic_pull**2))
                    if monotonic_pull is not None
                    else ""
                ),
            }
        )
        for index in range(curve.radius_kpc.size):
            row = {
                "galaxy": curve.name,
                "radius_kpc": curve.radius_kpc[index],
                "velocity_observed_kms": curve.velocity_observed[index],
                "velocity_error_kms": curve.velocity_error[index],
                "velocity_baryon_kms": baryon_velocity[index],
                "velocity_yukawa_kms": predicted[index],
                "normalized_residual_yukawa": pull[index],
            }
            if monotonic_predicted is not None:
                row["velocity_monotonic_kms"] = monotonic_predicted[index]
                row["normalized_residual_monotonic"] = monotonic_pull[index]
            rows.append(row)

    with (output_dir / "sparc_global_fit_residuals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "sparc_global_fit_by_galaxy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=galaxy_rows[0].keys())
        writer.writeheader()
        writer.writerows(galaxy_rows)


def plot_ngc3198(
    curves: list[RotationCurve],
    result: dict,
    output_path: Path,
    monotonic_result: dict | None = None,
):
    curve = next((item for item in curves if item.name.upper() == "NGC3198"), None)
    if curve is None:
        return

    baryon_velocity = np.sqrt(curve.velocity_baryon_squared)
    fitted_velocity = model_velocity(
        curve.radius_kpc,
        curve.velocity_baryon_squared,
        result["alpha"],
        result["mu_per_kpc"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, height_ratios=(3, 1))
    axes[0].errorbar(
        curve.radius_kpc,
        curve.velocity_observed,
        yerr=curve.velocity_error,
        fmt="o",
        markersize=3,
        color="#222222",
        alpha=0.75,
        label="SPARC observations",
    )
    axes[0].plot(curve.radius_kpc, baryon_velocity, "--", label="Baryons")
    axes[0].plot(curve.radius_kpc, fitted_velocity, label="Global fit applied to NGC 3198")
    if monotonic_result is not None:
        monotonic_velocity = monotonic_response_velocity(
            curve.radius_kpc,
            curve.velocity_baryon_squared,
            monotonic_result["amplitude"],
            monotonic_result["exponent"],
        )
        axes[0].plot(
            curve.radius_kpc,
            monotonic_velocity,
            label="Monotonic source-response fit",
        )
    axes[0].set_ylabel("Circular velocity [km/s]")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].errorbar(
        curve.radius_kpc,
        (fitted_velocity - curve.velocity_observed) / curve.velocity_error,
        fmt="o",
        markersize=3,
        color="#b23a48",
        label="Yukawa residual",
    )
    if monotonic_result is not None:
        axes[1].plot(
            curve.radius_kpc,
            (monotonic_velocity - curve.velocity_observed) / curve.velocity_error,
            ".",
            color="#2a7f62",
            markersize=3,
            label="Monotonic-response residual",
        )
    axes[1].set_xlabel("Radius [kpc]")
    axes[1].set_ylabel("Residual / errV")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("SPARC NGC 3198 with shared 175-galaxy fit parameters")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def run_global_fit(output_dir: Path | None = None) -> tuple[list[RotationCurve], dict]:
    output_dir = output_dir or Path(__file__).resolve().parent
    curves = load_sparc_sample()
    result = fit_global_model(curves)
    save_global_fit(curves, result, output_dir)
    plot_ngc3198(curves, result, output_dir / "curva_rotacion_NGC3198.png")
    return curves, result


def spherical_equivalent_deflection(curve: RotationCurve, velocity_squared):
    """Compute a finite-range spherical-equivalent deflection estimate.

    SPARC rotation curves are measured in the disk plane, not as lensing data.
    This projection is only a diagnostic and is truncated at the last measured
    radius; it must not be interpreted as an observed lensing profile.
    """
    from scipy.integrate import quad

    radius = curve.radius_kpc
    velocity_squared = np.asarray(velocity_squared)
    interpolator = PchipInterpolator(radius, velocity_squared, extrapolate=False)
    impact_parameters = np.linspace(max(radius[0], 0.5), 0.95 * radius[-1], 50)
    deflection_arcsec = []

    for impact in impact_parameters:
        z_max = math.sqrt(radius[-1] ** 2 - impact**2)

        def integrand(z):
            spherical_radius = math.sqrt(impact**2 + z**2)
            circular_speed_squared = float(interpolator(spherical_radius))
            acceleration = circular_speed_squared / spherical_radius
            return acceleration * impact / spherical_radius

        integral, _ = quad(integrand, 0.0, z_max, epsabs=1e-7, epsrel=1e-6, limit=300)
        angle_radians = 4.0 * integral / C_KMS**2
        deflection_arcsec.append(angle_radians * 206265.0)

    return impact_parameters, np.asarray(deflection_arcsec)


def run_lensing_estimate():
    output_dir = Path(__file__).resolve().parent
    curves, result = run_global_fit(output_dir)
    curve = next(item for item in curves if item.name.upper() == "NGC3198")
    model_squared = model_velocity(
        curve.radius_kpc,
        curve.velocity_baryon_squared,
        result["alpha"],
        result["mu_per_kpc"],
    ) ** 2
    profiles = {
        "baryonic spherical-equivalent": curve.velocity_baryon_squared,
        "observed-dynamics spherical-equivalent": curve.velocity_observed**2,
        "global-fit spherical-equivalent": model_squared,
    }

    with (output_dir / "sparc_lensing_estimate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["impact_parameter_kpc", *profiles.keys()])
        estimates = [spherical_equivalent_deflection(curve, values) for values in profiles.values()]
        for index, impact in enumerate(estimates[0][0]):
            writer.writerow([impact, *(values[1][index] for values in estimates)])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (label, _), (_, values) in zip(profiles.items(), estimates):
        ax.plot(estimates[0][0], values, label=label)
    ax.set_title("Spherical-equivalent estimate from SPARC NGC 3198")
    ax.set_xlabel("Impact parameter [kpc]")
    ax.set_ylabel("Truncated deflection estimate [arcsec]")
    ax.text(
        0.02,
        0.03,
        "Not lensing observations; assumes spherical symmetry and truncates at the last SPARC radius.",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sparc_lensing_estimate.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent
    curves = load_sparc_sample()
    result = fit_global_model(curves)
    monotonic_result = fit_monotonic_response(curves)
    save_global_fit(curves, result, output_dir, monotonic_result)
    plot_ngc3198(
        curves,
        result,
        output_dir / "curva_rotacion_NGC3198.png",
        monotonic_result,
    )
    cv_rows, cv_summary = cross_validate_models(curves)
    with (output_dir / "sparc_model_crossvalidation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=cv_rows[0].keys())
        writer.writeheader()
        writer.writerows(cv_rows)

    print(f"SPARC galaxies: {result['n_galaxies']}")
    print(f"Velocity points: {result['n_points']}; dof: {result['dof']}")
    print(
        "Yukawa-like: "
        f"alpha={result['alpha']:.6g}, mu={result['mu_per_kpc']:.6g} kpc^-1, "
        f"chi2/dof={result['chi2_fit_per_dof']:.3f}, "
        f"at-bound={result['parameter_at_bound']}"
    )
    print(
        "Monotonic-response proxy: "
        f"amplitude={monotonic_result['amplitude']:.6g}, "
        f"exponent={monotonic_result['exponent']:.6g}, "
        f"chi2/dof={monotonic_result['chi2_per_dof']:.3f}"
    )
    print(
        "Held-out by galaxy chi2/N: "
        f"baryons={cv_summary['chi2_per_point_baryon']:.3f}, "
        f"Yukawa={cv_summary['chi2_per_point_yukawa']:.3f}, "
        f"monotonic={cv_summary['chi2_per_point_monotonic']:.3f}"
    )
    print("The monotonic model is an empirical proxy, not a derived field equation.")
    print("Saved global residuals, per-galaxy chi-square, cross-validation, and plot.")


if __name__ == "__main__":
    main()
