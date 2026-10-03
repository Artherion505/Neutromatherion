# scan_thinshell.py
import argparse
import csv
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq

# --- conversions and constants (same as corrected script) ---
c = 299792458.0
J_per_GeV = 1.602176634e-10
GeV_per_J = 1.0 / J_per_GeV
m_to_GeVinv = 5.0677307e15
Mpl_GeV = 2.4353234593382e18
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.9634e8
Phi_sun = G * M_sun / (R_sun * c**2)

MAX_DELTA_R_OVER_R = 1e-6
MAX_EPSILON_GAMMA_PROXY = 2e-5
SCAN_COLUMNS = [
    "g_GeVinv",
    "kappa_over_Lambda_GeVinv",
    "effective_dust_coupling_GeVinv",
    "lambda",
    "m0_eV",
    "N_in_GeV",
    "N_out_GeV",
    "mN_in_GeV",
    "mN_out_GeV",
    "mN_inR_sun",
    "DeltaR_over_R",
    "epsilon_gamma_proxy",
]
CYCLE_COLUMNS = [
    "cycle_index",
    "point_index",
    "random_seed",
    "grid_signature",
    *SCAN_COLUMNS,
]

def rho_kgm3_to_GeV4(rho_kgm3):
    eps_Jm3 = rho_kgm3 * c**2
    eps_GeV_per_m3 = eps_Jm3 * GeV_per_J
    return eps_GeV_per_m3 / (m_to_GeVinv**3)

rho_in = 1.0 * 1000.0      # kg/m^3
rho_out = 1e-24 * 1000.0  # kg/m^3
rho_in_GeV4 = rho_kgm3_to_GeV4(rho_in)
rho_out_GeV4 = rho_kgm3_to_GeV4(rho_out)

def find_N0(rho_GeV4, g, lam, m0, kappa_over_lambda=0.0):
    values = (rho_GeV4, g, lam, m0, kappa_over_lambda)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Los parámetros deben ser finitos.")
    if rho_GeV4 < 0 or lam < 0 or m0 < 0:
        raise ValueError("rho, lambda y m0 deben ser no negativos.")

    effective_coupling = g - kappa_over_lambda
    source = effective_coupling * rho_GeV4
    if source == 0:
        return 0.0

    source_magnitude = abs(source)
    root_scales = []
    if lam > 0:
        root_scales.append((source_magnitude / lam) ** (1.0 / 3.0))
    if m0 > 0:
        root_scales.append(source_magnitude / m0**2)
    if not root_scales:
        raise ValueError("Se necesita lambda o m0 distinto de cero para un minimo finito.")

    def f(magnitude):
        return m0**2 * magnitude + lam * magnitude**3 - source_magnitude

    upper = min(root_scales)
    while f(upper) < 0:
        upper *= 2.0

    magnitude = brentq(
        f,
        0.0,
        upper,
        xtol=np.nextafter(0.0, 1.0),
        rtol=4.0 * np.finfo(float).eps,
        maxiter=200,
    )
    return -np.copysign(magnitude, source)


def screening_diagnostics(n_in, n_out, effective_coupling):
    if not np.isfinite(effective_coupling):
        raise ValueError("El acoplamiento efectivo debe ser finito.")
    if effective_coupling == 0:
        return float("nan"), 0.0

    delta_r_over_r = abs(n_out - n_in) / (
        6.0 * abs(effective_coupling) * Mpl_GeV**2 * Phi_sun
    )
    beta = effective_coupling * Mpl_GeV
    beta_eff = beta * min(1.0, 3.0 * delta_r_over_r)
    epsilon_gamma_proxy = 2.0 * beta_eff**2 / (1.0 + beta_eff**2)
    return delta_r_over_r, epsilon_gamma_proxy


def massless_scalar_tensor_gamma(beta_source, beta_probe):
    """Return gamma for a massless, canonical, conformally coupled scalar.

    The dimensionless couplings are body-specific and normalized as
    beta_i = M_Pl * d(ln A_i)/dN. Finite range and explicit clock/photon
    couplings are outside this relation.
    """
    couplings = (beta_source, beta_probe)
    if any(
        np.ndim(value) != 0 or not np.isfinite(value)
        for value in couplings
    ):
        raise ValueError("Scalar-tensor couplings must be finite scalars.")

    coupling_product = 2.0 * float(beta_source) * float(beta_probe)
    if coupling_product <= -1.0:
        raise ValueError("The scalar-tensor coupling product must exceed -1.")
    return (1.0 - coupling_product) / (1.0 + coupling_product)


def _default_parameter_grid():
    g_values = np.unique(np.append(np.logspace(-30, -15, 8), 1e-20))
    lambda_values = np.logspace(-75, -10, 66)
    m0_values_eV = np.logspace(-30, -10, 9)
    return g_values, lambda_values, m0_values_eV


def _evaluate_parameter_point(g, lam, m0_eV, kappa_over_lambda):
    m0 = m0_eV * 1e-9
    N_in = find_N0(rho_in_GeV4, g, lam, m0, kappa_over_lambda)
    N_out = find_N0(rho_out_GeV4, g, lam, m0, kappa_over_lambda)
    mN_in = max(0.0, m0**2 + 3.0 * lam * N_in**2) ** 0.5
    mN_out = max(0.0, m0**2 + 3.0 * lam * N_out**2) ** 0.5
    mN_in_R_sun = mN_in * R_sun * m_to_GeVinv
    effective_coupling = g - kappa_over_lambda
    delta_r_over_r, epsilon_gamma_proxy = screening_diagnostics(
        N_in, N_out, effective_coupling
    )
    return [
        g,
        kappa_over_lambda,
        effective_coupling,
        lam,
        m0_eV,
        N_in,
        N_out,
        mN_in,
        mN_out,
        mN_in_R_sun,
        delta_r_over_r,
        epsilon_gamma_proxy,
    ]


def _is_screening_candidate(
    row,
    max_delta_r_over_r=MAX_DELTA_R_OVER_R,
    max_epsilon_gamma_proxy=MAX_EPSILON_GAMMA_PROXY,
):
    return (
        row[10] < max_delta_r_over_r
        and row[11] < max_epsilon_gamma_proxy
    )


def _validate_screening_thresholds(
    max_delta_r_over_r, max_epsilon_gamma_proxy
):
    thresholds = (max_delta_r_over_r, max_epsilon_gamma_proxy)
    if any(
        np.ndim(value) != 0 or not np.isfinite(value) or value < 0.0
        for value in thresholds
    ):
        raise ValueError("Los umbrales deben ser escalares finitos y no negativos.")


def run_scan(
    output_path=None,
    kappa_over_lambda=0.0,
    max_delta_r_over_r=MAX_DELTA_R_OVER_R,
    max_epsilon_gamma_proxy=MAX_EPSILON_GAMMA_PROXY,
):
    if not np.isfinite(kappa_over_lambda):
        raise ValueError("kappa/Lambda debe ser finito.")
    _validate_screening_thresholds(
        max_delta_r_over_r, max_epsilon_gamma_proxy
    )

    g_values, lambda_values, m0_values_eV = _default_parameter_grid()

    out_rows = []
    for g in g_values:
        for lam in lambda_values:
            for m0_eV in m0_values_eV:
                out_rows.append(
                    _evaluate_parameter_point(
                        g, lam, m0_eV, kappa_over_lambda
                    )
                )

    output_path = Path(output_path) if output_path else Path(__file__).with_name(
        "thinshell_scan.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(SCAN_COLUMNS)
        writer.writerows(out_rows)

    screened = [
        row for row in out_rows
        if _is_screening_candidate(
            row, max_delta_r_over_r, max_epsilon_gamma_proxy
        )
    ]
    print(
        f"Using kappa/Lambda={kappa_over_lambda:.6g} GeV^-1 "
        "in the dust-aligned approximation."
    )
    print(f"Scanned {len(out_rows)} parameter combinations.")
    print(
        "Thin-shell proxy candidates "
        f"(DeltaR/R < {max_delta_r_over_r:.3g} and "
        f"proxy < {max_epsilon_gamma_proxy:.3g}): "
        f"{len(screened)}"
    )
    print(f"Results saved to {output_path}")
    return out_rows


def search_first_candidate(
    output_path=None,
    kappa_over_lambda=0.0,
    max_delta_r_over_r=MAX_DELTA_R_OVER_R,
    max_epsilon_gamma_proxy=MAX_EPSILON_GAMMA_PROXY,
    g_values=None,
    lambda_values=None,
    m0_values_eV=None,
    resume=True,
    progress_interval=100,
):
    if not np.isfinite(kappa_over_lambda):
        raise ValueError("kappa/Lambda debe ser finito.")
    _validate_screening_thresholds(
        max_delta_r_over_r, max_epsilon_gamma_proxy
    )
    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, (int, np.integer))
        or progress_interval <= 0
    ):
        raise ValueError("progress_interval debe ser un entero positivo.")

    default_g_values, default_lambda_values, default_m0_values_eV = (
        _default_parameter_grid()
    )
    grids = (
        ("g", default_g_values if g_values is None else g_values, False),
        (
            "lambda",
            default_lambda_values if lambda_values is None else lambda_values,
            True,
        ),
        (
            "m0_eV",
            default_m0_values_eV if m0_values_eV is None else m0_values_eV,
            True,
        ),
    )
    validated_grids = []
    for name, values, nonnegative in grids:
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 1
            or values.size == 0
            or np.any(~np.isfinite(values))
            or (nonnegative and np.any(values < 0.0))
        ):
            raise ValueError(f"La grilla {name} debe contener valores físicos.")
        validated_grids.append(values)
    g_values, lambda_values, m0_values_eV = validated_grids

    output_path = (
        Path(output_path)
        if output_path
        else Path(__file__).with_name("thinshell_search.csv")
    )
    requested_points = {
        (float(g), float(kappa_over_lambda), float(lam), float(m0_eV))
        for g in g_values
        for lam in lambda_values
        for m0_eV in m0_values_eV
    }
    processed_points = set()
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as saved:
            reader = csv.DictReader(saved)
            if reader.fieldnames != SCAN_COLUMNS:
                raise ValueError(
                    f"El CSV existente no tiene el formato esperado: {output_path}"
                )
            for record in reader:
                row = [float(record[column]) for column in SCAN_COLUMNS]
                key = (row[0], row[1], row[3], row[4])
                processed_points.add(key)
                if (
                    key in requested_points
                    and _is_screening_candidate(
                        row, max_delta_r_over_r, max_epsilon_gamma_proxy
                    )
                ):
                    print(f"Candidato ya guardado en {output_path}.")
                    return row

    new_file = (
        not resume
        or not output_path.exists()
        or output_path.stat().st_size == 0
    )
    mode = "a" if resume else "w"
    scanned = 0
    with output_path.open(mode, newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if new_file:
            writer.writerow(SCAN_COLUMNS)
            output.flush()
        for g in g_values:
            for lam in lambda_values:
                for m0_eV in m0_values_eV:
                    key = (
                        float(g),
                        float(kappa_over_lambda),
                        float(lam),
                        float(m0_eV),
                    )
                    if key in processed_points:
                        continue
                    row = _evaluate_parameter_point(
                        g, lam, m0_eV, kappa_over_lambda
                    )
                    writer.writerow(row)
                    output.flush()
                    processed_points.add(key)
                    scanned += 1
                    if scanned % progress_interval == 0:
                        print(f"Puntos nuevos evaluados: {scanned}.")
                    if _is_screening_candidate(
                        row, max_delta_r_over_r, max_epsilon_gamma_proxy
                    ):
                        print(
                            "Primer candidato del proxy: "
                            f"g={row[0]:.8g} GeV^-1, "
                            f"kappa/Lambda={row[1]:.8g} GeV^-1, "
                            f"lambda={row[3]:.8g}, m0={row[4]:.8g} eV, "
                            f"DeltaR/R={row[10]:.4g}, "
                            f"epsilon_gamma_proxy={row[11]:.4g}."
                        )
                        print(f"Progreso guardado en {output_path}.")
                        return row

    print(
        "No apareció ningún candidato en la grilla solicitada; "
        f"se evaluaron {scanned} puntos nuevos. Resultados: {output_path}"
    )
    return None


def search_parameter_cycles(
    output_path=None,
    kappa_over_lambda=0.0,
    max_delta_r_over_r=MAX_DELTA_R_OVER_R,
    max_epsilon_gamma_proxy=MAX_EPSILON_GAMMA_PROXY,
    g_values=None,
    lambda_values=None,
    m0_values_eV=None,
    cycles=None,
    batch_size=25,
    random_seed=20261002,
    cycle_pause_seconds=5.0,
    resume=True,
):
    if not np.isfinite(kappa_over_lambda):
        raise ValueError("kappa/Lambda debe ser finito.")
    _validate_screening_thresholds(
        max_delta_r_over_r, max_epsilon_gamma_proxy
    )
    if cycles is not None and (
        isinstance(cycles, bool)
        or not isinstance(cycles, (int, np.integer))
        or cycles <= 0
    ):
        raise ValueError("cycles debe ser None o un entero positivo.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size debe ser un entero positivo.")
    if isinstance(random_seed, bool) or not isinstance(
        random_seed, (int, np.integer)
    ):
        raise ValueError("random_seed debe ser un entero.")
    if (
        np.ndim(cycle_pause_seconds) != 0
        or not np.isfinite(cycle_pause_seconds)
        or cycle_pause_seconds < 0.0
    ):
        raise ValueError("cycle_pause_seconds debe ser finito y no negativo.")

    default_g_values, default_lambda_values, default_m0_values_eV = (
        _default_parameter_grid()
    )
    grids = (
        ("g", default_g_values if g_values is None else g_values),
        (
            "lambda",
            default_lambda_values if lambda_values is None else lambda_values,
        ),
        (
            "m0_eV",
            default_m0_values_eV if m0_values_eV is None else m0_values_eV,
        ),
    )
    validated_grids = []
    for name, values in grids:
        values = np.asarray(values, dtype=float)
        if (
            values.ndim != 1
            or values.size == 0
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                f"La grilla {name} debe contener valores positivos y finitos."
            )
        validated_grids.append(np.unique(values))
    g_values, lambda_values, m0_values_eV = validated_grids
    log_bounds = tuple(
        (np.log10(values.min()), np.log10(values.max()))
        for values in (g_values, lambda_values, m0_values_eV)
    )
    grid_signature = hashlib.sha256(
        repr(
            tuple(
                tuple(float(value) for value in values)
                for values in (g_values, lambda_values, m0_values_eV)
            )
        ).encode("ascii")
    ).hexdigest()
    base_point_count = (
        g_values.size * lambda_values.size * m0_values_eV.size
    )

    output_path = (
        Path(output_path)
        if output_path
        else Path(__file__).with_name("thinshell_cycles.csv")
    )
    last_record = None
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as saved:
            reader = csv.DictReader(saved)
            if reader.fieldnames != CYCLE_COLUMNS:
                raise ValueError(
                    f"El CSV existente no tiene el formato esperado: {output_path}"
                )
            for record in reader:
                last_record = record

    cycle_index = 0
    start_point_index = 0
    if last_record is not None:
        if last_record["grid_signature"] != grid_signature:
            raise ValueError(
                "La grilla cambió desde el último guardado; usa otro CSV "
                "o reinicia con --no-resume."
            )
        if int(last_record["random_seed"]) != int(random_seed):
            raise ValueError(
                "La semilla cambió desde el último guardado; usa otro CSV "
                "o reinicia con --no-resume."
            )
        if float(last_record["kappa_over_Lambda_GeVinv"]) != float(
            kappa_over_lambda
        ):
            raise ValueError(
                "kappa/Lambda cambió desde el último guardado; usa otro CSV "
                "o reinicia con --no-resume."
            )
        last_cycle = int(last_record["cycle_index"])
        last_point = int(last_record["point_index"])
        if last_cycle < 0 or last_point < 0:
            raise ValueError("El checkpoint cíclico contiene índices inválidos.")
        cycle_length = base_point_count if last_cycle == 0 else batch_size
        if last_point + 1 < cycle_length:
            cycle_index = last_cycle
            start_point_index = last_point + 1
        else:
            cycle_index = last_cycle + 1
    end_cycle_index = None if cycles is None else cycle_index + cycles

    new_file = (
        not resume
        or not output_path.exists()
        or output_path.stat().st_size == 0
    )
    total_scanned = 0
    total_candidates = 0
    completed_cycles = 0
    mode = "a" if resume else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if new_file:
            writer.writerow(CYCLE_COLUMNS)
            output.flush()

        while end_cycle_index is None or cycle_index < end_cycle_index:
            if cycle_index == 0:
                parameter_points = enumerate(
                    (g, lam, m0_eV)
                    for g in g_values
                    for lam in lambda_values
                    for m0_eV in m0_values_eV
                )
                cycle_description = "grilla base"
            else:
                rng = np.random.default_rng(int(random_seed) + cycle_index)
                sampled_values = tuple(
                    10.0 ** rng.uniform(lower, upper, size=batch_size)
                    for lower, upper in log_bounds
                )
                parameter_points = enumerate(zip(*sampled_values))
                cycle_description = f"lote log-uniforme de {batch_size} puntos"

            cycle_scanned = 0
            cycle_candidates = 0
            first_candidate = None
            for point_index, (g, lam, m0_eV) in parameter_points:
                if point_index < start_point_index:
                    continue
                row = _evaluate_parameter_point(
                    g, lam, m0_eV, kappa_over_lambda
                )
                writer.writerow(
                    [
                        cycle_index,
                        point_index,
                        int(random_seed),
                        grid_signature,
                        *row,
                    ]
                )
                output.flush()
                cycle_scanned += 1
                if _is_screening_candidate(
                    row, max_delta_r_over_r, max_epsilon_gamma_proxy
                ):
                    cycle_candidates += 1
                    if first_candidate is None:
                        first_candidate = row

            total_scanned += cycle_scanned
            total_candidates += cycle_candidates
            completed_cycles += 1
            if first_candidate is None:
                candidate_summary = "sin candidatos del proxy"
            else:
                candidate_summary = (
                    "primer candidato: "
                    f"g={first_candidate[0]:.8g} GeV^-1, "
                    f"lambda={first_candidate[3]:.8g}, "
                    f"m0={first_candidate[4]:.8g} eV"
                )
            print(
                f"Ciclo {cycle_index + 1} ({cycle_description}): "
                f"{cycle_scanned} puntos nuevos, "
                f"{cycle_candidates} candidatos del proxy; "
                f"{candidate_summary}. CSV: {output_path}"
            )
            cycle_index += 1
            start_point_index = 0
            if (
                cycle_pause_seconds > 0.0
                and (
                    end_cycle_index is None
                    or cycle_index < end_cycle_index
                )
            ):
                time.sleep(cycle_pause_seconds)

    return {
        "cycles_completed": completed_cycles,
        "points_evaluated": total_scanned,
        "candidate_points": total_candidates,
    }


def _main():
    parser = argparse.ArgumentParser(
        description="Busca candidatos del proxy de thin-shell solar."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kappa-over-lambda", type=float, default=0.0)
    parser.add_argument(
        "--max-delta-r-over-r",
        type=float,
        default=MAX_DELTA_R_OVER_R,
    )
    parser.add_argument(
        "--max-epsilon-gamma-proxy",
        type=float,
        default=MAX_EPSILON_GAMMA_PROXY,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Sobrescribe el CSV de búsqueda en lugar de reanudarlo.",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Evalúa toda la grilla y escribe el CSV completo.",
    )
    parser.add_argument(
        "--first-candidate",
        action="store_true",
        help="Se detiene en el primer candidato de la grilla base.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        help="Número de ciclos; por defecto continúa hasta Ctrl+C.",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.exhaustive:
        run_scan(
            args.output,
            args.kappa_over_lambda,
            args.max_delta_r_over_r,
            args.max_epsilon_gamma_proxy,
        )
    elif args.first_candidate:
        first_candidate_output = args.output or Path(__file__).with_name(
            "thinshell_first_candidate.csv"
        )
        search_first_candidate(
            output_path=first_candidate_output,
            kappa_over_lambda=args.kappa_over_lambda,
            max_delta_r_over_r=args.max_delta_r_over_r,
            max_epsilon_gamma_proxy=args.max_epsilon_gamma_proxy,
            resume=not args.no_resume,
        )
    else:
        output_path = args.output or Path(__file__).with_name(
            "thinshell_cycles.csv"
        )
        try:
            search_parameter_cycles(
                output_path=output_path,
                kappa_over_lambda=args.kappa_over_lambda,
                max_delta_r_over_r=args.max_delta_r_over_r,
                max_epsilon_gamma_proxy=args.max_epsilon_gamma_proxy,
                cycles=args.cycles,
                batch_size=args.batch_size,
                random_seed=args.random_seed,
                cycle_pause_seconds=args.pause_seconds,
                resume=not args.no_resume,
            )
        except KeyboardInterrupt:
            print(f"Búsqueda interrumpida. Progreso guardado en {output_path}.")


if __name__ == "__main__":
    _main()
