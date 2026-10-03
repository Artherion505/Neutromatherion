import math

from solar_thinshell import (
    Mpl_GeV,
    Phi_sun,
    find_N0,
    rho_in_GeV4,
    rho_kgm3_to_GeV4,
    rho_out_GeV4,
    screening_diagnostics,
)


def relative_residual(rho, g, lam, m0, field_value, kappa_over_lambda=0.0):
    effective_coupling = g - kappa_over_lambda
    terms = (
        m0**2 * field_value,
        lam * field_value**3,
        effective_coupling * rho,
    )
    return abs(sum(terms)) / max(sum(abs(term) for term in terms), 1e-300)


def run_checks():
    expected_density = 4.310130573076879e-18
    assert math.isclose(
        rho_kgm3_to_GeV4(1000.0), expected_density, rel_tol=1e-12
    )
    assert math.isclose(Phi_sun, 2.120615921867822e-6, rel_tol=1e-12)

    g = 1e-20
    m0 = 1e-30 * 1e-9
    cases = (
        (1e-75, 4.646260602860241, 1.1854569994416853e-3),
        (1e-69, 4.646260602860241e-2, 2.3045595550908553e-5),
        (1e-40, 1.0010065021730556e-11, 1.0696941155995104e-24),
    )

    for lam, expected_delta, expected_proxy in cases:
        n_in = find_N0(rho_in_GeV4, g, lam, m0)
        n_out = find_N0(rho_out_GeV4, g, lam, m0)
        assert n_in < n_out < 0.0
        assert relative_residual(rho_in_GeV4, g, lam, m0, n_in) < 1e-12
        assert relative_residual(rho_out_GeV4, g, lam, m0, n_out) < 1e-12

        delta, proxy = screening_diagnostics(n_in, n_out, g)
        assert math.isclose(delta, expected_delta, rel_tol=2e-8)
        assert math.isclose(proxy, expected_proxy, rel_tol=2e-8)
        print(f"lambda={lam:.0e}: DeltaR/R={delta:.6e}, epsilon_gamma_proxy={proxy:.6e}")

    lam = 1e-69
    n_trace = find_N0(rho_in_GeV4, g, lam, m0)
    n_half_coupling = find_N0(
        rho_in_GeV4, g, lam, m0, kappa_over_lambda=0.5 * g
    )
    n_equivalent = find_N0(rho_in_GeV4, 0.5 * g, lam, m0)
    assert math.isclose(n_half_coupling, n_equivalent, rel_tol=1e-12)
    assert relative_residual(
        rho_in_GeV4, g, lam, m0, n_half_coupling, 0.5 * g
    ) < 1e-12

    n_cancelled = find_N0(
        rho_in_GeV4, g, lam, m0, kappa_over_lambda=g
    )
    assert n_cancelled == 0.0
    delta_cancelled, proxy_cancelled = screening_diagnostics(
        n_cancelled, n_cancelled, 0.0
    )
    assert math.isnan(delta_cancelled)
    assert proxy_cancelled == 0.0

    n_reversed = find_N0(
        rho_in_GeV4, g, lam, m0, kappa_over_lambda=2.0 * g
    )
    assert math.isclose(n_reversed, -n_trace, rel_tol=1e-12)
    assert relative_residual(
        rho_in_GeV4, g, lam, m0, n_reversed, 2.0 * g
    ) < 1e-12
    delta_trace, proxy_trace = screening_diagnostics(
        find_N0(rho_in_GeV4, g, lam, m0),
        find_N0(rho_out_GeV4, g, lam, m0),
        g,
    )
    delta_reversed, proxy_reversed = screening_diagnostics(
        n_reversed,
        find_N0(rho_out_GeV4, g, lam, m0, kappa_over_lambda=2.0 * g),
        -g,
    )
    assert math.isclose(delta_reversed, delta_trace, rel_tol=1e-12)
    assert math.isclose(proxy_reversed, proxy_trace, rel_tol=1e-12)

    print("Solar conversion, minima, and screening-proxy checks passed.")


if __name__ == "__main__":
    run_checks()
