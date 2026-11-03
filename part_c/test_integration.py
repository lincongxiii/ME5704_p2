"""Verification tests for the integration methods in part_c.py and the heat flow in heat_flow.py."""

from __future__ import annotations

import numpy as np

from heat_flow import (
    GRID_SPACINGS,
    heat_flow_boundary,
    heat_flow_source,
    solve_temperature,
    temperature_at,
)
from part_c import (
    boole_rule,
    composite_trapezoidal,
    curve_fit_integral,
    romberg_extrapolation,
    simpson_one_third,
)


def _run_case(name, function, exact, intervals_list=(4, 8, 16)) -> None:
    """Print errors for every requested method and several mesh sizes."""
    print(f"\n{name}: exact integral = {exact:.12f}")
    print("-" * 108)
    print(f"{'Intervals':>9} {'Method':<24} {'Numerical':>18} {'Absolute error':>18}")
    print("-" * 108)
    methods = (
        ("Trapezoidal", composite_trapezoidal),
        ("Simpson 1/3", simpson_one_third),
        ("Boole", boole_rule),
        ("Romberg", romberg_extrapolation),
    )
    for intervals in intervals_list:
        x = np.linspace(0.0, 4.0, intervals + 1)
        values = function(x)
        h = 4.0 / intervals
        for method_name, method in methods:
            result = method(values, h)
            print(f"{intervals:>9} {method_name:<24} {result:>18.12f} {abs(result - exact):>18.6e}")

        fitted, _, _ = curve_fit_integral(values, h)
        print(f"{intervals:>9} {'Curve fit (model check)':<24} {fitted:>18.12f} {abs(fitted - exact):>18.6e}")


# Nodal temperatures reported in Part A for h = 0.015625.
PART_A_TEMPERATURES = {(2.0, 2.0): 279.7359, (1.0, 1.0): 226.6487, (3.0, 1.0): 216.0715}


def test_energy_conservation(tolerance: float = 1e-8) -> None:
    """Heat leaving the source must equal heat leaving the outer edges on every grid."""
    print("\nEnergy conservation: |Q_edges - Q_source| per unit k")
    for h in GRID_SPACINGS:
        T = solve_temperature(h)
        difference = abs(heat_flow_boundary(T) - heat_flow_source(T))
        print(f"  h = {h:<10g} difference = {difference:.3e}")
        assert difference < tolerance, f"energy not conserved for h = {h}: {difference:.3e}"


def test_part_a_temperatures(h: float = 0.015625, tolerance: float = 1e-3) -> None:
    """The temperature field must reproduce the values reported in Part A."""
    T = solve_temperature(h)
    print(f"\nPart A temperature check (h = {h:g})")
    for (x, y), expected in PART_A_TEMPERATURES.items():
        computed = temperature_at(T, x, y)
        print(f"  T({x:g},{y:g}) = {computed:.4f} K (Part A: {expected:.4f} K)")
        assert abs(computed - expected) < tolerance, f"T({x:g},{y:g}) differs from Part A"


def main() -> None:
    """Run exact-integral checks and the heat-flow verification tests."""
    _run_case("Integral of x^2", lambda x: x**2, 64.0 / 3.0)
    _run_case("Integral of sin(x)", np.sin, 1.0 - np.cos(4.0))
    test_energy_conservation()
    test_part_a_temperatures()
    print("\nAll heat-flow tests passed.")


if __name__ == "__main__":
    main()
