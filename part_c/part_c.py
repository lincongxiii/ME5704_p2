"""Numerical integration methods for the time-dependent heat-conduction problem."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from heat_flow import run_grid_study


# The steady temperature field does not change with time, so the heat absorbed is
# h = (Q/k) * integral_0^4 k(t) dt, where Q/k is the total heat flow out of the
# source per unit conductivity (unit thickness). See heat_flow.py.
GRID_STUDY = run_grid_study()

# Manual override for Q/k. Leave as None to use the Richardson-extrapolated Q*/k
# from the Part A grid study. Set it to a number (e.g. FLUX_OVERRIDE = 460.94) when
# the group agrees on a common value with Part B; table (a) then uses that value,
# while the grid-sensitivity table (b) always uses the grid-study values.
FLUX_OVERRIDE: Optional[float] = None

FLUX_FACTOR = GRID_STUDY.extrapolated if FLUX_OVERRIDE is None else float(FLUX_OVERRIDE)

TIME = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
CONDUCTIVITY = np.array([0.10, 0.25, 0.35, 0.40, 0.42])


def _validate_data(values: Sequence[float], h: float) -> np.ndarray:
    """Validate equally spaced samples and return them as a one-dimensional array."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("values must be a one-dimensional array with at least two points")
    if h <= 0:
        raise ValueError("h must be positive")
    return data


def composite_trapezoidal(values: Sequence[float], h: float) -> float:
    """Integrate equally spaced values with the composite trapezoidal rule."""
    data = _validate_data(values, h)
    return float(h * (0.5 * data[0] + data[1:-1].sum() + 0.5 * data[-1]))


def simpson_one_third(values: Sequence[float], h: float) -> float:
    """Integrate equally spaced values with the composite Simpson 1/3 rule."""
    data = _validate_data(values, h)
    intervals = data.size - 1
    if intervals % 2:
        raise ValueError("Simpson 1/3 rule requires an even number of intervals")
    return float(
        h
        / 3.0
        * (data[0] + data[-1] + 4.0 * data[1:-1:2].sum() + 2.0 * data[2:-1:2].sum())
    )


def boole_rule(values: Sequence[float], h: float) -> float:
    """Integrate equally spaced values with the composite Boole rule."""
    data = _validate_data(values, h)
    intervals = data.size - 1
    if intervals % 4:
        raise ValueError("Boole rule requires a number of intervals divisible by four")
    total = 7.0 * (data[0] + data[-1])
    total += 32.0 * data[1:-1:4].sum()
    total += 12.0 * data[2:-1:4].sum()
    total += 32.0 * data[3:-1:4].sum()
    total += 14.0 * data[4:-1:4].sum()
    return float(2.0 * h * total / 45.0)


def romberg_extrapolation(values: Sequence[float], h: float) -> float:
    """Apply Romberg extrapolation to trapezoidal results at h, 2h, and 4h."""
    data = _validate_data(values, h)
    intervals = data.size - 1
    if intervals < 4 or intervals % 4:
        raise ValueError("Romberg extrapolation requires intervals divisible by four")

    finest = composite_trapezoidal(data, h)
    medium = composite_trapezoidal(data[::2], 2.0 * h)
    coarse = composite_trapezoidal(data[::4], 4.0 * h)
    first_extrapolation = medium + (medium - coarse) / 3.0
    second_extrapolation = finest + (finest - medium) / 3.0
    return float(second_extrapolation + (second_extrapolation - first_extrapolation) / 15.0)


def _exponential_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.exp(-c * t)


def curve_fit_integral(
    values: Sequence[float], h: float
) -> Tuple[float, Tuple[float, float, float], float]:
    """Fit a-e^(-ct), integrate the fit analytically, and return R-squared."""
    data = _validate_data(values, h)
    t = np.arange(data.size, dtype=float) * h
    parameters, _ = curve_fit(
        _exponential_model,
        t,
        data,
        p0=(float(data[-1]), float(data[-1] - data[0]), 0.5),
        maxfev=10000,
    )
    a, b, c = parameters
    if abs(c) < 1e-14:
        integral = a * (t[-1] - t[0]) - b * (t[-1] - t[0])
    else:
        integral = a * t[-1] - (b / c) * (1.0 - np.exp(-c * t[-1]))
    fitted = _exponential_model(t, *parameters)
    residual_sum = np.sum((data - fitted) ** 2)
    total_sum = np.sum((data - data.mean()) ** 2)
    r_squared = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    return float(integral), (float(a), float(b), float(c)), float(r_squared)


def _print_results() -> None:
    """Print the requested comparison table and create the explanatory plot."""
    trapezoid = composite_trapezoidal(CONDUCTIVITY, 1.0)
    simpson = simpson_one_third(CONDUCTIVITY, 1.0)
    boole = boole_rule(CONDUCTIVITY, 1.0)
    romberg = romberg_extrapolation(CONDUCTIVITY, 1.0)
    fitted_integral, parameters, r_squared = curve_fit_integral(CONDUCTIVITY, 1.0)

    results: Dict[str, float] = {
        "Composite trapezoidal": trapezoid,
        "Simpson 1/3": simpson,
        "Boole": boole,
        "Romberg": romberg,
        "Curve fit": fitted_integral,
    }
    flux_source = (
        "Richardson-extrapolated Q*/k" if FLUX_OVERRIDE is None else "manual FLUX_OVERRIDE"
    )
    print(f"FLUX_FACTOR = Q/k = {FLUX_FACTOR:.6f} ({flux_source})")
    print("\n(a) Integral of k dt and heat absorbed, h = (Q/k) * integral")
    print("-" * 82)
    print(f"{'Method':<24} {'Integral':>12} {'Final h':>14} {'Rel. to Simpson':>20}")
    print("-" * 82)
    for method, integral in results.items():
        relative_difference = 100.0 * (integral - simpson) / simpson
        print(
            f"{method:<24} {integral:>12.6f} {FLUX_FACTOR * integral:>14.6f}"
            f" {relative_difference:>19.6f}%"
        )
    print("-" * 82)
    print(
        "Curve-fit parameters: "
        f"a = {parameters[0]:.8f}, b = {parameters[1]:.8f}, c = {parameters[2]:.8f}"
    )
    print(f"Curve-fit R^2: {r_squared:.8f}")

    # Sensitivity of the final h to the spatial grid used for Q/k.
    sensitivity_methods = (("Trapezoidal", trapezoid), ("Simpson 1/3", simpson), ("Boole", boole))
    flux_rows = [(f"h = {h:g}", q) for h, q in zip(GRID_STUDY.spacings, GRID_STUDY.q_source)]
    flux_rows.append(("Extrapolated", GRID_STUDY.extrapolated))
    print("\n(b) Grid sensitivity of the heat absorbed (rows: grid used for Q/k)")
    print("-" * 82)
    print(
        f"{'Grid':<16} {'Q/k':>14}"
        + "".join(f" {name:>16}" for name, _ in sensitivity_methods)
    )
    print("-" * 82)
    for label, flux in flux_rows:
        print(
            f"{label:<16} {flux:>14.4f}"
            + "".join(f" {flux * integral:>16.4f}" for _, integral in sensitivity_methods)
        )
    print("-" * 82)

    # Spread = (max - min) / max, over the six grids and over the five methods.
    grid_spread = 100.0 * (max(GRID_STUDY.q_source) - min(GRID_STUDY.q_source)) / max(
        GRID_STUDY.q_source
    )
    method_spread = 100.0 * (max(results.values()) - min(results.values())) / max(
        results.values()
    )
    print(
        f"\nConclusion: the spatial grid changes h by about {grid_spread:.1f}%"
        f" (h = {GRID_STUDY.spacings[0]:g} to {GRID_STUDY.spacings[-1]:g}), far more than"
        f" the {method_spread:.1f}% spread between integration methods;\n"
        "the accuracy of Q/k, not the time integration, controls the accuracy of h."
    )

    fit_t = np.linspace(TIME[0], TIME[-1], 300)
    fit_values = _exponential_model(fit_t, *parameters)
    plt.figure(figsize=(8, 5))
    plt.plot(TIME, CONDUCTIVITY, "o", label="Original data")
    plt.plot(fit_t, fit_values, "-", label="Exponential curve fit")
    plt.plot(TIME, CONDUCTIVITY, "--", label="Trapezoidal rule")
    plt.xlabel("Time, t")
    plt.ylabel("Thermal conductivity, k(t)")
    plt.title("Conductivity data and integration approximations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("k_vs_t.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    _print_results()
