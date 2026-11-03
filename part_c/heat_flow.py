"""Steady-state heat flow from the line source, using the Part A temperature model.

The plate is 4 x 4 with unit thickness. All outer edges are held at 200 K and the
line x = 1.5, 2 <= y <= 3 is held at 350 K. Laplace's equation is discretised with
the five-point central-difference stencil on a uniform grid (dx = dy = h) and solved
with scipy.sparse (LIL assembly -> CSR) and spsolve, exactly as in Part A.

Because the temperature field is steady, grad(T) does not change with time; only k
does. The heat absorbed is therefore

    h_abs = (Q/k) * integral_0^4 k(t) dt,

where Q/k = closed-contour integral of (-grad(T) . n) dl * thickness is the total
heat flow out of the source per unit conductivity. In steady state this equals the
total heat flow leaving through the four outer edges.

Discrete heat flows (dx = dy = h, so the flux (dT/h) times the edge length h is dT):
  * outer edges: sum over the four edges of (T_adj - 200), where T_adj is the
    interior node next to each edge node (corner nodes have no interior neighbour);
  * source: sum over every source node and every non-fixed neighbour of
    (350 - T_nb).
Summing the five-point equations over all free nodes shows both sums are equal,
so their difference checks energy conservation of the discrete solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


PLATE_SIZE = 4.0
T_BOUNDARY = 200.0
T_SOURCE = 350.0
SOURCE_X = 1.5
SOURCE_Y_RANGE = (2.0, 3.0)
GRID_SPACINGS = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)


def _grid_index(coordinate: float, h: float) -> int:
    """Return the node index of a coordinate that must lie exactly on the grid."""
    index = coordinate / h
    if abs(index - round(index)) > 1e-9:
        raise ValueError(f"coordinate {coordinate} is not a grid line for h = {h}")
    return int(round(index))


def _spacing_from_field(T: np.ndarray) -> float:
    """Recover the grid spacing from the shape of a temperature field."""
    return PLATE_SIZE / (T.shape[0] - 1)


def _fixed_masks(h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks [i, j] of the outer-boundary nodes and the source nodes."""
    n = _grid_index(PLATE_SIZE, h)
    boundary = np.zeros((n + 1, n + 1), dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True

    source = np.zeros_like(boundary)
    i_source = _grid_index(SOURCE_X, h)
    j_low = _grid_index(SOURCE_Y_RANGE[0], h)
    j_high = _grid_index(SOURCE_Y_RANGE[1], h)
    source[i_source, j_low : j_high + 1] = True
    return boundary, source


def solve_temperature(h: float) -> np.ndarray:
    """Solve the five-point Laplace system and return T[i, j] at x = i*h, y = j*h."""
    boundary, source = _fixed_masks(h)
    size = boundary.shape[0]
    n_nodes = size * size
    # Node numbering k = i * size + j (x index i, y index j).
    node = np.arange(n_nodes).reshape(size, size)
    fixed = (boundary | source).ravel()
    free = node.ravel()[~fixed]

    # Fixed-temperature nodes: identity rows. Free nodes: five-point stencil
    # T_E + T_W + T_N + T_S - 4 T_P = 0. Entries are collected first and then
    # written into the LIL matrix in one vectorised assignment.
    rows = [node.ravel()[fixed], free]
    cols = [node.ravel()[fixed], free]
    vals = [np.ones(fixed.sum()), np.full(free.size, -4.0)]
    for offset in (size, -size, 1, -1):  # E, W, N, S neighbours
        rows.append(free)
        cols.append(free + offset)
        vals.append(np.ones(free.size))

    A = lil_matrix((n_nodes, n_nodes))
    A[np.concatenate(rows), np.concatenate(cols)] = np.concatenate(vals)
    b = np.where(boundary, T_BOUNDARY, np.where(source, T_SOURCE, 0.0)).ravel()

    return spsolve(A.tocsr(), b).reshape(size, size)


def temperature_at(T: np.ndarray, x: float, y: float) -> float:
    """Return the nodal temperature at grid point (x, y)."""
    h = _spacing_from_field(T)
    return float(T[_grid_index(x, h), _grid_index(y, h)])


def heat_flow_boundary(T: np.ndarray) -> float:
    """Total heat flow per unit k leaving through the four outer edges (corners excluded)."""
    edge_neighbours = np.concatenate(
        (
            T[1, 1:-1],  # next to x = 0
            T[-2, 1:-1],  # next to x = 4
            T[1:-1, 1],  # next to y = 0
            T[1:-1, -2],  # next to y = 4
        )
    )
    return float(np.sum(edge_neighbours - T_BOUNDARY))


def heat_flow_source(T: np.ndarray) -> float:
    """Total heat flow per unit k leaving the source nodes into non-fixed neighbours."""
    boundary, source = _fixed_masks(_spacing_from_field(T))
    fixed = boundary | source
    total = 0.0
    for i, j in zip(*np.nonzero(source)):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if not fixed[i + di, j + dj]:
                total += T_SOURCE - T[i + di, j + dj]
    return float(total)


@dataclass(frozen=True)
class GridStudy:
    """Q/k on every grid, the observed convergence order, and the extrapolated Q*/k."""

    spacings: Tuple[float, ...]
    q_boundary: Tuple[float, ...]
    q_source: Tuple[float, ...]
    t_centre: Tuple[float, ...]
    order: float
    extrapolated: float


@lru_cache(maxsize=None)
def run_grid_study(spacings: Tuple[float, ...] = GRID_SPACINGS) -> GridStudy:
    """Solve on every grid, then estimate p and Q*/k from the three finest grids."""
    q_boundary, q_source, t_centre = [], [], []
    for h in spacings:
        T = solve_temperature(h)
        q_boundary.append(heat_flow_boundary(T))
        q_source.append(heat_flow_source(T))
        t_centre.append(temperature_at(T, 2.0, 2.0))

    # Three finest grids, coarse -> fine, with a constant refinement ratio r.
    q1, q2, q3 = q_source[-3:]
    ratio = spacings[-3] / spacings[-2]
    order = float(np.log((q1 - q2) / (q2 - q3)) / np.log(ratio))
    # Richardson extrapolation: Q* = Q_fine + (Q_fine - Q_medium) / (r^p - 1).
    extrapolated = float(q3 + (q3 - q2) / (ratio**order - 1.0))

    return GridStudy(
        spacings=tuple(spacings),
        q_boundary=tuple(q_boundary),
        q_source=tuple(q_source),
        t_centre=tuple(t_centre),
        order=order,
        extrapolated=extrapolated,
    )


def plot_grid_study(study: GridStudy, filename: str = "Q_vs_h.png") -> None:
    """Plot Q/k against grid spacing with the extrapolated value as a reference line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(study.extrapolated, color="#5f5e5a", linewidth=1.5, linestyle="--")
    ax.plot(
        study.spacings,
        study.q_source,
        "-o",
        color="#2a78d6",
        linewidth=2,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax.annotate(
        f"Richardson extrapolation Q*/k = {study.extrapolated:.2f}",
        xy=(max(study.spacings), study.extrapolated),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#5f5e5a",
    )
    ax.set_xlim(0.0, 1.05 * max(study.spacings))
    ax.set_xlabel("Grid spacing, h")
    ax.set_ylabel("Q/k (heat flow per unit conductivity)")
    ax.set_title(f"Grid convergence of Q/k (observed order p = {study.order:.3f})")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _print_results() -> None:
    """Print the grid study table and save Q_vs_h.png."""
    study = run_grid_study()
    print("Heat flow per unit conductivity, Q/k (unit thickness)")
    print("-" * 86)
    print(
        f"{'h':>10} {'Q/k (edges)':>16} {'Q/k (source)':>16}"
        f" {'Difference':>14} {'T(2,2) [K]':>14}"
    )
    print("-" * 86)
    for h, q_edges, q_src, t_centre in zip(
        study.spacings, study.q_boundary, study.q_source, study.t_centre
    ):
        print(
            f"{h:>10.6g} {q_edges:>16.6f} {q_src:>16.6f}"
            f" {q_edges - q_src:>14.3e} {t_centre:>14.6f}"
        )
    print("-" * 86)
    print(f"Observed convergence order (three finest grids): p = {study.order:.4f}")
    print(f"Richardson-extrapolated Q*/k = {study.extrapolated:.4f}")

    plot_grid_study(study)


if __name__ == "__main__":
    _print_results()
