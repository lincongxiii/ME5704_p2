import numpy as np
import csv
from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
#
# Chapter 5.5: Central Finite Difference
# Chapter 3: Linear Equations
#
# Grid Convergence + Normalized Residual
# ============================================================


# ===================== EDIT HERE ============================

Lx = 4.0
Ly = 4.0

T_boundary = 200.0
T_source = 350.0

source_x = 1.5
source_y_min = 2.0
source_y_max = 3.0

grid_spacings = [
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625
]

stability_tolerance_percent = 0.1

# ============================================================


target_points = [
    (float(x), float(y))
    for y in (1, 2, 3)
    for x in (1, 2, 3)
]

output_dir = (
    Path(__file__).resolve().parent.parent / "results"
)

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. FDM SOLVER
# ============================================================

def solve_temperature(h):

    nx_intervals = round(Lx / h)
    ny_intervals = round(Ly / h)

    if not (
        np.isclose(nx_intervals * h, Lx)
        and np.isclose(ny_intervals * h, Ly)
    ):
        raise ValueError(
            "Grid spacing must divide both domain lengths."
        )

    Nx = nx_intervals + 1
    Ny = ny_intervals + 1

    x_values = np.linspace(0, Lx, Nx)
    y_values = np.linspace(0, Ly, Ny)

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]

    T = np.full((Ny, Nx), np.nan, dtype=float)
    fixed = np.zeros((Ny, Nx), dtype=bool)

    # --------------------------------------------------------
    # Outer boundary: 200 K
    # --------------------------------------------------------

    T[0, :] = T_boundary
    T[-1, :] = T_boundary
    T[:, 0] = T_boundary
    T[:, -1] = T_boundary

    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True

    outer_mask = np.zeros_like(fixed)

    outer_mask[0, :] = True
    outer_mask[-1, :] = True
    outer_mask[:, 0] = True
    outer_mask[:, -1] = True

    # --------------------------------------------------------
    # Internal heat source: 350 K
    # --------------------------------------------------------

    source_mask = (
        np.isclose(x_values[None, :], source_x)
        &
        (y_values[:, None] >= source_y_min - 1e-12)
        &
        (y_values[:, None] <= source_y_max + 1e-12)
    )

    if np.any(source_mask & outer_mask):
        raise ValueError(
            "Source overlaps outer boundary."
        )

    expected_sources = (
        round((source_y_max - source_y_min) / h) + 1
    )

    source_count = int(np.count_nonzero(source_mask))

    if source_count != expected_sources:
        raise ValueError(
            "Source endpoints are not aligned with the grid."
        )

    T[source_mask] = T_source
    fixed[source_mask] = True

    # --------------------------------------------------------
    # Unknown nodes
    # --------------------------------------------------------

    unknown_nodes = [
        (j, i)
        for j in range(1, Ny - 1)
        for i in range(1, Nx - 1)
        if not fixed[j, i]
    ]

    n = len(unknown_nodes)

    node_index = {
        node: index
        for index, node in enumerate(unknown_nodes)
    }

    # --------------------------------------------------------
    # Assemble A @ u = b
    # --------------------------------------------------------

    A = lil_matrix((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    cx = 1.0 / dx**2
    cy = 1.0 / dy**2

    directions = [
        (0, 1, cx),
        (0, -1, cx),
        (1, 0, cy),
        (-1, 0, cy)
    ]

    for row, (j, i) in enumerate(unknown_nodes):

        A[row, row] = 2.0 * cx + 2.0 * cy

        for dj, di, coefficient in directions:

            nj = j + dj
            ni = i + di

            if fixed[nj, ni]:

                b[row] += coefficient * T[nj, ni]

            else:

                col = node_index[(nj, ni)]
                A[row, col] -= coefficient

    # --------------------------------------------------------
    # Solve linear equations
    # --------------------------------------------------------

    A = A.tocsr()

    solution = spsolve(A, b)

    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "Non-finite solution encountered."
        )

    for index, (j, i) in enumerate(unknown_nodes):
        T[j, i] = solution[index]

    # --------------------------------------------------------
    # Absolute and normalized algebraic residuals
    # --------------------------------------------------------

    residual = A @ solution - b

    max_residual = float(
        np.linalg.norm(residual, ord=np.inf)
    )

    A_inf = float(
        np.max(
            np.asarray(np.abs(A).sum(axis=1)).ravel()
        )
    )

    solution_inf = float(
        np.linalg.norm(solution, ord=np.inf)
    )

    b_inf = float(
        np.linalg.norm(b, ord=np.inf)
    )

    denominator = A_inf * solution_inf + b_inf

    if denominator > 0:
        normalized_residual = max_residual / denominator
    else:
        normalized_residual = (
            0.0 if max_residual == 0 else np.inf
        )

    # --------------------------------------------------------
    # Boundary verification
    # --------------------------------------------------------

    boundary_error = float(
        np.max(np.abs(T[outer_mask] - T_boundary))
    )

    source_error = float(
        np.max(np.abs(T[source_mask] - T_source))
    )

    if boundary_error > 1e-9 or source_error > 1e-9:
        raise RuntimeError(
            "Dirichlet boundary conditions were not maintained."
        )

    if (
        np.any(~np.isfinite(T))
        or T.min() < T_boundary - 1e-8
        or T.max() > T_source + 1e-8
    ):
        raise RuntimeError(
            "Temperature field violates expected bounds."
        )

    # --------------------------------------------------------
    # Extract nine target temperatures
    # --------------------------------------------------------

    target_temperatures = {}

    for x, y in target_points:

        i = int(round(x / dx))
        j = int(round(y / dy))

        if not (
            np.isclose(x_values[i], x)
            and np.isclose(y_values[j], y)
        ):
            raise ValueError(
                f"Target {x, y} is not on the grid."
            )

        target_temperatures[(x, y)] = float(T[j, i])

    return {
        "h": h,
        "Nx": Nx,
        "Ny": Ny,
        "unknowns": n,
        "source_count": source_count,
        "T": T,
        "x": x_values,
        "y": y_values,
        "targets": target_temperatures,
        "residual": max_residual,
        "normalized_residual": normalized_residual,
        "boundary_error": boundary_error,
        "source_error": source_error
    }


# ============================================================
# 2. RUN ALL GRIDS
# ============================================================

results = {}

print("\n" + "=" * 90)
print("EXTENDED GRID CONVERGENCE STUDY")
print("=" * 90)

for h in grid_spacings:

    result = solve_temperature(h)
    results[h] = result

    print()
    print(
        f"h = {h} | "
        f"Grid = {result['Nx']} x {result['Ny']} | "
        f"Unknowns = {result['unknowns']} | "
        f"Source nodes = {result['source_count']}"
    )

    print(
        f"Maximum residual    : "
        f"{result['residual']:.12e}"
    )

    print(
        f"Normalized residual : "
        f"{result['normalized_residual']:.12e}"
    )

    print(
        f"Outer boundary error: "
        f"{result['boundary_error']:.12e} K"
    )

    print(
        f"Source boundary error: "
        f"{result['source_error']:.12e} K"
    )


# ============================================================
# 3. NINE TARGET TEMPERATURES
# ============================================================

print("\n" + "=" * 110)
print("NINE TARGET TEMPERATURES (K)")
print("=" * 110)

header = f"{'Point':>12}"

for h in grid_spacings:
    header += f"{('h=' + str(h)):>18}"

print(header)

comparison_rows = []

for point in target_points:

    values = [
        results[h]["targets"][point]
        for h in grid_spacings
    ]

    line = f"{str(point):>12}"

    for value in values:
        line += f"{value:18.8f}"

    print(line)

    comparison_rows.append([
        point[0], point[1], *values
    ])


# ============================================================
# 4. INTER-GRID TEMPERATURE CHANGES
# ============================================================

print("\n" + "=" * 90)
print("INTER-GRID TEMPERATURE CHANGES")
print("=" * 90)

change_arrays = []

for k in range(len(grid_spacings) - 1):

    h_coarse = grid_spacings[k]
    h_fine = grid_spacings[k + 1]

    changes = np.array([
        abs(
            results[h_fine]["targets"][point]
            - results[h_coarse]["targets"][point]
        )
        for point in target_points
    ])

    change_arrays.append(changes)

    print(
        f"{h_coarse} -> {h_fine}: "
        f"Max = {changes.max():.10f} K, "
        f"Mean = {changes.mean():.10f} K, "
        f"Min = {changes.min():.10f} K"
    )


# ============================================================
# 5. OBSERVED CONVERGENCE ORDER
# ============================================================

print("\n" + "=" * 110)
print("OBSERVED CONVERGENCE ORDER")
print("=" * 110)

order_labels = [
    (
        f"p({grid_spacings[k]},"
        f"{grid_spacings[k+1]},"
        f"{grid_spacings[k+2]})"
    )
    for k in range(len(grid_spacings) - 2)
]

header = f"{'Point':>12}"

for label in order_labels:
    header += f"{label:>29}"

print(header)

observed_orders = []

for index, point in enumerate(target_points):

    p_values = []

    for k in range(len(change_arrays) - 1):

        d_coarse = change_arrays[k][index]
        d_fine = change_arrays[k + 1][index]

        if d_coarse > 1e-14 and d_fine > 1e-14:
            p = float(np.log2(d_coarse / d_fine))
        else:
            p = float("nan")

        p_values.append(p)

    observed_orders.append(p_values)

    line = f"{str(point):>12}"

    for p in p_values:
        line += f"{p:29.8f}"

    print(line)

latest_orders = np.array([
    values[-1] for values in observed_orders
])

valid_orders = latest_orders[
    np.isfinite(latest_orders)
]

if valid_orders.size:

    print()
    print(
        f"Latest observed order: "
        f"mean = {valid_orders.mean():.8f}, "
        f"min = {valid_orders.min():.8f}, "
        f"max = {valid_orders.max():.8f}"
    )


# ============================================================
# 6. FINEST GRID STABILITY CHECK
# ============================================================

print("\n" + "=" * 90)
print("FINEST GRID STABILITY CHECK")
print("=" * 90)

finest_h = grid_spacings[-1]
finest_change = change_arrays[-1]

finest_values = np.array([
    results[finest_h]["targets"][point]
    for point in target_points
])

relative_change = (
    finest_change
    / np.maximum(np.abs(finest_values), 1e-12)
    * 100
)

max_relative = float(np.max(relative_change))

print(f"Finest grid: {finest_h}")

print(
    f"Maximum absolute change: "
    f"{finest_change.max():.10f} K"
)

print(
    f"Mean absolute change   : "
    f"{finest_change.mean():.10f} K"
)

print(
    f"Maximum relative change: "
    f"{max_relative:.8f}%"
)

print(
    f"Stability threshold    : "
    f"{stability_tolerance_percent}%"
)

if max_relative < stability_tolerance_percent:
    print("STATUS: BELOW THRESHOLD")
else:
    print("STATUS: ABOVE THRESHOLD")

print(
    "Note: inter-grid differences are NOT true-error bounds."
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

with (
    output_dir / "fdm_grid_convergence_extended.csv"
).open("w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "x", "y",
        *[
            f"T_h{str(h).replace('.', '')}_K"
            for h in grid_spacings
        ]
    ])

    writer.writerows(comparison_rows)


with (
    output_dir / "fdm_observed_convergence_order.csv"
).open("w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow(["x", "y", *order_labels])

    writer.writerows([
        [point[0], point[1], *p_values]
        for point, p_values in zip(
            target_points, observed_orders
        )
    ])


with (
    output_dir / "fdm_residual_summary.csv"
).open("w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "h",
        "unknown_nodes",
        "max_absolute_residual",
        "normalized_residual",
        "outer_boundary_error_K",
        "source_boundary_error_K"
    ])

    for h in grid_spacings:

        result = results[h]

        writer.writerow([
            h,
            result["unknowns"],
            result["residual"],
            result["normalized_residual"],
            result["boundary_error"],
            result["source_error"]
        ])


# Save complete temperature fields

for h in grid_spacings:

    result = results[h]

    filename = (
        output_dir
        / f"full_temperature_grid_h{str(h).replace('.', '')}.csv"
    )

    with filename.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "y/x", *result["x"]
        ])

        for j in range(result["Ny"] - 1, -1, -1):

            writer.writerow([
                result["y"][j],
                *result["T"][j, :]
            ])


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)

print(
    f"FINAL SUMMARY: "
    f"{len(grid_spacings)} grid solutions completed."
)

print("Files saved to:", output_dir)

print(
    "Internal line source is represented "
    "by fixed Dirichlet grid nodes."
)