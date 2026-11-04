import numpy as np
import csv
from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
#
# STEP 6.3: INTERNAL BOUNDARY DIAGNOSIS
#
# Chapter 5.5: Central Finite Difference
# Chapter 3: Linear Equations
#
# Test:
# Smooth analytical solution with a full-length
# internal Dirichlet boundary.
#
# This is a verification problem, NOT the original
# physical temperature field.
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

Lx = 4.0
Ly = 4.0

source_x = 1.5

grid_spacings = [
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125
]

output_dir = (
    Path(__file__).resolve().parent.parent / "results"
)

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ANALYTICAL SOLUTION
#
# T = 350 + sin(x - 1.5)*sinh(y)
#
# Laplace equation:
# T_xx + T_yy = 0
#
# At x = 1.5:
# T = 350 K
# ============================================================

def exact_temperature(x, y):

    return (
        350.0
        + np.sin(x - source_x) * np.sinh(y)
    )


# ============================================================
# 3. FDM SOLVER
# ============================================================

def solve_fdm(h):

    nx_intervals = round(Lx / h)
    ny_intervals = round(Ly / h)

    if not (
        np.isclose(nx_intervals * h, Lx)
        and np.isclose(ny_intervals * h, Ly)
        and np.isclose(round(source_x / h) * h, source_x)
    ):
        raise ValueError(
            "Grid is not aligned with domain/source."
        )

    Nx = nx_intervals + 1
    Ny = ny_intervals + 1

    x_values = np.linspace(0, Lx, Nx)
    y_values = np.linspace(0, Ly, Ny)

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]

    X, Y = np.meshgrid(x_values, y_values)

    T_exact = exact_temperature(X, Y)

    T = np.full((Ny, Nx), np.nan)
    fixed = np.zeros((Ny, Nx), dtype=bool)

    # --------------------------------------------------------
    # Outer boundary: exact analytical values
    # --------------------------------------------------------

    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True

    # --------------------------------------------------------
    # Full-length internal Dirichlet boundary
    # --------------------------------------------------------

    source_mask = np.broadcast_to(
        np.isclose(x_values[None, :], source_x),
        (Ny, Nx)
    )

    fixed[source_mask] = True

    # Apply exact temperature to all fixed nodes

    T[fixed] = T_exact[fixed]

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
    # Assemble linear system
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
    # Solve
    # --------------------------------------------------------

    A = A.tocsr()

    solution = spsolve(A, b)

    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "Non-finite solution."
        )

    for index, (j, i) in enumerate(unknown_nodes):
        T[j, i] = solution[index]

    # --------------------------------------------------------
    # Numerical errors
    # --------------------------------------------------------

    error = T - T_exact

    unknown_error = np.array([
        error[j, i]
        for j, i in unknown_nodes
    ])

    max_error = float(
        np.max(np.abs(unknown_error))
    )

    rmse = float(
        np.sqrt(np.mean(unknown_error**2))
    )

    # --------------------------------------------------------
    # Check internal boundary
    # --------------------------------------------------------

    source_error = float(
        np.max(
            np.abs(T[source_mask] - 350.0)
        )
    )

    # --------------------------------------------------------
    # Normalized residual
    # --------------------------------------------------------

    residual = A @ solution - b

    residual_inf = float(
        np.linalg.norm(residual, ord=np.inf)
    )

    A_inf = float(
        np.max(
            np.asarray(np.abs(A).sum(axis=1)).ravel()
        )
    )

    u_inf = float(
        np.linalg.norm(solution, ord=np.inf)
    )

    b_inf = float(
        np.linalg.norm(b, ord=np.inf)
    )

    denominator = A_inf * u_inf + b_inf

    normalized_residual = (
        residual_inf / denominator
        if denominator > 0 else np.nan
    )

    return {
        "h": h,
        "Nx": Nx,
        "Ny": Ny,
        "unknowns": n,
        "max_error": max_error,
        "rmse": rmse,
        "source_error": source_error,
        "normalized_residual": normalized_residual
    }


# ============================================================
# 4. RUN TEST
# ============================================================

results = []

print()
print("=" * 100)
print("FULL-LENGTH INTERNAL BOUNDARY TEST")
print("=" * 100)

print(
    "Exact solution: "
    "T(x,y) = 350 + sin(x-1.5)*sinh(y)"
)

print(
    "Internal boundary: x = 1.5, T = 350 K"
)

for h in grid_spacings:

    result = solve_fdm(h)
    results.append(result)

    print(
        f"h={h:<9} "
        f"Grid={result['Nx']}x{result['Ny']} "
        f"Unknowns={result['unknowns']}"
    )


# ============================================================
# 5. ERROR COMPARISON
# ============================================================

print()
print("=" * 100)
print("NUMERICAL ERROR COMPARISON")
print("=" * 100)

print(
    f"{'h':>12}"
    f"{'Max Error (K)':>22}"
    f"{'RMSE (K)':>22}"
)

for result in results:

    print(
        f"{result['h']:12.5f}"
        f"{result['max_error']:22.12e}"
        f"{result['rmse']:22.12e}"
    )


# ============================================================
# 6. OBSERVED CONVERGENCE ORDER
# ============================================================

print()
print("=" * 100)
print("OBSERVED CONVERGENCE ORDER")
print("=" * 100)

print(
    f"{'Refinement':>25}"
    f"{'Max Error Order':>22}"
    f"{'RMSE Order':>22}"
)

order_rows = []

for k in range(len(results) - 1):

    coarse = results[k]
    fine = results[k + 1]

    p_max = np.log2(
        coarse["max_error"] / fine["max_error"]
    )

    p_rmse = np.log2(
        coarse["rmse"] / fine["rmse"]
    )

    label = f"{coarse['h']} -> {fine['h']}"

    print(
        f"{label:>25}"
        f"{p_max:22.8f}"
        f"{p_rmse:22.8f}"
    )

    order_rows.append([
        coarse["h"],
        fine["h"],
        float(p_max),
        float(p_rmse)
    ])


# ============================================================
# 7. BOUNDARY AND RESIDUAL CHECKS
# ============================================================

print()
print("=" * 100)
print("BOUNDARY AND RESIDUAL VERIFICATION")
print("=" * 100)

for result in results:

    print(
        f"h={result['h']:<9} "
        f"Source Error={result['source_error']:.3e} K | "
        f"Normalized Residual="
        f"{result['normalized_residual']:.3e}"
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

error_file = (
    output_dir / "fdm_full_length_boundary_errors.csv"
)

with error_file.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "h",
        "Nx",
        "Ny",
        "unknowns",
        "max_error_K",
        "rmse_K",
        "source_error_K",
        "normalized_residual"
    ])

    for r in results:

        writer.writerow([
            r["h"],
            r["Nx"],
            r["Ny"],
            r["unknowns"],
            r["max_error"],
            r["rmse"],
            r["source_error"],
            r["normalized_residual"]
        ])


order_file = (
    output_dir / "fdm_full_length_boundary_orders.csv"
)

with order_file.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "h_coarse",
        "h_fine",
        "p_max_error",
        "p_rmse"
    ])

    writer.writerows(order_rows)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    f"Number of grid solutions: {len(results)}"
)

print(
    "This test uses a smooth analytical solution "
    "with an internal Dirichlet boundary."
)

print(
    "The original finite-length source has not "
    "been used in this verification test."
)

print()
print("FILES SAVED:")
print(error_file)
print(order_file)