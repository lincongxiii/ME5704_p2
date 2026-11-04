import numpy as np
import csv
from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
#
# STEP 6.2: SMOOTH ANALYTICAL SOLUTION TEST
#
# Chapter 5.5: Central Finite Difference
# Chapter 3: Linear Equations
#
# Purpose:
# Verify the observed order of accuracy of the
# five-point central finite difference scheme.
# ============================================================


# ============================================================
# 1. PROBLEM SETTINGS
# ============================================================

Lx = 4.0
Ly = 4.0

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
# T(x,y) = 200 + sin(x)*sinh(y)
#
# d2T/dx2 = -sin(x)*sinh(y)
# d2T/dy2 =  sin(x)*sinh(y)
#
# Therefore:
# d2T/dx2 + d2T/dy2 = 0
#
# IMPORTANT:
# np.sin uses radians.
# ============================================================

def analytical_temperature(x, y):

    return 200.0 + np.sin(x) * np.sinh(y)


# ============================================================
# 3. FINITE DIFFERENCE SOLVER
# ============================================================

def solve_fdm(h):

    Nx = int(round(Lx / h)) + 1
    Ny = int(round(Ly / h)) + 1

    x_values = np.linspace(0, Lx, Nx)
    y_values = np.linspace(0, Ly, Ny)

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]

    X, Y = np.meshgrid(x_values, y_values)

    T_exact = analytical_temperature(X, Y)

    T = np.full((Ny, Nx), np.nan)

    fixed = np.zeros((Ny, Nx), dtype=bool)

    # --------------------------------------------------------
    # Apply exact boundary temperatures
    # --------------------------------------------------------

    T[0, :] = T_exact[0, :]
    T[-1, :] = T_exact[-1, :]
    T[:, 0] = T_exact[:, 0]
    T[:, -1] = T_exact[:, -1]

    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True

    # --------------------------------------------------------
    # Identify unknown nodes
    # --------------------------------------------------------

    unknown_nodes = [
        (j, i)
        for j in range(1, Ny - 1)
        for i in range(1, Nx - 1)
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
    b = np.zeros(n)

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
        raise RuntimeError("Non-finite solution.")

    for index, (j, i) in enumerate(unknown_nodes):

        T[j, i] = solution[index]

    # --------------------------------------------------------
    # Calculate numerical errors
    # --------------------------------------------------------

    error = T - T_exact

    interior_error = error[1:-1, 1:-1]

    max_error = float(
        np.max(np.abs(interior_error))
    )

    rmse = float(
        np.sqrt(np.mean(interior_error**2))
    )

    # --------------------------------------------------------
    # Error at the nine original target points
    # --------------------------------------------------------

    target_errors = []

    for y in [1.0, 2.0, 3.0]:
        for x in [1.0, 2.0, 3.0]:

            i = int(round(x / dx))
            j = int(round(y / dy))

            target_errors.append(
                abs(T[j, i] - T_exact[j, i])
            )

    max_target_error = float(max(target_errors))

    # --------------------------------------------------------
    # Normalized algebraic residual
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
        "max_target_error": max_target_error,
        "normalized_residual": normalized_residual
    }


# ============================================================
# 4. RUN GRID REFINEMENT TEST
# ============================================================

results = []

print()
print("=" * 105)
print("SMOOTH ANALYTICAL SOLUTION TEST")
print("=" * 105)

print("T_exact(x,y) = 200 + sin(x)*sinh(y)")
print("Laplace equation: d2T/dx2 + d2T/dy2 = 0")
print()

for h in grid_spacings:

    result = solve_fdm(h)
    results.append(result)

    print(
        f"h = {h:<9} "
        f"Grid = {result['Nx']:3d} x {result['Ny']:3d} "
        f"Unknowns = {result['unknowns']:6d}"
    )


# ============================================================
# 5. ERROR COMPARISON
# ============================================================

print()
print("=" * 105)
print("NUMERICAL ERROR COMPARISON")
print("=" * 105)

print(
    f"{'h':>12}"
    f"{'Max Error (K)':>22}"
    f"{'RMSE (K)':>22}"
    f"{'Target Error (K)':>22}"
)

print("-" * 85)

for result in results:

    print(
        f"{result['h']:12.5f}"
        f"{result['max_error']:22.12e}"
        f"{result['rmse']:22.12e}"
        f"{result['max_target_error']:22.12e}"
    )


# ============================================================
# 6. OBSERVED CONVERGENCE ORDER
#
# Since the exact solution is known:
#
# p = log2(E_coarse / E_fine)
#
# For second-order convergence:
# p should approach 2.
# ============================================================

print()
print("=" * 105)
print("OBSERVED CONVERGENCE ORDER")
print("=" * 105)

print(
    f"{'Grid Refinement':>25}"
    f"{'Max Error Order':>22}"
    f"{'RMSE Order':>22}"
    f"{'Target Error Order':>22}"
)

print("-" * 95)

order_rows = []

for k in range(len(results) - 1):

    coarse = results[k]
    fine = results[k + 1]

    def calculate_order(e_coarse, e_fine):

        if e_coarse <= 0 or e_fine <= 0:
            return np.nan

        return float(
            np.log2(e_coarse / e_fine)
        )

    p_max = calculate_order(
        coarse["max_error"],
        fine["max_error"]
    )

    p_rmse = calculate_order(
        coarse["rmse"],
        fine["rmse"]
    )

    p_target = calculate_order(
        coarse["max_target_error"],
        fine["max_target_error"]
    )

    label = (
        f"{coarse['h']} -> {fine['h']}"
    )

    print(
        f"{label:>25}"
        f"{p_max:22.8f}"
        f"{p_rmse:22.8f}"
        f"{p_target:22.8f}"
    )

    order_rows.append([
        coarse["h"],
        fine["h"],
        p_max,
        p_rmse,
        p_target
    ])


# ============================================================
# 7. ALGEBRAIC RESIDUAL CHECK
# ============================================================

print()
print("=" * 85)
print("NORMALIZED ALGEBRAIC RESIDUALS")
print("=" * 85)

for result in results:

    print(
        f"h = {result['h']:<9} "
        f"Residual = "
        f"{result['normalized_residual']:.12e}"
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

error_file = (
    output_dir / "fdm_smooth_solution_errors.csv"
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
        "max_target_error_K",
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
            r["max_target_error"],
            r["normalized_residual"]
        ])


order_file = (
    output_dir / "fdm_smooth_solution_orders.csv"
)

with order_file.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "h_coarse",
        "h_fine",
        "p_max_error",
        "p_rmse",
        "p_target_error"
    ])

    writer.writerows(order_rows)


# ============================================================
# 9. FINAL CONCLUSION
# ============================================================

print()
print("=" * 85)
print("FINAL CONCLUSION")
print("=" * 85)

latest_order = order_rows[-1][2]

print(
    f"Latest observed maximum-error order: "
    f"{latest_order:.8f}"
)

if np.isfinite(latest_order):

    if abs(latest_order - 2.0) < 0.15:

        print(
            "RESULT: Consistent with second-order convergence."
        )

    else:

        print(
            "RESULT: Not yet sufficiently close to "
            "second-order convergence."
        )

print()
print(
    "This test verifies convergence for a smooth "
    "analytical solution without an internal source."
)

print(
    "It does not independently verify the accuracy "
    "of the internal line-source model."
)

print()
print("FILES SAVED:")
print(error_file)
print(order_file)