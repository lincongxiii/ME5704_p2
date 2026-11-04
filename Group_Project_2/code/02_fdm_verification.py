import numpy as np
import csv
from pathlib import Path


# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
#
# STEP 3: INDEPENDENT VERIFICATION
#
# Course methods:
# Chapter 5.5 - Central Finite Difference
# Chapter 3   - Linear Equations
#
# Tests:
# 1. Constant temperature field
# 2. Linear temperature field
# ============================================================


# ============================================================
# 1. GRID SETTINGS
# ============================================================

Lx = 4.0
Ly = 4.0

dx = 0.5
dy = 0.5

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

x_values = np.linspace(0, Lx, Nx)
y_values = np.linspace(0, Ly, Ny)

output_dir = Path(__file__).resolve().parent.parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GENERIC FDM SOLVER
#
# Solves:
#
# d2T/dx2 + d2T/dy2 = 0
#
# With boundary temperatures supplied by
# boundary_function(x, y).
#
# No internal heat source in these verification tests.
# ============================================================

def solve_laplace(boundary_function):

    T = np.full((Ny, Nx), np.nan, dtype=float)

    fixed = np.zeros((Ny, Nx), dtype=bool)

    # --------------------------------------------------------
    # Apply analytical boundary conditions
    # --------------------------------------------------------

    for j, y in enumerate(y_values):
        for i, x in enumerate(x_values):

            is_boundary = (
                i == 0
                or i == Nx - 1
                or j == 0
                or j == Ny - 1
            )

            if is_boundary:

                T[j, i] = boundary_function(x, y)
                fixed[j, i] = True

    # --------------------------------------------------------
    # Identify unknown interior nodes
    # --------------------------------------------------------

    unknown_nodes = []

    for j in range(Ny):
        for i in range(Nx):

            if not fixed[j, i]:
                unknown_nodes.append((j, i))

    node_to_index = {
        node: index
        for index, node in enumerate(unknown_nodes)
    }

    n_unknown = len(unknown_nodes)

    # --------------------------------------------------------
    # Assemble A @ u = b
    # --------------------------------------------------------

    A = np.zeros((n_unknown, n_unknown), dtype=float)
    b = np.zeros(n_unknown, dtype=float)

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

                col = node_to_index[(nj, ni)]
                A[row, col] -= coefficient

    # --------------------------------------------------------
    # Solve linear equations
    # --------------------------------------------------------

    solution = np.linalg.solve(A, b)

    for index, (j, i) in enumerate(unknown_nodes):
        T[j, i] = solution[index]

    # --------------------------------------------------------
    # Calculate algebraic residual
    # --------------------------------------------------------

    residual = A @ solution - b

    max_residual = float(
        np.max(np.abs(residual))
    )

    return T, max_residual


# ============================================================
# 3. DEFINE ANALYTICAL TEST CASES
# ============================================================

def constant_field(x, y):
    return 200.0


def linear_field(x, y):
    return 200.0 + 10.0 * x


test_cases = [
    ("Constant Field", constant_field),
    ("Linear Field", linear_field)
]


# ============================================================
# 4. RUN VERIFICATION TESTS
# ============================================================

summary = []

all_node_results = []

for test_name, analytical_function in test_cases:

    print()
    print("=" * 75)
    print(f"TEST CASE: {test_name}")
    print("=" * 75)

    T_numerical, max_residual = solve_laplace(
        analytical_function
    )

    T_exact = np.zeros_like(T_numerical)

    for j, y in enumerate(y_values):
        for i, x in enumerate(x_values):

            T_exact[j, i] = analytical_function(x, y)

    # --------------------------------------------------------
    # Errors at all interior nodes
    # --------------------------------------------------------

    numerical_interior = T_numerical[1:-1, 1:-1]
    exact_interior = T_exact[1:-1, 1:-1]

    error = numerical_interior - exact_interior

    max_abs_error = float(
        np.max(np.abs(error))
    )

    rmse = float(
        np.sqrt(np.mean(error**2))
    )

    # --------------------------------------------------------
    # Boundary error
    # --------------------------------------------------------

    boundary_mask = np.zeros_like(T_numerical, dtype=bool)

    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    boundary_error = float(
        np.max(
            np.abs(
                T_numerical[boundary_mask]
                - T_exact[boundary_mask]
            )
        )
    )

    # --------------------------------------------------------
    # Print verification metrics
    # --------------------------------------------------------

    print(f"Grid size                 : {Nx} x {Ny}")
    print(f"Grid spacing              : {dx}")
    print(f"Maximum absolute error    : {max_abs_error:.12e} K")
    print(f"RMSE                      : {rmse:.12e} K")
    print(f"Maximum algebraic residual: {max_residual:.12e}")
    print(f"Maximum boundary error    : {boundary_error:.12e} K")

    # --------------------------------------------------------
    # Print nine target locations
    # --------------------------------------------------------

    print()
    print("NINE TARGET LOCATIONS")
    print("-" * 75)

    print(
        "{:>8}{:>8}{:>18}{:>18}{:>18}".format(
            "x", "y", "Numerical", "Analytical", "Abs Error"
        )
    )

    for y in [1.0, 2.0, 3.0]:
        for x in [1.0, 2.0, 3.0]:

            i = int(round(x / dx))
            j = int(round(y / dy))

            numerical = float(T_numerical[j, i])
            analytical = float(T_exact[j, i])
            abs_error = abs(numerical - analytical)

            print(
                f"{x:8.2f}"
                f"{y:8.2f}"
                f"{numerical:18.10f}"
                f"{analytical:18.10f}"
                f"{abs_error:18.10e}"
            )

            all_node_results.append([
                test_name,
                x,
                y,
                numerical,
                analytical,
                abs_error
            ])

    # --------------------------------------------------------
    # Verification status
    # --------------------------------------------------------

    tolerance = 1e-9

    passed = (
        max_abs_error < tolerance
        and boundary_error < tolerance
        and max_residual < tolerance
    )

    print()
    print(
        "VERIFICATION STATUS:",
        "PASS" if passed else "FAIL"
    )

    summary.append([
        test_name,
        max_abs_error,
        rmse,
        max_residual,
        boundary_error,
        "PASS" if passed else "FAIL"
    ])


# ============================================================
# 5. SAVE VERIFICATION RESULTS
# ============================================================

summary_file = output_dir / "fdm_verification_summary.csv"

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "test_case",
        "max_absolute_error_K",
        "rmse_K",
        "max_algebraic_residual",
        "max_boundary_error_K",
        "status"
    ])

    writer.writerows(summary)


nodes_file = output_dir / "fdm_verification_nodes.csv"

with open(
    nodes_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "test_case",
        "x",
        "y",
        "numerical_K",
        "analytical_K",
        "absolute_error_K"
    ])

    writer.writerows(all_node_results)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("FINAL VERIFICATION SUMMARY")
print("=" * 75)

for row in summary:

    print(
        f"{row[0]:20s} | "
        f"Max Error = {row[1]:.3e} K | "
        f"RMSE = {row[2]:.3e} K | "
        f"{row[5]}"
    )

print()
print("FILES SAVED:")
print(summary_file)
print(nodes_file)

print()
print("NOTE:")
print(
    "These tests verify the standard Laplace FDM solver."
)
print(
    "Internal heat-source discretisation and grid "
    "convergence require separate checks."
)