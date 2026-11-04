import numpy as np
import csv
from pathlib import Path

# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
#
# Method:
# Chapter 5.5 - Central Finite Difference
# Chapter 3   - Linear Equations
#
# First numerical solution: grid spacing = 0.5
# ============================================================


# ============================================================
# 1. PROBLEM DEFINITION
# ============================================================

Lx = 4.0
Ly = 4.0

dx = 0.5
dy = 0.5

T_boundary = 200.0
T_source = 350.0

# Internal heat source:
# x = 1.5, 2 <= y <= 3

source_x = 1.5
source_y_min = 2.0
source_y_max = 3.0

# Requested temperature locations

target_points = [
    (1, 1), (2, 1), (3, 1),
    (1, 2), (2, 2), (3, 2),
    (1, 3), (2, 3), (3, 3)
]


# ============================================================
# 2. CREATE COMPUTATIONAL GRID
# ============================================================

Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

x_values = np.linspace(0, Lx, Nx)
y_values = np.linspace(0, Ly, Ny)

T = np.full((Ny, Nx), np.nan)

fixed = np.zeros((Ny, Nx), dtype=bool)

# Outer boundary: T = 200 K

T[0, :] = T_boundary
T[-1, :] = T_boundary
T[:, 0] = T_boundary
T[:, -1] = T_boundary

fixed[0, :] = True
fixed[-1, :] = True
fixed[:, 0] = True
fixed[:, -1] = True


# ============================================================
# 3. APPLY INTERNAL HEAT SOURCE
# ============================================================

source_nodes = []

for j, y in enumerate(y_values):
    for i, x in enumerate(x_values):

        on_source = (
            np.isclose(x, source_x)
            and source_y_min - 1e-12 <= y
            and y <= source_y_max + 1e-12
        )

        if on_source:

            T[j, i] = T_source
            fixed[j, i] = True

            source_nodes.append((x, y))

if len(source_nodes) == 0:
    raise ValueError(
        "The grid does not contain any source nodes."
    )

print("\nINTERNAL HEAT SOURCE NODES")

for point in source_nodes:
    print(point)


# ============================================================
# 4. IDENTIFY UNKNOWN TEMPERATURE NODES
# ============================================================

unknown_nodes = []

for j in range(Ny):
    for i in range(Nx):

        if not fixed[j, i]:
            unknown_nodes.append((j, i))

N_unknown = len(unknown_nodes)

node_to_index = {
    node: index
    for index, node in enumerate(unknown_nodes)
}

print("\nGRID INFORMATION")
print("=" * 55)

print(f"Grid spacing dx = {dx}")
print(f"Grid spacing dy = {dy}")
print(f"Grid size       = {Nx} x {Ny}")
print(f"Total nodes     = {Nx * Ny}")
print(f"Unknown nodes   = {N_unknown}")


# ============================================================
# 5. ASSEMBLE LINEAR SYSTEM
#
# Laplace equation:
#
# d2T/dx2 + d2T/dy2 = 0
#
# Central difference:
#
# (T_E - 2T_P + T_W)/dx^2
# +
# (T_N - 2T_P + T_S)/dy^2 = 0
#
# A @ u = b
# ============================================================

A = np.zeros((N_unknown, N_unknown))
b = np.zeros(N_unknown)

cx = 1.0 / dx**2
cy = 1.0 / dy**2

directions = [
    (0, 1, cx),    # East
    (0, -1, cx),   # West
    (1, 0, cy),    # North
    (-1, 0, cy)    # South
]

for row, (j, i) in enumerate(unknown_nodes):

    A[row, row] = 2.0 * cx + 2.0 * cy

    for dj, di, coefficient in directions:

        nj = j + dj
        ni = i + di

        if fixed[nj, ni]:

            # Known boundary/source temperature
            # contributes to RHS.

            b[row] += coefficient * T[nj, ni]

        else:

            # Unknown neighbouring temperature
            # contributes to matrix A.

            col = node_to_index[(nj, ni)]

            A[row, col] -= coefficient


# ============================================================
# 6. SOLVE LINEAR SYSTEM
# ============================================================

solution = np.linalg.solve(A, b)

for index, (j, i) in enumerate(unknown_nodes):
    T[j, i] = solution[index]


# ============================================================
# 7. CHECK LINEAR SYSTEM RESIDUAL
# ============================================================

residual = A @ solution - b

max_residual = np.max(np.abs(residual))

print("\nLINEAR SYSTEM VERIFICATION")
print("=" * 55)

print(f"Maximum |A@u - b| = {max_residual:.12e}")


# ============================================================
# 8. CHECK BOUNDARY CONDITIONS
# ============================================================

boundary_values = np.concatenate([
    T[0, :],
    T[-1, :],
    T[:, 0],
    T[:, -1]
])

boundary_error = np.max(
    np.abs(boundary_values - T_boundary)
)

source_error = max(
    abs(T[
        int(round(y / dy)),
        int(round(x / dx))
    ] - T_source)
    for x, y in source_nodes
)

print("\nBOUNDARY CONDITION VERIFICATION")
print("=" * 55)

print(f"Outer boundary error = {boundary_error:.12e}")
print(f"Source boundary error = {source_error:.12e}")


# ============================================================
# 9. EXTRACT NINE REQUESTED TEMPERATURES
# ============================================================

print("\nNINE REQUESTED TEMPERATURES")
print("=" * 55)

print(
    "{:>10}{:>10}{:>20}".format(
        "x", "y", "Temperature (K)"
    )
)

print("-" * 40)

results = []

for x, y in target_points:

    i = int(round(x / dx))
    j = int(round(y / dy))

    temperature = T[j, i]

    results.append((x, y, temperature))

    print(
        f"{x:10.2f}"
        f"{y:10.2f}"
        f"{temperature:20.8f}"
    )


# ============================================================
# 10. PRINT COMPLETE TEMPERATURE GRID
# ============================================================

print("\nCOMPLETE TEMPERATURE GRID (K)")
print("=" * 80)

print("Rows: y = 4 down to y = 0")
print("Columns: x = 0 to x = 4")
print()

for j in range(Ny - 1, -1, -1):

    print(
        f"y={y_values[j]:3.1f} | ",
        end=""
    )

    for i in range(Nx):

        print(f"{T[j, i]:9.3f}", end=" ")

    print()


# ============================================================
# 11. SAVE RESULTS
# ============================================================

output_dir = Path(__file__).resolve().parent.parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

nine_points_file = output_dir / "nine_temperatures_h05.csv"
full_grid_file = output_dir / "full_temperature_grid_h05.csv"

with open(
    nine_points_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "x", "y", "temperature_K"
    ])

    writer.writerows(results)


with open(
    full_grid_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "y/x", *x_values
    ])

    for j in range(Ny - 1, -1, -1):

        writer.writerow([
            y_values[j], *T[j, :]
        ])


print("\nFILES SAVED")
print("=" * 55)

print(nine_points_file)
print(full_grid_file)

print("\nFIRST FDM SOLUTION COMPLETED.")
print("NOTE: Grid convergence has not yet been verified.")