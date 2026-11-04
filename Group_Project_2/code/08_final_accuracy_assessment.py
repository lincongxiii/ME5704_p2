import numpy as np
import csv
from pathlib import Path

# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
# STEP 8: FINAL ACCURACY ASSESSMENT
# ============================================================

results_dir = Path(__file__).resolve().parent.parent / "results"

grid_file = results_dir / "fdm_grid_convergence_extended.csv"
rich_file = results_dir / "richardson_extrapolation.csv"

tolerance_percent = 0.1

# ============================================================
# 1. READ RESULTS
# ============================================================

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

grid_data = read_csv(grid_file)
rich_data = read_csv(rich_file)

if len(grid_data) != 9 or len(rich_data) != 9:
    raise ValueError("Expected nine target points in each file.")

rich_lookup = {
    (float(r["x"]), float(r["y"])): r
    for r in rich_data
}

# ============================================================
# 2. COMPARE FINE GRID AND RICHARDSON ESTIMATES
# ============================================================

print("\nFINAL ACCURACY ASSESSMENT")
print("=" * 100)

print(
    f"{'Point':>12}"
    f"{'T_fine':>16}"
    f"{'T_Rich':>16}"
    f"{'Correction K':>18}"
    f"{'Relative %':>16}"
)

print("-" * 100)

output_rows = []

for row in grid_data:

    x = float(row["x"])
    y = float(row["y"])

    point = (x, y)

    T_fine = float(row["T_h003125_K"])

    rich = rich_lookup[point]
    T_rich = float(rich["richardson_estimate_K"])

    correction = abs(T_rich - T_fine)

    relative = correction / abs(T_rich) * 100

    status = (
        "BELOW" if relative < tolerance_percent
        else "ABOVE"
    )

    print(
        f"{str(point):>12}"
        f"{T_fine:16.8f}"
        f"{T_rich:16.8f}"
        f"{correction:18.8f}"
        f"{relative:16.6f}"
    )

    output_rows.append([
        x, y, T_fine, T_rich,
        correction, relative, status
    ])

# ============================================================
# 3. SUMMARY
# ============================================================

relative_errors = np.array([
    r[5] for r in output_rows
])

corrections = np.array([
    r[4] for r in output_rows
])

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"Maximum correction: {corrections.max():.8f} K")
print(f"Mean correction   : {corrections.mean():.8f} K")
print(f"Maximum relative correction: {relative_errors.max():.6f}%")
print(f"Selected threshold: {tolerance_percent}%")

if np.all(relative_errors < tolerance_percent):
    print("STATUS: ALL POINTS BELOW THRESHOLD")
else:
    print("STATUS: SOME POINTS ABOVE THRESHOLD")

print()
print("IMPORTANT:")
print("Richardson values are extrapolated estimates.")
print("Relative corrections are NOT verified true errors.")

# ============================================================
# 4. SAVE
# ============================================================

output_file = results_dir / "final_accuracy_assessment.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "x", "y",
        "T_fine_K",
        "T_Richardson_K",
        "absolute_correction_K",
        "relative_correction_percent",
        "status"
    ])

    writer.writerows(output_rows)

print("\nSaved:", output_file)