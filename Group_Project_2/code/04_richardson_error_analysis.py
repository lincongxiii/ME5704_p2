import numpy as np
import csv
from pathlib import Path


# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
#
# STEP 10: FINAL RICHARDSON EXTRAPOLATION
#
# Chapter 5: Richardson Extrapolation
#
# Uses the three finest available grids.
# ============================================================


# ===================== SETTINGS ============================

results_dir = Path(__file__).resolve().parent.parent / "results"

input_file = results_dir / "fdm_grid_convergence_extended.csv"

output_file = results_dir / "richardson_extrapolation_final.csv"

# Reporting threshold (not a guaranteed true-error bound)
tolerance_percent = 0.1

# ============================================================


# ============================================================
# 1. READ GRID CONVERGENCE RESULTS
# ============================================================

if not input_file.exists():
    raise FileNotFoundError(f"Cannot find: {input_file}")

with input_file.open(
    "r", newline="", encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    fieldnames = reader.fieldnames
    data = list(reader)


if len(data) != 9:
    raise ValueError(
        f"Expected 9 target points, found {len(data)}."
    )


# ============================================================
# 2. AUTOMATICALLY IDENTIFY GRID COLUMNS
# ============================================================

def parse_grid_spacing(column):

    if not column.startswith("T_h") or not column.endswith("_K"):
        return None

    middle = column[3:-2]

    if not middle.isdigit():
        return None

    # Grid spacing is encoded without the decimal point.
    # Example:
    # T_h05_K      -> 0.5
    # T_h025_K     -> 0.25
    # T_h003125_K  -> 0.03125

    return int(middle) / (10 ** (len(middle) - 1))


grid_columns = []

for name in fieldnames:

    h = parse_grid_spacing(name)

    if h is not None:
        grid_columns.append((h, name))


grid_columns.sort(key=lambda item: item[0], reverse=True)

if len(grid_columns) < 3:
    raise ValueError(
        "At least three grid resolutions are required."
    )


# Select three finest grids

(h0, col0), (h1, col1), (h2, col2) = grid_columns[-3:]


if not (
    np.isclose(h0 / h1, 2.0)
    and np.isclose(h1 / h2, 2.0)
):
    raise ValueError(
        "The three finest grids must have a refinement ratio of 2."
    )


print()
print("=" * 115)
print("FINAL RICHARDSON EXTRAPOLATION")
print("=" * 115)

print(f"Coarse grid : h = {h0}")
print(f"Medium grid : h = {h1}")
print(f"Fine grid   : h = {h2}")

print()
print("Error model: T(h) = T* + C*h^p")
print("Observed order: p = log2(|T0-T1| / |T1-T2|)")
print("Richardson: T* = (2^p*T2 - T1)/(2^p - 1)")

print()


# ============================================================
# 3. CALCULATE RICHARDSON ESTIMATES
# ============================================================

print(
    f"{'Point':>12}"
    f"{'T_fine (K)':>16}"
    f"{'Order p':>13}"
    f"{'T_Rich (K)':>17}"
    f"{'Correction K':>17}"
    f"{'Relative %':>15}"
)

print("-" * 115)


output_rows = []

for row in data:

    x = float(row["x"])
    y = float(row["y"])

    T0 = float(row[col0])
    T1 = float(row[col1])
    T2 = float(row[col2])

    d1 = T1 - T0
    d2 = T2 - T1

    if (
        abs(d1) < 1e-14
        or abs(d2) < 1e-14
        or d1 * d2 <= 0
    ):
        p = np.nan
        T_rich = np.nan
        correction = np.nan
        relative = np.nan
        status = "UNDEFINED"

    else:

        p = float(np.log2(abs(d1 / d2)))

        factor = 2.0 ** p

        if abs(factor - 1.0) < 1e-10:
            T_rich = np.nan
            correction = np.nan
            relative = np.nan
            status = "UNDEFINED"

        else:

            T_rich = (
                factor * T2 - T1
            ) / (factor - 1.0)

            correction = T_rich - T2

            relative = (
                abs(correction) / abs(T_rich) * 100.0
                if abs(T_rich) > 1e-14
                else np.nan
            )

            status = (
                "BELOW" if relative < tolerance_percent
                else "ABOVE"
            )

    print(
        f"{str((x, y)):>12}"
        f"{T2:16.8f}"
        f"{p:13.6f}"
        f"{T_rich:17.8f}"
        f"{correction:17.8f}"
        f"{relative:15.6f}"
    )

    output_rows.append([
        x,
        y,
        T0,
        T1,
        T2,
        p,
        T_rich,
        correction,
        relative,
        status
    ])


# ============================================================
# 4. FINAL ACCURACY SUMMARY
# ============================================================

p_values = np.array([
    r[5] for r in output_rows
], dtype=float)

corrections = np.array([
    r[7] for r in output_rows
], dtype=float)

relative_values = np.array([
    r[8] for r in output_rows
], dtype=float)


print()
print("=" * 85)
print("FINAL ACCURACY SUMMARY")
print("=" * 85)

print(
    f"Mean observed order: "
    f"{np.nanmean(p_values):.8f}"
)

print(
    f"Maximum absolute correction: "
    f"{np.nanmax(np.abs(corrections)):.8f} K"
)

print(
    f"Mean absolute correction: "
    f"{np.nanmean(np.abs(corrections)):.8f} K"
)

print(
    f"Maximum relative correction: "
    f"{np.nanmax(relative_values):.8f}%"
)

print(
    f"Selected reporting threshold: "
    f"{tolerance_percent}%"
)

valid = np.isfinite(relative_values)

if not np.all(valid):

    print("STATUS: Some estimates are undefined.")

elif np.all(relative_values < tolerance_percent):

    print("STATUS: ALL POINTS BELOW THRESHOLD")

else:

    print("STATUS: SOME POINTS ABOVE THRESHOLD")


# ============================================================
# 5. SAVE FINAL RESULTS
# ============================================================

with output_file.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "x",
        "y",
        "T_coarse_K",
        "T_medium_K",
        "T_fine_K",
        "observed_order_p",
        "richardson_estimate_K",
        "correction_K",
        "relative_correction_percent",
        "status"
    ])

    writer.writerows(output_rows)


# ============================================================
# 6. FINAL NOTES
# ============================================================

print()
print("=" * 85)
print("FINAL NOTES")
print("=" * 85)

print(
    "Richardson estimates are extrapolated estimates, "
    "not analytical solutions."
)

print(
    "The relative correction is not a verified "
    "true-error bound."
)

print(
    "Observed convergence order is estimated "
    "from the three finest grids."
)

print()
print("RESULT SAVED:")
print(output_file)