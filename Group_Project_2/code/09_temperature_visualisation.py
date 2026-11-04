import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ME5704 GROUP PROJECT 2 - PART A
# STEP 11: TEMPERATURE FIELD VISUALISATION
# ============================================================

results_dir = Path(__file__).resolve().parent.parent / "results"

grid_file = results_dir / "full_temperature_grid_h0015625.csv"
comparison_file = results_dir / "fdm_grid_convergence_extended.csv"

# ============================================================
# 1. LOAD FULL TEMPERATURE FIELD
# ============================================================

df = pd.read_csv(grid_file)

x = np.array([float(c) for c in df.columns[1:]])
y = df.iloc[:, 0].to_numpy(dtype=float)
T = df.iloc[:, 1:].to_numpy(dtype=float)

# CSV stores y from 4 down to 0.
# Sort ascending for plotting.

order = np.argsort(y)
y = y[order]
T = T[order, :]

X, Y = np.meshgrid(x, y)

target_points = [
    (i, j)
    for j in [1, 2, 3]
    for i in [1, 2, 3]
]

# ============================================================
# 2. TEMPERATURE CONTOUR PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

levels = np.linspace(200, 350, 31)

contour = ax.contourf(
    X, Y, T,
    levels=levels,
    cmap="inferno"
)

ax.contour(
    X, Y, T,
    levels=np.arange(210, 350, 10),
    colors="white",
    linewidths=0.45,
    alpha=0.65
)

ax.plot(
    [1.5, 1.5],
    [2, 3],
    color="cyan",
    linewidth=3,
    label="Internal heat source (350 K)"
)

ax.set(
    xlabel="x",
    ylabel="y",
    title="Steady-State Temperature Distribution"
)

ax.set_aspect("equal")
ax.legend(loc="upper right")

fig.colorbar(
    contour,
    ax=ax,
    label="Temperature (K)"
)

fig.tight_layout()

fig.savefig(
    results_dir / "temperature_contour.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 3. NINE TARGET POINTS VISUALISATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

heatmap = ax.pcolormesh(
    X, Y, T,
    shading="auto",
    cmap="inferno",
    vmin=200,
    vmax=350
)

for px, py in target_points:

    i = int(np.argmin(np.abs(x - px)))
    j = int(np.argmin(np.abs(y - py)))

    temperature = T[j, i]

    ax.scatter(
        px, py,
        color="cyan",
        edgecolor="black",
        s=55,
        zorder=5
    )

    ax.annotate(
        f"{temperature:.2f} K",
        (px, py),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
        color="white",
        fontweight="bold"
    )

ax.plot(
    [1.5, 1.5],
    [2, 3],
    color="cyan",
    linewidth=3
)

ax.set(
    xlabel="x",
    ylabel="y",
    title="Temperatures at the Nine Required Nodes"
)

ax.set_aspect("equal")

fig.colorbar(
    heatmap,
    ax=ax,
    label="Temperature (K)"
)

fig.tight_layout()

fig.savefig(
    results_dir / "nine_target_temperatures.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 4. GRID CONVERGENCE PLOT
# ============================================================

df_conv = pd.read_csv(comparison_file)

grid_columns = [
    "T_h05_K",
    "T_h025_K",
    "T_h0125_K",
    "T_h00625_K",
    "T_h003125_K",
    "T_h0015625_K"
]

h_values = np.array([
    0.5, 0.25, 0.125,
    0.0625, 0.03125, 0.015625
])

fig, ax = plt.subplots(figsize=(9, 6))

for _, row in df_conv.iterrows():

    values = row[grid_columns].to_numpy(dtype=float)

    label = f"({row['x']:.0f}, {row['y']:.0f})"

    ax.plot(
        h_values,
        values,
        marker="o",
        linewidth=1.4,
        label=label
    )

refinement_levels = np.arange(1, len(h_values) + 1)

ax.clear()

for _, row in df_conv.iterrows():

    values = row[grid_columns].to_numpy(dtype=float)

    label = f"({row['x']:.0f}, {row['y']:.0f})"

    ax.plot(
        refinement_levels,
        values,
        marker="o",
        linewidth=1.4,
        label=label
    )

ax.set_xticks(refinement_levels)

ax.set_xticklabels([
    "0.5",
    "0.25",
    "0.125",
    "0.0625",
    "0.03125",
    "0.015625"
])

ax.set(
    xlabel="Grid Spacing h (decreasing with refinement)",
    ylabel="Temperature (K)",
    title="Grid Convergence at Nine Target Nodes"
)

ax.grid(True, alpha=0.3)
ax.legend(ncol=3, fontsize=8)

ax.set(
    xlabel="Grid Spacing h",
    ylabel="Temperature (K)",
    title="Grid Convergence at Nine Target Nodes"
)

ax.grid(True, alpha=0.3)
ax.legend(ncol=3, fontsize=8)

fig.tight_layout()

fig.savefig(
    results_dir / "grid_convergence.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 5. FINAL MESSAGE
# ============================================================

print("\nTEMPERATURE VISUALISATION COMPLETED")
print("=" * 60)

print("Files generated:")
print("1. temperature_contour.png")
print("2. nine_target_temperatures.png")
print("3. grid_convergence.png")

print("\nSaved to:")
print(results_dir)