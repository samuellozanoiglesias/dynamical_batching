"""Figure 1 -- ICLR-style version (2 panels).

Modified Version:
  * Font sizes increased across all labels, ticks, and legends.
  * Amplitude labels ("Low amplitude" / "High amplitude") moved on top 
    of subplots, styled in bold black font.
  * Left panel x-axis updated to "Training cycles" with ticks [0, 1, 2].
  * Subplot (b) legend enlarged and positioned outside on the far right.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ----------------------------------------------------------------- style
# Standard global font configuration with larger base font size.
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 14,             # Increased base font size
    "axes.linewidth": 1.1,
})

FONTSIZE_LEGEND = 24
FONTSIZE_TICKS = 22
FONTSIZE_LABELS = 24
FONTSIZE_TITLE = 24

CLASS_NAMES = ['Class 1', 'Class 2', 'Class 3']
#CLASS_BASE = ['#C0392B', '#2471A3', '#B7950B']
CLASS_BASE = ["#d7191c", "#808080", "#2b83ba",]

def lighten(hex_color, amount=0.72):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)


def darken(hex_color, amount=0.15):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(rgb * (1.0 - amount))


CLASS_FILL = [lighten(c, 0.22) for c in CLASS_BASE]
CLASS_EDGE = [darken(c, 0.15) for c in CLASS_BASE]

# Enlarged panel label font settings
PANEL_LABEL_KW = dict(fontsize=FONTSIZE_TITLE, fontweight='bold', va='bottom', ha='left')


def _compute_focus_weight(period_step: int, period_length: int, w_max: float):
    t = float(period_step)
    period = float(period_length)
    slope = 2.0 * (w_max - 1.0) / period
    if t < period / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    phase = t / period
    return focus_weight, phase


def compute_class_proportions(total_steps, period_length, w_max, num_classes=3):
    steps = np.arange(total_steps)
    full_cycle = num_classes * period_length
    weights = np.ones((total_steps, num_classes), dtype=np.float64)
    pos_in_cycle = steps % full_cycle
    active_class = (pos_in_cycle // period_length).astype(int)
    period_step = pos_in_cycle % period_length
    for step_idx in range(total_steps):
        c = int(active_class[step_idx])
        fw, _ = _compute_focus_weight(int(period_step[step_idx]), period_length, w_max)
        weights[step_idx, c] = fw
    proportions = weights / weights.sum(axis=1, keepdims=True)
    return steps, proportions


def style_3d_axes(ax, zoom=1.35, show_grid=True):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.tick_params(labelsize=14, pad=1)

    if show_grid:
        grid_rgba = mcolors.to_rgba('#cccccc', alpha=0.8)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid'].update({
                'color': grid_rgba,
                'linestyle': '--',
                'linewidth': 0.6
            })
        ax.grid(True)
    else:
        ax.grid(False)


# ----------------------------------------------------------------- layout
# Slightly enlarged figure dimensions and adjusted margins to accommodate external legend
fig = plt.figure(figsize=(12.0, 5.2), dpi=150)
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0], wspace=0.35,
              left=0.08, right=0.82, top=0.80, bottom=0.15)

# Increased vertical spacing between stacked subplots to avoid title overlap
gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.55)
ax_a1 = fig.add_subplot(gs_a[0, 0])
ax_a2 = fig.add_subplot(gs_a[1, 0], sharex=ax_a1)
ax_b = fig.add_subplot(gs[0, 1], projection='3d')

# ----------------------------------------------------------------- (a)
total_steps = 30000
period_length = 5000
amp_regimes = [(5.0, ax_a1, 'Low amplitude ($A=5$)'),
               (70.0, ax_a2, 'High amplitude ($A=70$)')]

for w_max, ax, title_str in amp_regimes:
    steps, proportions = compute_class_proportions(total_steps, period_length, w_max)
    for c in range(3):
        ax.plot(steps, proportions[:, c], color=CLASS_BASE[c], linewidth=3.0)
    ax.set_xlim(0, total_steps)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=FONTSIZE_TICKS)
    ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc', linewidth=0.5)
    ax.tick_params(axis='both', which='major', width=1.0, length=4, labelsize=FONTSIZE_TICKS)
    
    # Subplot titles placed on top, in bold black font
    ax.set_title(title_str, fontsize=FONTSIZE_TITLE, color='black', fontweight='bold', pad=8)

ax_a1.tick_params(labelbottom=False)

# Left panel X-axis modified to "Training cycles" [0, 1, 2]
ax_a2.set_xticks([0, 15000, 30000])
ax_a2.set_xticklabels(['0', '1', '2'], fontsize=FONTSIZE_TICKS)
ax_a2.set_xlabel('Training cycles', fontsize=FONTSIZE_LABELS, labelpad=4)
fig.text(0.02, 0.48, 'Class proportion', rotation='vertical',
         va='center', ha='center', fontsize=FONTSIZE_LABELS)

# ----------------------------------------------------------------- (b)
R_b, r_b = 0.8, 0.55
angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
z_vals = np.linspace(0, 1.0, 80)
theta_vals = np.linspace(0, 2 * np.pi, 80)
Z, THETA = np.meshgrid(z_vals, theta_vals)
for i, angle in enumerate(angles):
    x_c = R_b * np.cos(angle) * (1.0 - Z)
    y_c = R_b * np.sin(angle) * (1.0 - Z)
    r_z = r_b * (1.0 - Z)
    X = x_c + r_z * np.cos(THETA)
    Y = y_c + r_z * np.sin(THETA)
    # Solid surface with less transparency (alpha=0.38) and no wireframe lines
    ax_b.plot_surface(X, Y, Z, color=CLASS_FILL[i], alpha=0.25, shade=True, linewidth=0, antialiased=True)
    
    # Circular line around the bottom base of each cone (Z = 0)
    ax_b.plot(X[:, 0], Y[:, 0], Z[:, 0], color=CLASS_EDGE[i], linewidth=1.5, zorder=5)

t = np.linspace(0, 1.0, 500)
rotations = 3.75
theta_spiral = 2 * np.pi * rotations * t - np.pi / 6
r_spiral = (R_b - 0.25*r_b) * (1.0 - t)
x_spiral, y_spiral, z_spiral = r_spiral * np.cos(theta_spiral), r_spiral * np.sin(theta_spiral), t

rng = np.random.default_rng(42)
t_std = np.linspace(0, 0.45, 200)
x_std = x_spiral[0] + 0.12 * np.sin(8 * t_std) + rng.normal(0, 0.015, 200) + 0.15 * t_std
y_std = y_spiral[0] + 0.12 * np.cos(8 * t_std) + rng.normal(0, 0.015, 200) + 3.5 * t_std
z_std = 0.05 * t_std + rng.normal(0, 0.008, 200)

ax_b.scatter([x_spiral[0]], [y_spiral[0]], [z_spiral[0]], color='#1F4E79', edgecolors='black',
             linewidths=0.7, s=200, zorder=10, label=r'Initialization')
ax_b.scatter([0], [0], [1.0], color='#FFC000', marker='*', s=340, edgecolor='black',
             linewidth=0.8, zorder=12, label='Optimum')
ax_b.scatter([x_std[-1]], [y_std[-1]], [z_std[-1]], color='#C00000', marker='X', s=200,
             edgecolors='black', linewidths=0.7, zorder=10, label='Suboptimal')
ax_b.plot(x_std, y_std, z_std, color='#595959', linestyle=':', linewidth=2.4, alpha=0.9, label='Unif. Batching')
ax_b.plot(x_spiral, y_spiral, z_spiral, color='#107C41', linewidth=3.3, label='Dyn. Batching')


ax_b.set_xlabel(r'$\theta_1$', labelpad=2, fontsize=FONTSIZE_LABELS)
ax_b.set_ylabel(r'$\theta_2$', labelpad=2, fontsize=FONTSIZE_LABELS)
ax_b.zaxis.set_rotate_label(False)  # Disable auto-alignment along 3D projection
ax_b.set_zlabel(r'$\theta_3$', labelpad=2, fontsize=FONTSIZE_LABELS, rotation=0)
ax_b.view_init(elev=20, azim=50)
style_3d_axes(ax_b, zoom=1.35, show_grid=True)

ax_b.set_xticks([-0.8, 0, 0.8])
ax_b.set_xticklabels(['-1', '0', '1'], fontsize=FONTSIZE_TICKS)
ax_b.set_yticks([-0.8, 0, 0.8])
ax_b.set_yticklabels(['-1', '0', '1'], fontsize=FONTSIZE_TICKS)
ax_b.set_zticks([0.0, 0.5, 1.0])
ax_b.set_zticklabels(['-1', '0', '1'], fontsize=FONTSIZE_TICKS)

# 3D Subplot Legend placed outside the plot on the far right
ax_b.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), frameon=True, framealpha=0.9,
            edgecolor='#e0e0e0', facecolor='#ffffff', fontsize=FONTSIZE_LEGEND, handlelength=1.5)

# ----------------------------------------------------------------- panel labels (ICLR style)
ax_a1.text(-0.16, 1.22, '(a)', transform=ax_a1.transAxes, **PANEL_LABEL_KW)
ax_b.text2D(-0.10, 1.02, '(b)', transform=ax_b.transAxes, **PANEL_LABEL_KW)

# ----------------------------------------------------------------- global class legend
lines = [plt.Line2D([0], [0], color=CLASS_BASE[i], lw=2.8) for i in range(3)]
fig.legend(lines, CLASS_NAMES, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
           frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9,
           columnspacing=2.5, fontsize=FONTSIZE_LEGEND)

fig.savefig('figures/fig1_full_panel.png', bbox_inches='tight')
fig.savefig('figures/fig1_full_panel.pdf', bbox_inches='tight')
print("Successfully generated plot layout")