import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- USER PARAMETERS: 2D RANDOM WALK ONLY ---
n_walkers = 800
n_steps = 100
step_size = 1.0
trail_length = 30
sample_trails = n_walkers  # show trails for all walkers
interval = 40
fit_skip_initial = 5
np.random.seed(1)

# Theoretical D for 2D
D_theory = (step_size**2) / 4.0

# --- initialization ---

positions = np.zeros((n_walkers, 2))
history = np.zeros((trail_length + 1, n_walkers, 2))
history[-1] = positions.copy()
msd = []


trail_indices = np.arange(n_walkers)


# --- plotting setup ---
fig, (ax_walk, ax_msd) = plt.subplots(1, 2, figsize=(13, 6))

# scatter for current positions (2D only)
scat = ax_walk.scatter(positions[:, 0], positions[:, 1], s=8, alpha=0.6)
R = np.sqrt(n_steps) * step_size * 1.1
ax_walk.set_xlim(-R, R)
ax_walk.set_ylim(-R, R)
ax_walk.set_xlabel("x")
ax_walk.set_ylabel("y")
ax_walk.set_title(f"2D Random Walk (n_walkers={n_walkers})")

# trail lines
trail_lines = []
for _ in range(sample_trails):
    line, = ax_walk.plot([], [], lw=0.9, alpha=0.55)
    trail_lines.append(line)


# MSD plot with fitted line
line_msd, = ax_msd.plot([], [], lw=2, label="MSD (sim)")
line_fit, = ax_msd.plot([], [], lw=1.6, ls='--', label="Linear fit")
ax_msd.set_xlim(0, n_steps)
ax_msd.set_ylim(0, step_size**2 * n_steps * 1.5)
ax_msd.set_xlabel("Time step")
ax_msd.set_ylabel("MSD")
ax_msd.set_title("Mean Squared Displacement vs t")
ax_msd.grid(True)
ax_msd.legend(loc='upper left')


# text box for D estimates
txt = ax_msd.text(0.02, 0.95, f"Theoretical D = {D_theory:.4f}",
                  transform=ax_msd.transAxes, va='top', fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()


# helper: linear fit + r²
def linear_fit_with_r2(x, y):
    if len(x) < 2:
        return None
    p = np.polyfit(x, y, 1)
    slope, intercept = p
    y_pred = np.polyval(p, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


# update function (2D only)
def update(frame):
    global positions, history, msd

    # 2D step
    angles = np.random.uniform(0, 2 * np.pi, n_walkers)
    step = np.stack((np.cos(angles), np.sin(angles)), axis=1) * step_size
    positions += step

    # history buffer
    history = np.roll(history, -1, axis=0)
    history[-1] = positions.copy()

    # scatter update
    scat.set_offsets(positions)

    # trails update
    for i, idx in enumerate(trail_indices):
        h = history[:, idx, :]
        xs = h[:, 0]
        ys = h[:, 1]
        trail_lines[i].set_data(xs, ys)

    # MSD update
    sq_disp = np.sum(positions**2, axis=1)
    msd.append(np.mean(sq_disp))
    t_arr = np.arange(len(msd))
    line_msd.set_data(t_arr, msd)

    # fit to MSD(t)
    if len(msd) > fit_skip_initial + 1:
        fit_x = t_arr[fit_skip_initial:]
        fit_y = np.array(msd[fit_skip_initial:])
        fit_res = linear_fit_with_r2(fit_x, fit_y)
        if fit_res is not None:
            slope, intercept, r2 = fit_res
            fit_y_line = slope * fit_x + intercept
            line_fit.set_data(fit_x, fit_y_line)
            D_est = slope / 4.0
            txt.set_text(f"Theoretical D = {D_theory:.4f}\n"
                         f"Estimated D = {D_est:.4f}\n"
                         f"R² = {r2:.4f}")

            # y-limit adjust
            if fit_y_line.max() > ax_msd.get_ylim()[1] * 0.9:
                ax_msd.set_ylim(0, fit_y_line.max() * 1.4)

    return [scat, *trail_lines, line_msd, line_fit, txt]


ani = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=True, repeat=False)
plt.show()


# after animation: print final comparison (2D only)
if len(msd) > fit_skip_initial + 1:
    fit_x = np.arange(fit_skip_initial, len(msd))
    fit_y = np.array(msd[fit_skip_initial:])
    slope, intercept, r2 = linear_fit_with_r2(fit_x, fit_y)
    D_final = slope / 4.0
    print(f"Theoretical D = {D_theory:.6f}")
    print(f"Final Estimated D = {D_final:.6f}")
    print(f"Slope = {slope:.6f}")
    print(f"Fit R² = {r2:.6f}")
else:
    print("Not enough data to fit D.")

# --- Plot density of position with respect to distance from origin (2D) ---
distances = np.linalg.norm(positions, axis=1)
plt.figure(figsize=(6,4))
counts, bins, _ = plt.hist(distances, bins=40, density=True, alpha=0.7, color='tab:blue', edgecolor='k')
plt.xlabel('Distance from origin')
plt.ylabel('Density')
plt.title('Density of final position vs distance from origin')
plt.grid(True)
plt.show()
