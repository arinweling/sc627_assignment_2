#!/usr/bin/env python3
"""Week 1: Open-loop trajectory comparison (fixed dt in both methods).

Model:
    x_k = [p_x, p_y, v_x, v_y]^T
    u_k = [a_x, a_y]^T

Dynamics:
    p_{k+1} = p_k + dt * v_k
    v_{k+1} = v_k + dt * u_k

Comparison setup:
1) min_energy: minimize sum ||v_k||^2 * dt for a chosen horizon N_ENERGY.
1b) min_energy_u2: minimize sum ||u_k||^2 * dt for the same horizon.
2) min_time: dt is fixed too, so minimize time by finding the smallest feasible
   horizon N in [N_TIME_MIN, N_TIME_MAX], where total time is N * dt.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


# ------------------------------
# Problem setup (hardcoded)
# ------------------------------
DT_FIXED = 0.1
STATE_DIM = 4
CTRL_DIM = 2

# Fixed horizon used for the min-energy solve.
N_ENERGY = 50

# Search band for min-time solve (smallest feasible N wins).
N_TIME_MIN = 2
N_TIME_MAX = 60

START_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
GOAL_POS = np.array([8.0, 6.0], dtype=float)

# Rectangle defined by min/max corners.
OBSTACLE_X_MIN = 1.0
OBSTACLE_X_MAX = 8.2
OBSTACLE_Y_MIN = 2.0
OBSTACLE_Y_MAX = 4.4
OBSTACLE_MARGIN = 0.8

# Control limits.
A_MAX = 3e99

# Regularization used only to stabilize feasibility solves in min-time search.
W_TERM_VEL_AUX = 1.0
W_CTRL_AUX = 1e-4

# Smooth approximation settings for differentiable obstacle constraints.
ABS_EPS = 1e-6
SMOOTH_MAX_ALPHA = 25.0

# Number of interpolation samples checked on each segment for continuous
# obstacle avoidance (0 means waypoint-only constraints).
COLLISION_SUBSAMPLES = 8

# Extra conservatism to offset smooth max/abs approximation and solver tolerance.
COLLISION_BUFFER = 0.1


# ------------------------------
# Decision packing utilities
# ------------------------------
def unpack_decision(z: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Split flat decision vector into state and control trajectories."""
    n_x = (n_steps + 1) * STATE_DIM
    n_u = n_steps * CTRL_DIM
    x = z[:n_x].reshape((n_steps + 1, STATE_DIM))
    u = z[n_x : n_x + n_u].reshape((n_steps, CTRL_DIM))
    return x, u


def pack_decision(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Pack state and control trajectories into one flat decision vector."""
    return np.concatenate([x.ravel(), u.ravel()])


# ------------------------------
# Dynamics and objectives
# ------------------------------
def dynamics_step(xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
    """Forward-Euler step with fixed dt."""
    px, py, vx, vy = xk
    ax, ay = uk
    px_next = px + DT_FIXED * vx
    py_next = py + DT_FIXED * vy
    vx_next = vx + DT_FIXED * ax
    vy_next = vy + DT_FIXED * ay
    return np.array([px_next, py_next, vx_next, vy_next], dtype=float)


def objective_energy(z: np.ndarray, n_steps: int) -> float:
    """J_energy = sum_k ||v_k||^2 * dt."""
    x, _ = unpack_decision(z, n_steps=n_steps)
    speed_sq = x[:-1, 2] ** 2 + x[:-1, 3] ** 2
    return float(np.sum(speed_sq) * DT_FIXED)


def objective_energy_u2(z: np.ndarray, n_steps: int) -> float:
    """Alternative energy objective: J_energy_u2 = sum_k ||u_k||^2 * dt."""
    _, u = unpack_decision(z, n_steps=n_steps)
    control_sq = u[:, 0] ** 2 + u[:, 1] ** 2
    return float(np.sum(control_sq) * DT_FIXED)


def objective_aux_for_feasibility(z: np.ndarray, n_steps: int) -> float:
    """Small auxiliary objective for stable feasibility search.

    This is NOT the min-time criterion. Min-time criterion is smallest feasible N.
    """
    x, u = unpack_decision(z, n_steps=n_steps)
    ctrl_reg = W_CTRL_AUX * np.sum(u[:, 0] ** 2 + u[:, 1] ** 2) * DT_FIXED
    term_vel_reg = W_TERM_VEL_AUX * np.dot(x[-1, 2:], x[-1, 2:])
    return float(ctrl_reg + term_vel_reg)


# ------------------------------
# Constraints
# ------------------------------
def eq_constraints(z: np.ndarray, n_steps: int) -> np.ndarray:
    """Initial state, dynamics, and terminal position constraints."""
    x, u = unpack_decision(z, n_steps=n_steps)
    constraints = [x[0] - START_STATE]

    for k in range(n_steps):
        constraints.append(x[k + 1] - dynamics_step(x[k], u[k]))

    constraints.append(x[-1, :2] - GOAL_POS)
    return np.concatenate(constraints)


def smooth_abs(v: np.ndarray) -> np.ndarray:
    """Differentiable approximation of absolute value."""
    return np.sqrt(v * v + ABS_EPS)


def smooth_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Differentiable approximation of max(a, b)."""
    alpha = SMOOTH_MAX_ALPHA
    return np.log(np.exp(alpha * a) + np.exp(alpha * b)) / alpha


def obstacle_outside_measure(pos: np.ndarray) -> np.ndarray:
    """Signed outside measure wrt axis-aligned rectangle."""
    cx = 0.5 * (OBSTACLE_X_MIN + OBSTACLE_X_MAX)
    cy = 0.5 * (OBSTACLE_Y_MIN + OBSTACLE_Y_MAX)
    hx = 0.5 * (OBSTACLE_X_MAX - OBSTACLE_X_MIN)
    hy = 0.5 * (OBSTACLE_Y_MAX - OBSTACLE_Y_MIN)

    dx = pos[:, 0] - cx
    dy = pos[:, 1] - cy

    a = smooth_abs(dx) - hx
    b = smooth_abs(dy) - hy
    return smooth_max(a, b)


def obstacle_ineq_constraints(z: np.ndarray, n_steps: int) -> np.ndarray:
    """Keep waypoints and segment-interpolated points outside obstacle margin."""
    x, _ = unpack_decision(z, n_steps=n_steps)
    samples = [x[:, :2]]

    if COLLISION_SUBSAMPLES > 0:
        taus = np.linspace(0.0, 1.0, COLLISION_SUBSAMPLES + 2)[1:-1]
        for tau in taus:
            interp_pos = (1.0 - tau) * x[:-1, :2] + tau * x[1:, :2]
            samples.append(interp_pos)

    all_pos = np.vstack(samples)
    outside_measure = obstacle_outside_measure(all_pos)
    return outside_measure - (OBSTACLE_MARGIN + COLLISION_BUFFER)


def control_bounds(n_steps: int) -> Bounds:
    """Bounds for all decision variables: controls bounded, states free."""
    n_x = (n_steps + 1) * STATE_DIM
    n_u = n_steps * CTRL_DIM

    lower = np.full(n_x + n_u, -np.inf, dtype=float)
    upper = np.full(n_x + n_u, np.inf, dtype=float)

    lower[n_x:] = -A_MAX
    upper[n_x:] = A_MAX
    return Bounds(lower, upper)


# ------------------------------
# Initialization and solve
# ------------------------------
def _detour_waypoint() -> np.ndarray:
    """Pick a waypoint above the obstacle for robust warm start."""
    x_via = max(START_STATE[0] + 0.3, OBSTACLE_X_MIN - 0.4)
    y_via = OBSTACLE_Y_MAX + OBSTACLE_MARGIN + 0.9
    return np.array([x_via, y_via], dtype=float)


def initial_guess(n_steps: int) -> np.ndarray:
    """Build a feasible-ish warm start using a two-segment detour path."""
    x0 = np.zeros((n_steps + 1, STATE_DIM), dtype=float)
    u0 = np.zeros((n_steps, CTRL_DIM), dtype=float)

    via = _detour_waypoint()
    split = max(2, int(0.35 * n_steps))

    t1 = np.linspace(0.0, 1.0, split + 1)
    x0[: split + 1, 0] = START_STATE[0] + (via[0] - START_STATE[0]) * t1
    x0[: split + 1, 1] = START_STATE[1] + (via[1] - START_STATE[1]) * t1

    t2 = np.linspace(0.0, 1.0, n_steps - split + 1)
    x0[split:, 0] = via[0] + (GOAL_POS[0] - via[0]) * t2
    x0[split:, 1] = via[1] + (GOAL_POS[1] - via[1]) * t2

    x0[:-1, 2:] = (x0[1:, :2] - x0[:-1, :2]) / DT_FIXED
    x0[-1, 2:] = 0.0
    x0[0] = START_STATE

    return pack_decision(x0, u0)


def solve_for_horizon(mode: str, n_steps: int) -> tuple[np.ndarray, np.ndarray, object]:
    """Solve one optimization problem for a specific horizon."""
    if mode not in {"min_energy", "min_energy_u2", "feasibility_aux"}:
        raise ValueError("mode must be 'min_energy', 'min_energy_u2', or 'feasibility_aux'")

    z0 = initial_guess(n_steps=n_steps)

    if mode == "min_energy":
        objective = lambda z: objective_energy(z, n_steps=n_steps)
    elif mode == "min_energy_u2":
        objective = lambda z: objective_energy_u2(z, n_steps=n_steps)
    else:
        objective = lambda z: objective_aux_for_feasibility(z, n_steps=n_steps)

    constraints = [
        {"type": "eq", "fun": lambda z: eq_constraints(z, n_steps=n_steps)},
        {"type": "ineq", "fun": lambda z: obstacle_ineq_constraints(z, n_steps=n_steps)},
    ]

    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=control_bounds(n_steps=n_steps),
        constraints=constraints,
        options={"maxiter": 800, "ftol": 1e-7, "disp": False},
    )

    x_opt, u_opt = unpack_decision(result.x, n_steps=n_steps)
    return x_opt, u_opt, result


def solve_min_energy() -> tuple[np.ndarray, np.ndarray, int, object]:
    """Solve min-energy trajectory at fixed horizon N_ENERGY."""
    x_opt, u_opt, result = solve_for_horizon(mode="min_energy", n_steps=N_ENERGY)
    return x_opt, u_opt, N_ENERGY, result


def solve_min_energy_u2() -> tuple[np.ndarray, np.ndarray, int, object]:
    """Solve alternative min-energy trajectory using u^2 cost."""
    x_opt, u_opt, result = solve_for_horizon(mode="min_energy_u2", n_steps=N_ENERGY)
    return x_opt, u_opt, N_ENERGY, result


def solve_min_time() -> tuple[np.ndarray, np.ndarray, int, object]:
    """Solve min-time trajectory by searching smallest feasible N."""
    last_result = None

    for n_steps in range(N_TIME_MIN, N_TIME_MAX + 1):
        x_opt, u_opt, result = solve_for_horizon(mode="feasibility_aux", n_steps=n_steps)
        last_result = result
        if result.success:
            return x_opt, u_opt, n_steps, result

    raise RuntimeError(
        "No feasible trajectory found in min-time search range. "
        f"Last solver message: {last_result.message if last_result else 'N/A'}"
    )


# ------------------------------
# Metrics, reporting, and plotting
# ------------------------------
def trajectory_metrics(x: np.ndarray, u: np.ndarray, n_steps: int) -> dict[str, float]:
    """Compute comparable metrics for both trajectories."""
    velocity_energy = float(np.sum(x[:-1, 2] ** 2 + x[:-1, 3] ** 2) * DT_FIXED)
    control_energy = float(np.sum(u[:, 0] ** 2 + u[:, 1] ** 2) * DT_FIXED)
    total_time = float(n_steps * DT_FIXED)
    max_acc = float(np.max(np.linalg.norm(u, axis=1))) if u.size else 0.0

    pts = [x[:, :2]]
    dense_taus = np.linspace(0.0, 1.0, 42)[1:-1]
    for tau in dense_taus:
        pts.append((1.0 - tau) * x[:-1, :2] + tau * x[1:, :2])
    dense_pos = np.vstack(pts)
    min_clear = float(np.min(obstacle_outside_measure(dense_pos) - OBSTACLE_MARGIN))

    return {
        "velocity_energy": velocity_energy,
        "control_energy": control_energy,
        "time": total_time,
        "max_acc": max_acc,
        "min_clearance_vs_margin": min_clear,
    }


def print_mode_report(mode: str, x: np.ndarray, u: np.ndarray, n_steps: int, result: object) -> None:
    """Print solver status and key metrics for one optimization mode."""
    metrics = trajectory_metrics(x, u, n_steps=n_steps)
    print(f"\n=== {mode} ===")
    print("Optimization success:", result.success)
    print("Solver message:", result.message)
    print("Iterations:", result.nit)
    print("Objective value:", float(result.fun))
    print(f"N: {n_steps}")
    print(f"dt (fixed): {DT_FIXED:.5f} s")
    print(f"Total time (N*dt): {metrics['time']:.4f} s")
    print(f"Velocity-energy (sum ||v||^2 dt): {metrics['velocity_energy']:.4f}")
    print(f"Control-energy (sum ||u||^2 dt): {metrics['control_energy']:.4f}")
    print(f"Max ||u||: {metrics['max_acc']:.4f} (limit {A_MAX:.4f})")
    print(f"Min dense clearance vs margin: {metrics['min_clearance_vs_margin']:.4f}")


def plot_comparison(x_energy_v2: np.ndarray, x_energy_u2: np.ndarray, x_time: np.ndarray) -> None:
    """Plot all three trajectories on one graph."""
    fig, ax = plt.subplots(figsize=(9, 7))

    rect = Rectangle(
        (OBSTACLE_X_MIN, OBSTACLE_Y_MIN),
        OBSTACLE_X_MAX - OBSTACLE_X_MIN,
        OBSTACLE_Y_MAX - OBSTACLE_Y_MIN,
        facecolor="#d95f02",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.30,
        label="Obstacle",
    )
    ax.add_patch(rect)

    margin_rect = Rectangle(
        (OBSTACLE_X_MIN - OBSTACLE_MARGIN, OBSTACLE_Y_MIN - OBSTACLE_MARGIN),
        (OBSTACLE_X_MAX - OBSTACLE_X_MIN) + 2.0 * OBSTACLE_MARGIN,
        (OBSTACLE_Y_MAX - OBSTACLE_Y_MIN) + 2.0 * OBSTACLE_MARGIN,
        fill=False,
        edgecolor="#d95f02",
        linestyle="--",
        linewidth=1.1,
        label="Safety margin",
    )
    ax.add_patch(margin_rect)

    ax.plot(
        x_energy_v2[:, 0],
        x_energy_v2[:, 1],
        "-",
        linewidth=2.2,
        color="#1f77b4",
        label="Min-energy path (v^2 dt)",
    )
    ax.plot(
        x_energy_u2[:, 0],
        x_energy_u2[:, 1],
        "-",
        linewidth=2.2,
        color="#9467bd",
        label="Min-energy path (u^2 dt)",
    )
    ax.plot(
        x_time[:, 0],
        x_time[:, 1],
        "-",
        linewidth=2.2,
        color="#d62728",
        label="Min-time path",
    )

    ax.scatter(START_STATE[0], START_STATE[1], s=70, c="#222222", marker="s", label="Start")
    ax.scatter(GOAL_POS[0], GOAL_POS[1], s=90, c="#2ca02c", marker="*", label="Goal")

    ax.set_title("Open-loop comparison: all 3 paths (fixed dt)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    x_all = np.array([START_STATE[0], GOAL_POS[0], OBSTACLE_X_MIN - OBSTACLE_MARGIN, OBSTACLE_X_MAX + OBSTACLE_MARGIN])
    y_all = np.array([START_STATE[1], GOAL_POS[1], OBSTACLE_Y_MIN - OBSTACLE_MARGIN, OBSTACLE_Y_MAX + OBSTACLE_MARGIN])
    pad = 1.0
    ax.set_xlim(np.min(x_all) - pad, np.max(x_all) + pad)
    ax.set_ylim(np.min(y_all) - pad, np.max(y_all) + pad)

    plt.tight_layout()
    # plt.show()


def plot_energy_definition_comparison(x_energy_v2: np.ndarray, x_energy_u2: np.ndarray) -> None:
    """Plot two min-energy definitions: v^2 dt vs u^2 dt."""
    fig, ax = plt.subplots(figsize=(9, 7))

    rect = Rectangle(
        (OBSTACLE_X_MIN, OBSTACLE_Y_MIN),
        OBSTACLE_X_MAX - OBSTACLE_X_MIN,
        OBSTACLE_Y_MAX - OBSTACLE_Y_MIN,
        facecolor="#d95f02",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.30,
        label="Obstacle",
    )
    ax.add_patch(rect)

    margin_rect = Rectangle(
        (OBSTACLE_X_MIN - OBSTACLE_MARGIN, OBSTACLE_Y_MIN - OBSTACLE_MARGIN),
        (OBSTACLE_X_MAX - OBSTACLE_X_MIN) + 2.0 * OBSTACLE_MARGIN,
        (OBSTACLE_Y_MAX - OBSTACLE_Y_MIN) + 2.0 * OBSTACLE_MARGIN,
        fill=False,
        edgecolor="#d95f02",
        linestyle="--",
        linewidth=1.1,
        label="Safety margin",
    )
    ax.add_patch(margin_rect)

    ax.plot(
        x_energy_v2[:, 0],
        x_energy_v2[:, 1],
        "-",
        linewidth=2.2,
        color="#1f77b4",
        label="Min-energy path (v^2 dt)",
    )
    ax.plot(
        x_energy_u2[:, 0],
        x_energy_u2[:, 1],
        "-",
        linewidth=2.2,
        color="#9467bd",
        label="Min-energy path (u^2 dt)",
    )

    ax.scatter(START_STATE[0], START_STATE[1], s=70, c="#222222", marker="s", label="Start")
    ax.scatter(GOAL_POS[0], GOAL_POS[1], s=90, c="#2ca02c", marker="*", label="Goal")

    ax.set_title("Min-energy definition comparison: v^2 dt vs u^2 dt")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    x_all = np.array([START_STATE[0], GOAL_POS[0], OBSTACLE_X_MIN - OBSTACLE_MARGIN, OBSTACLE_X_MAX + OBSTACLE_MARGIN])
    y_all = np.array([START_STATE[1], GOAL_POS[1], OBSTACLE_Y_MIN - OBSTACLE_MARGIN, OBSTACLE_Y_MAX + OBSTACLE_MARGIN])
    pad = 1.0
    ax.set_xlim(np.min(x_all) - pad, np.max(x_all) + pad)
    ax.set_ylim(np.min(y_all) - pad, np.max(y_all) + pad)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Solve both objectives, print formulas/metrics, and plot comparison."""
    x_energy, u_energy, n_energy, res_energy = solve_min_energy()
    x_energy_u2, u_energy_u2, n_energy_u2, res_energy_u2 = solve_min_energy_u2()
    x_time, u_time, n_time, res_time = solve_min_time()

    print_mode_report("min_energy", x_energy, u_energy, n_energy, res_energy)
    print_mode_report("min_energy_u2", x_energy_u2, u_energy_u2, n_energy_u2, res_energy_u2)
    print_mode_report("min_time", x_time, u_time, n_time, res_time)

    if not res_energy.success or not res_energy_u2.success or not res_time.success:
        print("Warning: At least one optimization failed. Plotting available results anyway.")

    print("\nExact objective formulas used:")
    print("J_energy(v) = sum_k ||v_k||^2 * dt")
    print("J_energy(u) = sum_k ||u_k||^2 * dt")
    print("J_time   = N * dt, with dt fixed and N chosen as the smallest feasible horizon")

    plot_comparison(x_energy, x_energy_u2, x_time)
    out_path = Path(__file__).with_name("trajectories.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")
    # plot_energy_definition_comparison(x_energy, x_energy_u2)


if __name__ == "__main__":
    main()
