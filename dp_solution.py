"""
=============================================================================
BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI
WORK INTEGRATED LEARNING PROGRAMMES DIVISION
Deep Reinforcement Learning - Lab Assignment 1
Part #2: Autonomous Drone Rescue Using Dynamic Programming

Group Number: 32
Authors     : Team 32
=============================================================================

Assignment Parameters for Group 32  (G = 32, ends in digit '2'):
  • Grid        : 5×5  (last digit 0–4)
  • Battery max : 10   (last digit even)
  • Wind prob   : 20%  (last digit 0–4)
  • Rescue tgts : 2
  • Charging stn: 1
  • Danger zones: 3
  • Blocked cells: 2

Grid Layout (0-indexed row, col):
  S  F  F  F  F
  F  D  F  R  F
  F  F  C  F  D
  F  R  F  F  F
  F  F  D  X  X

  S = Start (0,0)  |  F = Free  |  D = Danger  |  R = Rescue
  C = Charging     |  W = Wind  |  X = Blocked
"""

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import time
import itertools
import datetime

# ===========================================================================
# SECTION 1 — ENVIRONMENT DEFINITION
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.1  Grid & environment constants (Group 32)
# ---------------------------------------------------------------------------
GRID_ROWS    = 5
GRID_COLS    = 5
MAX_BATTERY  = 10          # even last digit → 10
WIND_PROB    = 0.20        # last digit 0–4 → 20%
MAX_STEPS    = 50          # 5×5 grid limit

# Reward structure (as given)
REWARD_RESCUE   =  20
REWARD_DANGER   = -10
REWARD_BATTERY  = -20
REWARD_CHARGE   =   5
REWARD_MOVE     =  -1

# ---------------------------------------------------------------------------
# 1.2  Fixed grid configuration
# ---------------------------------------------------------------------------
# Cell type map: (row, col) → symbol
CELL_MAP = {
    (0, 0): 'S',   # Start
    (1, 1): 'D',   # Danger zone 1
    (1, 3): 'R',   # Rescue target 1
    (2, 2): 'C',   # Charging station
    (2, 4): 'D',   # Danger zone 2
    (3, 1): 'R',   # Rescue target 2
    (4, 2): 'D',   # Danger zone 3
    (4, 3): 'X',   # Blocked cell 1
    (4, 4): 'X',   # Blocked cell 2
}

# Canonical sets derived from CELL_MAP
BLOCKED_CELLS  = {(r, c) for (r, c), t in CELL_MAP.items() if t == 'X'}
DANGER_CELLS   = {(r, c) for (r, c), t in CELL_MAP.items() if t == 'D'}
CHARGING_CELLS = {(r, c) for (r, c), t in CELL_MAP.items() if t == 'C'}
RESCUE_CELLS   = [(r, c) for (r, c), t in CELL_MAP.items() if t == 'R']  # ordered list
START_POS      = (0, 0)
N_RESCUE       = len(RESCUE_CELLS)   # 2


def cell_type(pos: tuple) -> str:
    """Return the base cell type symbol for a given (row, col) position."""
    return CELL_MAP.get(pos, 'F')


# ---------------------------------------------------------------------------
# 1.3  Action definitions
# ---------------------------------------------------------------------------
# Action index → (Δrow, Δcol) for movement; Hover = stay
ACTIONS = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
    4: ( 0,  0),   # Hover
}
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right", 4: "Hover"}
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "H"}
N_ACTIONS    = len(ACTIONS)


def is_valid_cell(row: int, col: int) -> bool:
    """Check whether (row, col) is inside the grid and not a blocked cell."""
    if row < 0 or row >= GRID_ROWS or col < 0 or col >= GRID_COLS:
        return False
    if (row, col) in BLOCKED_CELLS:
        return False
    return True


def get_valid_actions(pos: tuple, battery: int) -> list:
    """
    Return list of valid action indices from a given (pos, battery) state.
    All 5 actions are always considered; movement into blocked/out-of-bounds
    cells is handled by staying in place (still costs 1 battery).
    If battery == 0 the episode has terminated — no actions available.
    """
    if battery == 0:
        return []
    return list(range(N_ACTIONS))


# ===========================================================================
# SECTION 2 — STATE SPACE & TRANSITION MODEL
# ===========================================================================

# ---------------------------------------------------------------------------
# 2.1  State representation
# ---------------------------------------------------------------------------
# State = (row, col, battery, rescue_mask)
# rescue_mask: integer bitmask over N_RESCUE targets
#   bit i = 0 → target i still needs rescue
#   bit i = 1 → target i already rescued
# Total states = GRID_ROWS * GRID_COLS * (MAX_BATTERY+1) * 2^N_RESCUE
TOTAL_STATES_ESTIMATE = (
    GRID_ROWS * GRID_COLS * (MAX_BATTERY + 1) * (2 ** N_RESCUE)
)
print("=" * 65)
print("PART #2 — DRONE RESCUE: DYNAMIC PROGRAMMING SOLUTION")
print("=" * 65)
print(f"\nGroup 32 Environment Parameters:")
print(f"  Grid           : {GRID_ROWS}×{GRID_COLS}")
print(f"  Max Battery    : {MAX_BATTERY}")
print(f"  Wind Probability: {WIND_PROB*100:.0f}%")
print(f"  Rescue Targets : {N_RESCUE}  at {RESCUE_CELLS}")
print(f"  Charging Stn   : {list(CHARGING_CELLS)}")
print(f"  Danger Zones   : {sorted(DANGER_CELLS)}")
print(f"  Blocked Cells  : {sorted(BLOCKED_CELLS)}")
print(f"\nState space upper bound: "
      f"{GRID_ROWS}×{GRID_COLS}×{MAX_BATTERY+1}×2^{N_RESCUE} = {TOTAL_STATES_ESTIMATE} states")


# ---------------------------------------------------------------------------
# 2.2  State enumeration
# ---------------------------------------------------------------------------
def enumerate_states() -> list:
    """
    Enumerate ALL reachable states.

    Returns
    -------
    list of (row, col, battery, rescue_mask) tuples
    Non-reachable states (blocked cells, battery=0 initial) are excluded.
    """
    states = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in BLOCKED_CELLS:
                continue                    # blocked: never reachable
            for bat in range(0, MAX_BATTERY + 1):
                for mask in range(2 ** N_RESCUE):
                    states.append((r, c, bat, mask))
    return states


ALL_STATES = enumerate_states()
STATE_INDEX = {s: i for i, s in enumerate(ALL_STATES)}
print(f"Enumerated reachable states: {len(ALL_STATES)}")


# ---------------------------------------------------------------------------
# 2.3  Transition function
# ---------------------------------------------------------------------------
def get_transitions(state: tuple, action: int) -> list:
    """
    Compute the list of (probability, next_state, reward) tuples for a
    given (state, action) pair.

    Handles:
      - Battery depletion (terminal if battery drops to 0)
      - Charging station refill
      - Rescue target collection (mask update)
      - Danger zone penalty
      - Wind stochasticity (only movement actions, not hover)
      - Blocked-cell bouncing (agent stays in place)

    Parameters
    ----------
    state  : (row, col, battery, rescue_mask)
    action : int  in {0,1,2,3,4}

    Returns
    -------
    list of (prob: float, next_state: tuple, reward: float)
    """
    row, col, battery, mask = state

    # Terminal state: battery exhausted
    if battery == 0:
        return []

    # ----- Determine intended movement directions -----
    if action == 4:
        # Hover action: no stochastic wind effect
        intended_directions = [(1.0, (0, 0))]
    else:
        dr, dc = ACTIONS[action]
        # Wind zone: current cell is W → 20% chance of random direction
        # (No W cells in our fixed map, but logic is present for completeness)
        if cell_type((row, col)) == 'W':
            # 20% probability: random direction (uniform over 4 moves)
            rand_prob = WIND_PROB
            intended_directions = [(1.0 - rand_prob, (dr, dc))]
            for wind_a in range(4):  # Up/Down/Left/Right only
                wdr, wdc = ACTIONS[wind_a]
                intended_directions.append((rand_prob / 4, (wdr, wdc)))
        else:
            intended_directions = [(1.0, (dr, dc))]

    # ----- Aggregate outcomes over all possible directions -----
    outcome_map = {}   # next_state → (total_prob, reward)

    for prob_dir, (dr, dc) in intended_directions:
        # Compute tentative next position
        nr, nc = row + dr, col + dc

        # Bounce back if out-of-bounds or blocked
        if not is_valid_cell(nr, nc):
            nr, nc = row, col

        # Battery: each action costs 1 unit; hover on charger gains +2 net (+1 net)
        new_battery = battery - 1

        # --- Charging station entry ---
        charge_reward = 0
        if (nr, nc) in CHARGING_CELLS:
            new_battery = MAX_BATTERY  # full recharge on entry
            charge_reward = REWARD_CHARGE

        # Hover special case: if on charging station, battery actually goes up
        if action == 4 and (row, col) in CHARGING_CELLS:
            new_battery = min(MAX_BATTERY, battery + 2)  # net +2 (no cost)

        new_battery = max(0, min(new_battery, MAX_BATTERY))

        # --- Rescue target ---
        new_mask   = mask
        rescue_reward = 0
        for i, (rr, rc) in enumerate(RESCUE_CELLS):
            if (nr, nc) == (rr, rc) and not (mask >> i & 1):
                new_mask       = mask | (1 << i)   # mark as rescued
                rescue_reward += REWARD_RESCUE

        # --- Danger zone ---
        danger_reward = REWARD_DANGER if (nr, nc) in DANGER_CELLS else 0

        # --- Battery exhaustion ---
        bat_reward = 0
        if new_battery == 0:
            bat_reward = REWARD_BATTERY

        # Total reward for this transition
        total_reward = (REWARD_MOVE + charge_reward + rescue_reward
                        + danger_reward + bat_reward)

        next_state = (nr, nc, new_battery, new_mask)

        if next_state in outcome_map:
            ep, er = outcome_map[next_state]
            outcome_map[next_state] = (ep + prob_dir, er)
        else:
            outcome_map[next_state] = (prob_dir, total_reward)

    return [(p, ns, r) for ns, (p, r) in outcome_map.items()]


def is_terminal(state: tuple) -> bool:
    """
    Check if a state is terminal:
      - battery == 0, OR
      - all rescue targets collected (mask == 2^N_RESCUE - 1)
    """
    _, _, battery, mask = state
    return battery == 0 or mask == (2 ** N_RESCUE - 1)


# ===========================================================================
# SECTION 3 — VALUE ITERATION
# ===========================================================================

def value_iteration(gamma: float = 0.95,
                    theta: float = 1e-3,
                    max_iter: int = 10_000) -> tuple:
    """
    Value Iteration algorithm to compute the optimal value function V* and
    optimal policy π*.

    Update rule:
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ·V(s')]

    Parameters
    ----------
    gamma    : float – discount factor
    theta    : float – convergence threshold (stop when max |ΔV| < theta)
    max_iter : int   – hard iteration cap

    Returns
    -------
    (V, policy, iters, final_delta, elapsed_time)
      V       : dict  state → float   optimal value function
      policy  : dict  state → int     optimal action index
      iters   : int   number of sweeps until convergence
      delta   : float final max change across all states
      elapsed : float wall-clock seconds
    """
    # Initialise V to zero for all states
    V = {s: 0.0 for s in ALL_STATES}

    start_time = time.time()
    delta_history = []

    for iteration in range(1, max_iter + 1):
        delta = 0.0

        for state in ALL_STATES:
            if is_terminal(state):
                V[state] = 0.0   # terminal states have value 0
                continue

            v_old = V[state]
            best_v = -float('inf')

            for action in range(N_ACTIONS):
                transitions = get_transitions(state, action)
                if not transitions:
                    continue
                q_val = sum(
                    prob * (reward + gamma * V.get(ns, 0.0))
                    for prob, ns, reward in transitions
                )
                if q_val > best_v:
                    best_v = q_val

            if best_v == -float('inf'):
                best_v = 0.0

            V[state] = best_v
            delta = max(delta, abs(v_old - best_v))

        delta_history.append(delta)

        # Print progress every 50 iterations
        if iteration % 50 == 0 or iteration == 1:
            print(f"  Iteration {iteration:4d}  |  max ΔV = {delta:.6f}")

        if delta < theta:
            print(f"\n  ✓ Converged at iteration {iteration}  |  "
                  f"final ΔV = {delta:.8f}")
            break

    elapsed = time.time() - start_time

    # Extract greedy policy from converged V
    policy = {}
    for state in ALL_STATES:
        if is_terminal(state):
            policy[state] = 4   # hover (no-op) in terminal states
            continue

        best_action, best_v = 4, -float('inf')
        for action in range(N_ACTIONS):
            transitions = get_transitions(state, action)
            if not transitions:
                continue
            q_val = sum(
                prob * (reward + gamma * V.get(ns, 0.0))
                for prob, ns, reward in transitions
            )
            if q_val > best_v:
                best_v = q_val
                best_action = action

        policy[state] = best_action

    return V, policy, iteration, delta, elapsed, delta_history


print("\n" + "=" * 65)
print("RUNNING VALUE ITERATION (γ=0.95, θ=1e-3)")
print("=" * 65)
V_star, pi_star, n_iters, final_delta, elapsed, delta_hist = value_iteration()
print(f"\n  Runtime           : {elapsed:.4f} seconds")
print(f"  Convergence iters : {n_iters}")
print(f"  Final Δ (error)   : {final_delta:.8f}")


# ===========================================================================
# SECTION 4 — POLICY VISUALISATION
# ===========================================================================

def render_grid(state: tuple = None) -> str:
    """
    Render the grid as a text grid.
    If a state (row, col, battery, mask) is given, shows drone position.
    """
    grid_chars = []
    for r in range(GRID_ROWS):
        row_chars = []
        for c in range(GRID_COLS):
            sym = cell_type((r, c))
            if sym == 'S':
                sym = 'F'   # start is free otherwise
            if (r, c) in BLOCKED_CELLS:
                sym = 'X'
            row_chars.append(sym)
        grid_chars.append(row_chars)

    if state is not None:
        r, c, bat, mask = state
        grid_chars[r][c] = '🤖'

    lines = []
    for r, row in enumerate(grid_chars):
        lines.append("  " + " ".join(f"{s:2}" for s in row))
    return "\n".join(lines)


def visualise_policy(V: dict, policy: dict,
                     battery: int = None,
                     mask: int = None,
                     save_path: str = "dp_policy.png"):
    """
    Visualise the optimal policy and value function as a grid heatmap.

    For a fixed (battery, rescue_mask) slice the heatmap shows:
      - Cell colour: V*(s) value
      - Arrow / symbol: optimal action
      - Cell labels: special zone types

    Parameters
    ----------
    V         : dict state → float   value function
    policy    : dict state → int     optimal policy
    battery   : int  – battery level for the slice (default: MAX_BATTERY)
    mask      : int  – rescue mask for the slice (default: 0 = none rescued)
    save_path : str  – file path for saving the figure
    """
    if battery is None:
        battery = MAX_BATTERY
    if mask is None:
        mask = 0   # no rescues done yet

    val_grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pol_grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    bg_color = [['white' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in BLOCKED_CELLS:
                bg_color[r][c] = '#555555'
                pol_grid[r][c] = 'X'
                continue

            state = (r, c, battery, mask)
            val_grid[r][c] = V.get(state, 0.0)
            act = policy.get(state, 4)
            pol_grid[r][c] = ACTION_ARROWS[act]

            ct = cell_type((r, c))
            if ct == 'D':
                bg_color[r][c] = '#ff6666'
            elif ct == 'C':
                bg_color[r][c] = '#66ff66'
            elif ct == 'R':
                bg_color[r][c] = '#66b3ff'
            elif (r, c) == START_POS:
                bg_color[r][c] = '#ffff99'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Policy arrows ---
    ax = axes[0]
    ax.set_title(f"Optimal Policy  (battery={battery}, rescue_mask={mask:0{N_RESCUE}b})",
                 fontsize=12, fontweight='bold')

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            color = bg_color[r][c]
            rect = plt.Rectangle([c - 0.5, GRID_ROWS - r - 1.5], 1, 1,
                                  facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

            arrow = pol_grid[r][c]
            ct = cell_type((r, c))
            label = f"{ct}\n{arrow}" if (r, c) not in BLOCKED_CELLS else "X"
            ax.text(c, GRID_ROWS - r - 1, label,
                    ha='center', va='center', fontsize=14,
                    color='black' if color != '#555555' else 'white',
                    fontweight='bold')

    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_xticks(range(GRID_COLS)); ax.set_xticklabels([f'c{i}' for i in range(GRID_COLS)])
    ax.set_yticks(range(GRID_ROWS)); ax.set_yticklabels([f'r{GRID_ROWS-1-i}' for i in range(GRID_ROWS)])
    ax.set_xlabel("Column"); ax.set_ylabel("Row")

    # Legend
    patches = [
        mpatches.Patch(color='#ffff99', label='S = Start'),
        mpatches.Patch(color='#ff6666', label='D = Danger'),
        mpatches.Patch(color='#66ff66', label='C = Charging'),
        mpatches.Patch(color='#66b3ff', label='R = Rescue'),
        mpatches.Patch(color='#555555', label='X = Blocked'),
        mpatches.Patch(color='white',   label='F = Free'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)

    # --- Right: Value heatmap ---
    ax2 = axes[1]
    ax2.set_title(f"State Value Heatmap V*(s)  (battery={battery}, mask={mask:0{N_RESCUE}b})",
                  fontsize=12, fontweight='bold')

    # Flip vertically so (row=0) is at top
    plot_vals = np.flipud(val_grid)
    cmap = LinearSegmentedColormap.from_list('rv', ['#d73027', '#fee090', '#1a9850'])
    im = ax2.imshow(plot_vals, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax2, label='V*(s)')

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            v = val_grid[r][c]
            txt = f"{v:.1f}" if not np.isnan(v) else "X"
            ax2.text(c, GRID_ROWS - 1 - r, txt,
                     ha='center', va='center', fontsize=10,
                     color='white' if (r, c) in BLOCKED_CELLS else 'black',
                     fontweight='bold')

    ax2.set_xticks(range(GRID_COLS)); ax2.set_xticklabels([f'c{i}' for i in range(GRID_COLS)])
    ax2.set_yticks(range(GRID_ROWS)); ax2.set_yticklabels([f'r{i}' for i in range(GRID_ROWS)])
    ax2.set_xlabel("Column"); ax2.set_ylabel("Row")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n[Plot saved] → {save_path}")


# Visualise for full battery / no rescues done yet
visualise_policy(V_star, pi_star,
                 battery=MAX_BATTERY,
                 mask=0,
                 save_path="dp_policy.png")

# Also visualise after 1 rescue completed (mask=01 in binary)
visualise_policy(V_star, pi_star,
                 battery=MAX_BATTERY,
                 mask=1,
                 save_path="dp_policy_after_rescue1.png")


# ===========================================================================
# SECTION 5 — STATE-VALUE ANALYSIS (HEATMAP SLICE)
# ===========================================================================

def plot_value_heatmap_over_battery(V: dict,
                                    mask: int = 0,
                                    save_path: str = "dp_value_battery_analysis.png"):
    """
    Plot a grid of value heatmaps, one per battery level, fixing rescue_mask.
    This shows how V*(position) varies as battery depletes.

    Parameters
    ----------
    V         : dict state → float
    mask      : int  – rescue mask to hold fixed
    save_path : str  – output file path
    """
    battery_levels = [MAX_BATTERY, MAX_BATTERY // 2, 2, 1]
    fig, axes = plt.subplots(1, len(battery_levels),
                              figsize=(5 * len(battery_levels), 5))
    fig.suptitle(
        f"State-Value Analysis: V*(pos) across Battery Levels  "
        f"(rescue_mask={mask:0{N_RESCUE}b})",
        fontsize=13, fontweight='bold'
    )
    cmap = LinearSegmentedColormap.from_list('rv', ['#d73027', '#fee090', '#1a9850'])

    for ax, bat in zip(axes, battery_levels):
        val_grid = np.zeros((GRID_ROWS, GRID_COLS))
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                state = (r, c, bat, mask)
                val_grid[r][c] = V.get(state, 0.0)

        plot_vals = np.flipud(val_grid)
        im = ax.imshow(plot_vals, cmap=cmap, aspect='auto',
                       vmin=np.nanmin(val_grid), vmax=np.nanmax(val_grid))
        plt.colorbar(im, ax=ax)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                v = val_grid[r][c]
                ct = cell_type((r, c))
                label = f"{ct}\n{v:.1f}" if (r, c) not in BLOCKED_CELLS else "X"
                ax.text(c, GRID_ROWS - 1 - r, label,
                        ha='center', va='center', fontsize=8,
                        color='black', fontweight='bold')

        ax.set_title(f"Battery = {bat}", fontsize=11)
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n[Plot saved] → {save_path}")


plot_value_heatmap_over_battery(V_star, mask=0,
                                save_path="dp_value_analysis.png")


# ===========================================================================
# SECTION 6 — CONVERGENCE PLOT
# ===========================================================================

def plot_convergence(delta_history: list, save_path: str = "dp_convergence.png"):
    """
    Plot the convergence curve: max |ΔV| vs. iteration number.

    Parameters
    ----------
    delta_history : list[float] – max delta per iteration
    save_path     : str
    """
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(delta_history) + 1), delta_history,
             color='navy', linewidth=2)
    plt.axhline(y=1e-3, color='crimson', linestyle='--', linewidth=1.5,
                label='θ = 1e-3 (convergence threshold)')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Max |ΔV| (Bellman error)", fontsize=12)
    plt.title("Value Iteration Convergence — Group 32\n"
              f"Drone Rescue 5×5 Grid  |  "
              f"Converged at iteration {n_iters}",
              fontsize=13, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n[Plot saved] → {save_path}")


plot_convergence(delta_hist, save_path="dp_convergence.png")


# ===========================================================================
# SECTION 7 — SIMULATE OPTIMAL TRAJECTORY
# ===========================================================================

def simulate_trajectory(policy: dict,
                         start: tuple = START_POS,
                         start_battery: int = MAX_BATTERY,
                         verbose: bool = True) -> dict:
    """
    Simulate the drone following the optimal policy for one episode.

    Parameters
    ----------
    policy        : dict state → action
    start         : (row, col) starting position
    start_battery : int initial battery
    verbose       : bool whether to print each step

    Returns
    -------
    dict with total_reward, steps, path
    """
    row, col = start
    battery  = start_battery
    mask     = 0   # no rescues yet
    state    = (row, col, battery, mask)

    total_reward = 0.0
    path         = [state]
    steps        = 0

    if verbose:
        print("\n" + "=" * 55)
        print("SIMULATING OPTIMAL POLICY TRAJECTORY")
        print("=" * 55)
        print(f"Start: pos={start}, battery={battery}, mask={mask:0{N_RESCUE}b}")

    while not is_terminal(state) and steps < MAX_STEPS:
        action = policy.get(state, 4)
        transitions = get_transitions(state, action)

        if not transitions:
            break

        # Sample next state according to transition probabilities
        probs = [p for p, ns, r in transitions]
        idx   = np.random.choice(len(transitions), p=probs)
        _, next_state, reward = transitions[idx]

        total_reward += reward
        state = next_state
        path.append(state)
        steps += 1

        if verbose:
            r2, c2, bat2, msk2 = state
            print(f"  Step {steps:3d}: action={ACTION_NAMES[action]:<5} → "
                  f"pos=({r2},{c2}), battery={bat2:2d}, "
                  f"mask={msk2:0{N_RESCUE}b}, reward={reward:+.1f}, "
                  f"cumR={total_reward:+.1f}")

    if verbose:
        reason = ("All rescued!" if mask == (2**N_RESCUE-1)
                  else "Battery depleted" if state[2] == 0
                  else "Max steps reached")
        print(f"\n  Episode ended: {reason}")
        print(f"  Total reward: {total_reward:.2f}  |  Steps: {steps}")

    return {"total_reward": total_reward, "steps": steps, "path": path}


G_DP = 32
np.random.seed(G_DP)
traj = simulate_trajectory(pi_star)


# ===========================================================================
# SECTION 8 — SCALABILITY / CURSE OF DIMENSIONALITY DISCUSSION
# ===========================================================================

print("\n" + "=" * 65)
print("SCALABILITY DISCUSSION — CURSE OF DIMENSIONALITY")
print("=" * 65)

state_size_current = GRID_ROWS * GRID_COLS * (MAX_BATTERY + 1) * (2 ** N_RESCUE)
state_size_10x10   = 10 * 10 * (MAX_BATTERY + 1) * (2 ** N_RESCUE)
state_size_more_rt = GRID_ROWS * GRID_COLS * (MAX_BATTERY + 1) * (2 ** 5)
state_size_dynamic = GRID_ROWS * GRID_COLS * (MAX_BATTERY + 1) * (2 ** N_RESCUE) * 4

print(f"""
Current state space  ({GRID_ROWS}×{GRID_COLS}, bat={MAX_BATTERY}, {N_RESCUE} targets) :
  ≈ {state_size_current:,} states

If grid becomes 10×10 (same battery and targets):
  ≈ {state_size_10x10:,} states   (+{state_size_10x10/state_size_current:.1f}×)

If rescue targets increase from {N_RESCUE} to 5 (same grid/battery):
  ≈ {state_size_more_rt:,} states   (+{state_size_more_rt/state_size_current:.1f}×)

If weather conditions (4 states) are added to state:
  ≈ {state_size_dynamic:,} states   (+{state_size_dynamic/state_size_current:.1f}×)

─────────────────────────────────────────────────────────────────
WHY THE CURSE OF DIMENSIONALITY HURTS DP:
  1. Value Iteration sweeps through ALL states in every iteration.
     As the state space grows exponentially, each sweep becomes
     computationally and memory-prohibitive.
  2. Storing the value table V(s) requires memory proportional to
     |S|. At 10×10 with 6 targets and dynamic weather the table
     could require hundreds of millions of entries — infeasible.
  3. The transition model P(s'|s,a) for a tabular MDP must be
     pre-computed or queried for every (state, action) pair,
     further multiplying the compute burden.

HOW DEEP RL METHODS HELP:
  • Deep Q-Networks (DQN) and Actor-Critic (PPO, A3C) approximate
    V(s) or Q(s,a) using neural networks that generalise across
    similar states, eliminating the need to store a full lookup
    table.
  • Policy gradient methods learn directly in continuous or huge
    state spaces that would be computationally intractable for DP.
  • Model-free Deep RL can handle unknown or partially-observed
    transition dynamics — ideal for real disaster environments.
  • Experience replay and batched mini-updates make learning
    sample-efficient, whereas DP requires a complete environment
    model.

RELATION TO REAL-WORLD AUTONOMOUS DRONES:
  Real rescue drones face: high-resolution GPS coordinates
  (continuous position), exact battery voltage (continuous),
  variable number of targets, camera feeds, real-time wind
  data, and multi-agent coordination. The resulting state space
  is effectively infinite — making tabular DP completely
  impractical. Deep RL methods such as DQN, SAC, or PPO, often
  combined with domain randomisation and simulation-to-real
  transfer, are the state-of-the-art approach for real-world
  autonomous drone navigation and rescue.
""")
