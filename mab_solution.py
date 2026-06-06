"""
=============================================================================
BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI
WORK INTEGRATED LEARNING PROGRAMMES DIVISION
Deep Reinforcement Learning - Lab Assignment 1
Part #1: Adaptive Treatment Recommendation System using Multi-Armed Bandit

Group Number: 32
Authors     : Team 32
=============================================================================

Assignment Parameters for Group 32:
  G = 32
  K = (32 mod 3) + 5 = 2 + 5 = 7  (number of medicines / arms)
  Hidden success probabilities Pi = 0.4 + ((G + i) mod 6) * 0.07
    for i in {0, 1, 2, 3, 4, 5, 6}
"""

# ---------------------------------------------------------------------------
# Standard library and third-party imports
# ---------------------------------------------------------------------------
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import datetime

# ---------------------------------------------------------------------------
# Reproducibility: fix seeds using Group Number
# ---------------------------------------------------------------------------
G = 32                          # Group number
random.seed(G)
np.random.seed(G)

# ===========================================================================
# SECTION 1 – ENVIRONMENT / DATASET DESIGN
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.1  Compute environment constants
# ---------------------------------------------------------------------------
K = (G % 3) + 5                 # Number of medicines (arms)
N_PATIENTS = 1000               # Total patients in the simulation

# 1.2  Hidden success probability for each medicine arm
# Formula: Pi = 0.4 + ((G + i) mod 6) * 0.07
HIDDEN_PROBS = [
    0.4 + ((G + i) % 6) * 0.07
    for i in range(K)
]

# Identify the true best medicine (highest hidden probability)
TRUE_BEST_MEDICINE = int(np.argmax(HIDDEN_PROBS))

print("=" * 60)
print("PART #1 — MULTI-ARMED BANDIT: TREATMENT RECOMMENDATION")
print("=" * 60)
print(f"\nGroup Number (G)         : {G}")
print(f"Number of Medicines (K)  : {K}")
print("\nHidden Success Probabilities:")
for i, p in enumerate(HIDDEN_PROBS):
    marker = " ← TRUE BEST" if i == TRUE_BEST_MEDICINE else ""
    print(f"  Medicine {i}: P_{i} = {p:.4f}{marker}")


# ---------------------------------------------------------------------------
# 1.3  Helper – simulate a single patient interaction
# ---------------------------------------------------------------------------
def simulate_patient(patient_id: int, medicine: int) -> dict:
    """
    Simulate a single patient visit.

    Parameters
    ----------
    patient_id : int
        Sequential patient index (0–999).
    medicine : int
        Index of the medicine arm chosen by the algorithm.

    Returns
    -------
    dict  with keys: patient_id, severity_score, assigned_medicine,
                     clinical_outcome, utility_score
    """
    # Severity score: cycles 1–5 based on patient index
    severity = (patient_id % 5) + 1

    # Binary clinical outcome: 1 (recovered) with probability P_medicine
    clinical_outcome = int(np.random.random() < HIDDEN_PROBS[medicine])

    # Utility score accounts for severity penalty
    utility_score = clinical_outcome * (1 - severity / 10.0)

    return {
        "patient_id": patient_id,
        "severity_score": severity,
        "assigned_medicine": medicine,
        "clinical_outcome": clinical_outcome,
        "utility_score": round(utility_score, 4),
    }


# ---------------------------------------------------------------------------
# 1.4  Build the base dataset skeleton (patient_id + severity only)
#      Columns for algorithm fields are filled during each strategy run.
# ---------------------------------------------------------------------------
def build_base_dataset() -> pd.DataFrame:
    """
    Generate the base patient dataset with patient_id and severity_score.
    The algorithm-dependent columns are populated during each strategy run.

    Returns
    -------
    pd.DataFrame with columns: patient_id, severity_score
    """
    records = [
        {
            "patient_id": pid,
            "severity_score": (pid % 5) + 1,
        }
        for pid in range(N_PATIENTS)
    ]
    return pd.DataFrame(records)


base_df = build_base_dataset()
print("\n--- First 10 rows of base patient dataset ---")
print(base_df.head(10).to_string(index=False))


# ===========================================================================
# SECTION 2 – STRATEGY IMPLEMENTATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# TASK 2: Immediate Exploitation Strategy (Greedy after warm-up)
# ---------------------------------------------------------------------------

def strategy_greedy(n_patients: int = N_PATIENTS,
                    warm_up_per_arm: int = 10) -> dict:
    """
    Immediate Exploitation (Greedy) Strategy.

    Phase 1 – Warm-up:
        Each medicine is tested exactly `warm_up_per_arm` times to obtain
        initial success rate estimates.
    Phase 2 – Pure Exploitation:
        The medicine with the highest observed success rate is applied to
        ALL remaining patients (no further exploration).

    Parameters
    ----------
    n_patients       : int  – total number of patients
    warm_up_per_arm  : int  – number of initial tests per medicine

    Returns
    -------
    dict with keys: records (list[dict]), cumulative_rewards (list[float]),
                    best_medicine (int), total_reward (float)
    """
    # Reset seeds for fair comparison across strategies
    np.random.seed(G)

    # Bandit statistics: counts and total clinical outcomes per arm
    counts  = [0] * K
    successes = [0] * K

    records = []
    cumulative_rewards = []
    cumulative = 0.0
    patient_id = 0

    # ---- Phase 1: Warm-up — cycle through each arm sequentially ----
    warm_up_total = warm_up_per_arm * K
    arm_cycle = [arm for arm in range(K) for _ in range(warm_up_per_arm)]

    for arm in arm_cycle:
        if patient_id >= n_patients:
            break
        result = simulate_patient(patient_id, arm)
        counts[arm]   += 1
        successes[arm] += result["clinical_outcome"]
        cumulative     += result["utility_score"]
        cumulative_rewards.append(cumulative)
        records.append(result)
        patient_id += 1

    # Determine best arm after warm-up
    success_rates  = [successes[a] / counts[a] if counts[a] > 0 else 0
                      for a in range(K)]
    best_medicine  = int(np.argmax(success_rates))

    # ---- Phase 2: Pure exploitation — always use best_medicine ----
    while patient_id < n_patients:
        result = simulate_patient(patient_id, best_medicine)
        counts[best_medicine]    += 1
        successes[best_medicine] += result["clinical_outcome"]
        cumulative += result["utility_score"]
        cumulative_rewards.append(cumulative)
        records.append(result)
        patient_id += 1

    print(f"\n[Greedy] Warm-up complete. Best medicine identified: "
          f"Medicine {best_medicine} (rate={success_rates[best_medicine]:.4f})")
    print(f"[Greedy] Total cumulative reward: {cumulative:.4f}")

    return {
        "records": records,
        "cumulative_rewards": cumulative_rewards,
        "best_medicine": best_medicine,
        "total_reward": cumulative,
    }


# ---------------------------------------------------------------------------
# TASK 3: Epsilon-Greedy (Controlled Clinical Trial) Strategy
# ---------------------------------------------------------------------------

def strategy_epsilon_greedy(n_patients: int = N_PATIENTS,
                             epsilon: float = 0.10) -> dict:
    """
    Epsilon-Greedy Strategy (Controlled Clinical Trial).

    With probability `epsilon` the algorithm explores a random medicine
    (uniform over all K arms); otherwise it exploits the arm with the
    highest empirical success rate so far.

    Parameters
    ----------
    n_patients : int   – total number of patients
    epsilon    : float – exploration probability (e.g. 0.10 = 10%)

    Returns
    -------
    dict with keys: records, cumulative_rewards, total_reward
    """
    np.random.seed(G)
    random.seed(G)

    counts    = [0] * K
    successes = [0.0] * K

    records = []
    cumulative_rewards = []
    cumulative = 0.0

    for patient_id in range(n_patients):
        # Epsilon-greedy arm selection
        if random.random() < epsilon:
            # Explore: pick any medicine uniformly at random
            chosen_arm = random.randint(0, K - 1)
        else:
            if all(c == 0 for c in counts):
                chosen_arm = random.randint(0, K - 1)
            else:
                # Exploit: choose arm with best empirical success rate
                rates = [successes[a] / counts[a] if counts[a] > 0 else 0
                         for a in range(K)]
                chosen_arm = int(np.argmax(rates))

        result = simulate_patient(patient_id, chosen_arm)
        counts[chosen_arm]    += 1
        successes[chosen_arm] += result["clinical_outcome"]
        cumulative += result["utility_score"]
        cumulative_rewards.append(cumulative)
        records.append(result)

    print(f"\n[Epsilon-Greedy ε={epsilon}] Total cumulative reward: "
          f"{cumulative:.4f}")
    return {
        "records": records,
        "cumulative_rewards": cumulative_rewards,
        "total_reward": cumulative,
    }


# ---------------------------------------------------------------------------
# TASK 4: UCB1 (Confidence-Based) Strategy
# ---------------------------------------------------------------------------

def strategy_ucb1(n_patients: int = N_PATIENTS) -> dict:
    """
    UCB1 (Upper Confidence Bound) Strategy.

    Each arm is initially pulled once (forced exploration). Afterwards,
    the arm with the highest UCB score is selected:

        UCB(a) = (successes[a] / counts[a])
                 + sqrt(2 * ln(t) / counts[a])

    where t is the current time step. Rarely-pulled arms get a higher
    confidence bonus, naturally balancing exploration and exploitation.

    Parameters
    ----------
    n_patients : int – total number of patients

    Returns
    -------
    dict with keys: records, cumulative_rewards, total_reward
    """
    np.random.seed(G)

    counts    = [0] * K
    successes = [0.0] * K

    records = []
    cumulative_rewards = []
    cumulative = 0.0

    for patient_id in range(n_patients):
        t = patient_id + 1  # 1-indexed time step

        if patient_id < K:
            # Force each arm to be tried at least once
            chosen_arm = patient_id
        else:
            # Compute UCB score for every arm
            ucb_scores = [
                (successes[a] / counts[a]) + math.sqrt(2 * math.log(t) / counts[a])
                for a in range(K)
            ]
            chosen_arm = int(np.argmax(ucb_scores))

        result = simulate_patient(patient_id, chosen_arm)
        counts[chosen_arm]    += 1
        successes[chosen_arm] += result["clinical_outcome"]
        cumulative += result["utility_score"]
        cumulative_rewards.append(cumulative)
        records.append(result)

    print(f"\n[UCB1] Total cumulative reward: {cumulative:.4f}")
    return {
        "records": records,
        "cumulative_rewards": cumulative_rewards,
        "total_reward": cumulative,
    }


# ===========================================================================
# SECTION 3 – RUN ALL STRATEGIES
# ===========================================================================

print("\n" + "=" * 60)
print("RUNNING ALL STRATEGIES")
print("=" * 60)

greedy_result   = strategy_greedy()
eg_10_result    = strategy_epsilon_greedy(epsilon=0.10)
eg_01_result    = strategy_epsilon_greedy(epsilon=0.01)
eg_50_result    = strategy_epsilon_greedy(epsilon=0.50)
ucb1_result     = strategy_ucb1()

# ===========================================================================
# SECTION 4 – TASK 5: COMPARATIVE ANALYSIS & VISUALISATION
# ===========================================================================

def plot_cumulative_rewards(results: dict, filename: str = "mab_comparison.png"):
    """
    Plot Cumulative Reward vs. Number of Patients for all strategies.

    Parameters
    ----------
    results  : dict mapping strategy label → list of cumulative rewards
    filename : str – output PNG file name
    """
    plt.figure(figsize=(13, 7))

    # Style map: label → (color, linestyle, linewidth)
    styles = {
        "Greedy (exploit-only)"    : ("crimson",    "-",  2.2),
        "ε-Greedy (ε=0.10)"        : ("royalblue",  "-",  2.2),
        "ε-Greedy (ε=0.01)"        : ("forestgreen","-",  1.8),
        "ε-Greedy (ε=0.50)"        : ("darkorange", "--", 1.8),
        "UCB1 (confidence-based)"  : ("purple",     "-",  2.5),
    }

    for label, rewards in results.items():
        color, ls, lw = styles.get(label, ("grey", "-", 1.5))
        plt.plot(range(1, len(rewards) + 1), rewards,
                 label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

    plt.xlabel("Number of Patients", fontsize=13)
    plt.ylabel("Cumulative Reward (Utility Score)", fontsize=13)
    plt.title(
        "Multi-Armed Bandit — Cumulative Reward vs. Number of Patients\n"
        f"Group 32  |  K={K} Medicines  |  N=1000 Patients",
        fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"\n[Plot saved] → {filename}")


# Collect results for plotting
all_results = {
    "Greedy (exploit-only)"   : greedy_result["cumulative_rewards"],
    "ε-Greedy (ε=0.10)"       : eg_10_result["cumulative_rewards"],
    "ε-Greedy (ε=0.01)"       : eg_01_result["cumulative_rewards"],
    "ε-Greedy (ε=0.50)"       : eg_50_result["cumulative_rewards"],
    "UCB1 (confidence-based)" : ucb1_result["cumulative_rewards"],
}

plot_cumulative_rewards(all_results, "mab_comparison.png")

# ---------------------------------------------------------------------------
# Print first 10 rows of dataset for each strategy
# ---------------------------------------------------------------------------
def print_strategy_dataset(records: list, label: str, n: int = 10):
    """Display the first n rows of the dataset generated by a strategy."""
    df = pd.DataFrame(records)
    print(f"\n--- {label}: First {n} rows ---")
    print(df.head(n).to_string(index=False))


print_strategy_dataset(greedy_result["records"],  "Greedy Strategy")
print_strategy_dataset(eg_10_result["records"],   "ε-Greedy (ε=0.10)")
print_strategy_dataset(eg_01_result["records"],   "ε-Greedy (ε=0.01)")
print_strategy_dataset(eg_50_result["records"],   "ε-Greedy (ε=0.50)")
print_strategy_dataset(ucb1_result["records"],    "UCB1 Strategy")

# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARATIVE SUMMARY — FINAL CUMULATIVE REWARDS")
print("=" * 60)
summary = {
    "Strategy"         : ["Greedy", "ε=0.10", "ε=0.01", "ε=0.50", "UCB1"],
    "Total Reward"     : [
        greedy_result["total_reward"],
        eg_10_result["total_reward"],
        eg_01_result["total_reward"],
        eg_50_result["total_reward"],
        ucb1_result["total_reward"],
    ],
}
summary_df = pd.DataFrame(summary)
summary_df["Total Reward"] = summary_df["Total Reward"].round(4)
summary_df = summary_df.sort_values("Total Reward", ascending=False).reset_index(drop=True)
print(summary_df.to_string(index=False))

print("\n" + "=" * 60)
print("ANSWERS TO TASK 5 QUESTIONS")
print("=" * 60)
best_label  = summary_df.iloc[0]["Strategy"]
best_reward = summary_df.iloc[0]["Total Reward"]
print(f"""
Q1. Which strategy achieves the highest cumulative reward?
    → {best_label} with total reward = {best_reward:.4f}

Q2. Which strategy identifies the best medicine fastest?
    → UCB1 converges earliest because it uses the confidence bonus to rapidly
      eliminate under-performing arms. Greedy is fast post-warm-up but may
      lock onto a sub-optimal arm if warm-up outcomes were unlucky.

Q3. Which strategy shows the most stable performance over time?
    → ε-Greedy (ε=0.01) — very low exploration keeps the cumulative reward
      curve smooth, though it may miss the global optimum if the greedy arm
      is sub-optimal early on.

Q4. Which strategy is safest for real-world hospital deployment?
    → UCB1 is the recommended choice. It automatically balances exploration
      and exploitation without requiring manual tuning of an epsilon parameter.
      Its confidence bound shrinks as evidence accumulates, so the system
      naturally commits to the best-known treatment for the majority of patients
      while remaining open to discovering improvements — a critical property
      in a medical setting where both missed cures and unnecessary experiments
      carry real human cost.

Short Comparative Summary (3–5 sentences):
    The Greedy strategy achieves high rewards quickly after the warm-up phase
    but is vulnerable to locking onto a sub-optimal medicine if initial samples
    are misleading. Epsilon-Greedy with ε=0.10 strikes a reasonable balance,
    providing consistent exploration that allows the system to self-correct over
    time, while ε=0.01 converges fast but risks early lock-in, and ε=0.50
    wastes too many patients on random exploration. UCB1 consistently delivers
    competitive cumulative rewards by granting extra chances to under-observed
    medicines through a mathematically principled confidence bonus that
    diminishes as evidence grows, making it robust without any hyper-parameter
    tuning. For clinical deployment, UCB1 is preferred because it minimises
    regret in expectation while naturally adapting its exploration-exploitation
    balance to available evidence.
""")
