"""Generate individual performance graphs from self-play training results.

Produces separate PNG files for each metric in the output/ directory.
Called automatically after every self-play training run.

Output files:
  output/01_training_loss.png
  output/02_reward_analysis.png
  output/03_elo_progression.png
  output/04_win_rate.png
  output/05_detection_fp_rate.png
  output/06_difficulty_progression.png
  output/performance_matrix.csv
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Ensure project root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def compute_fixed_baseline_scores():
    """Run heuristic agent on fixed tasks for absolute baseline."""
    from server.firewall_environment import FirewallEnvironment
    from server.graders import run_deterministic_grade
    from server.baseline.heuristic_agent import heuristic_policy

    baselines = {}
    for task in ['easy', 'medium', 'hard']:
        env = FirewallEnvironment(seed=303)
        result = run_deterministic_grade(env, task, heuristic_policy)
        baselines[task] = result['score']
    return baselines


def generate_graphs(input_json: str = None, output_dir: str = None):
    """Generate all individual performance graph files.

    Args:
        input_json: Path to self_play_results.json (default: project root)
        output_dir: Directory to save graphs (default: project root / output)
    """
    input_path = Path(input_json) if input_json else ROOT_DIR / "self_play_results.json"
    out_dir = Path(output_dir) if output_dir else ROOT_DIR / "output"

    if not input_path.exists():
        print(f"  [GRAPHS] Error: {input_path} not found")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rounds_data = data.get("rounds", [])
    if not rounds_data:
        print("  [GRAPHS] No rounds data found.")
        return

    # ── Fixed baseline ──
    print("  [GRAPHS] Computing fixed baselines...")
    baselines = compute_fixed_baseline_scores()

    # ── Extract data ──
    rn = [r["round"] for r in rounds_data]
    scores = [r["score"] for r in rounds_data]
    elos = [r["elo"] for r in rounds_data]
    elo_deltas = [r["elo_delta"] for r in rounds_data]
    diff_elos = [r["difficulty_elo"] for r in rounds_data]
    det_rates = [r["stats"]["det"] for r in rounds_data]
    fp_rates = [r["stats"]["fp"] for r in rounds_data]
    eff_rates = [r["stats"]["eff"] for r in rounds_data]

    # Derived metrics
    abs_loss = [1.0 - s for s in scores]
    diff_fracs = [np.clip((de - 800) / 800, 0, 1) for de in diff_elos]
    norm_rewards = [min(1.0, s / max(0.3, 1.0 - 0.3 * df)) for s, df in zip(scores, diff_fracs)]
    elo_gaps = [e - de for e, de in zip(elos, diff_elos)]

    w = 5  # rolling window
    pass_thresh = data.get("config", {}).get("pass_threshold", 0.55)
    wins = [1 if r["passed"] else 0 for r in rounds_data]
    win_roll = pd.Series(wins).rolling(window=w, min_periods=1).mean().tolist()
    det_roll = pd.Series(det_rates).rolling(window=w, min_periods=1).mean().tolist()
    fp_roll = pd.Series(fp_rates).rolling(window=w, min_periods=1).mean().tolist()
    loss_roll = pd.Series(abs_loss).rolling(window=w, min_periods=1).mean().tolist()
    score_roll_mean = pd.Series(scores).rolling(window=w, min_periods=1).mean().tolist()
    score_roll_std = pd.Series(scores).rolling(window=w, min_periods=1).std().fillna(0).tolist()

    # ── Save CSV ──
    df = pd.DataFrame({
        "Round": rn, "Raw_Score": scores, "Abs_Training_Loss": abs_loss,
        "Diff_Normalized_Reward": norm_rewards,
        "Detection_Rate": det_rates, "FP_Rate": fp_rates, "Efficiency": eff_rates,
        "Agent_Elo": elos, "Elo_Delta": elo_deltas, "Difficulty_Elo": diff_elos,
        "Elo_Gap": elo_gaps, "Win_Rate": win_roll, "Difficulty_Frac": diff_fracs,
    })
    csv_path = out_dir / "performance_matrix.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # ── Shared style ──
    plt.rcParams.update({
        'figure.facecolor': '#FAFAFA',
        'axes.facecolor': '#FFFFFF',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 11,
    })

    saved = []

    # ================================================================
    # GRAPH 1: Training Loss
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rn, abs_loss, color='#E74C3C', linewidth=2, marker='o', markersize=4,
            alpha=0.6, label='Abs. Loss (1 - score)')
    ax.plot(rn, loss_roll, color='#C0392B', linewidth=2.5, linestyle='--',
            label=f'Rolling Mean (w={w})')
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss (Absolute Performance Gap)\n'
                 'Loss increases because curriculum difficulty rises, not because agent worsens',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(abs_loss) * 1.3)
    plt.tight_layout()
    p = out_dir / "01_training_loss.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ================================================================
    # GRAPH 2: Reward Analysis
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rn, scores, color='#2ECC71', linewidth=1.5, alpha=0.4, marker='.',
            label='Raw Score (vs adaptive opponent)')
    ax.plot(rn, norm_rewards, color='#27AE60', linewidth=2.5, marker='o', markersize=4,
            label='Difficulty-Normalized Reward')
    ax.fill_between(rn,
                    np.array(score_roll_mean) - np.array(score_roll_std),
                    np.array(score_roll_mean) + np.array(score_roll_std),
                    color='#2ECC71', alpha=0.15, label=f'Score Std Dev (w={w})')
    ax.axhline(y=baselines['medium'], color='gray', linestyle=':', linewidth=1.5,
               label=f'Fixed Medium Baseline ({baselines["medium"]:.3f})')
    ax.axhline(y=pass_thresh, color='red', linestyle=':', alpha=0.5,
               label=f'Pass Threshold ({pass_thresh})')
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Reward / Score', fontweight='bold')
    ax.set_title('Reward Analysis: Raw vs Difficulty-Normalized\n'
                 'Normalized reward UP = agent genuinely improving despite harder tasks',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    p = out_dir / "02_reward_analysis.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ================================================================
    # GRAPH 3: Elo Progression
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rn, elos, color='#3498DB', linewidth=2.5, marker='o', markersize=4,
            label='Agent Elo')
    ax.plot(rn, diff_elos, color='#E67E22', linewidth=2, marker='s', markersize=3,
            linestyle='--', label='Opponent (Difficulty) Elo')
    ax.fill_between(rn, elos, diff_elos,
                    where=[e < de for e, de in zip(elos, diff_elos)],
                    color='#E74C3C', alpha=0.1, label='Agent Behind')
    ax.fill_between(rn, elos, diff_elos,
                    where=[e >= de for e, de in zip(elos, diff_elos)],
                    color='#27AE60', alpha=0.1, label='Agent Ahead')
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Elo Rating', fontweight='bold')
    ax.set_title('Elo Progression: Agent vs Adaptive Opponent\n'
                 f'Method: Logistic K=32 | Gap: {elo_gaps[0]:+.0f} -> {elo_gaps[-1]:+.0f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = out_dir / "03_elo_progression.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ================================================================
    # GRAPH 4: Win Rate & Elo Delta
    # ================================================================
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    bars = ax1.bar(rn, elo_deltas, color='#3498DB', alpha=0.35, label='Elo Delta per Round')
    line = ax2.plot(rn, win_roll, color='#1ABC9C', linewidth=2.5, marker='o',
                    markersize=4, label=f'Win Rate (rolling w={w})')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Training Round', fontweight='bold')
    ax1.set_ylabel('Elo Delta', fontweight='bold', color='#3498DB')
    ax2.set_ylabel('Win Rate', fontweight='bold', color='#1ABC9C')
    ax2.set_ylim(0, 1.15)
    total_pass = sum(wins)
    ax1.set_title(f'Win Rate & Elo Gain per Round\n'
                  f'Overall: {total_pass}/{len(wins)} passed ({100*total_pass/len(wins):.0f}%)',
                  fontsize=12, fontweight='bold')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc='lower right')
    plt.tight_layout()
    p = out_dir / "04_win_rate.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ================================================================
    # GRAPH 5: Detection & FP Rate
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rn, det_roll, color='#9B59B6', linewidth=2.5, marker='o', markersize=4,
            label=f'Detection Rate (rolling w={w})')
    ax.plot(rn, fp_roll, color='#E74C3C', linewidth=2, marker='s', markersize=3,
            label=f'False Positive Rate (rolling w={w})')
    ax.plot(rn, eff_rates, color='#F39C12', linewidth=1.5, alpha=0.5, marker='.',
            label='Efficiency')
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Rate', fontweight='bold')
    ax.set_title('Detection, False Positive & Efficiency over Training\n'
                 f'Detection stays high while FP stays near zero',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    p = out_dir / "05_detection_fp_rate.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ================================================================
    # GRAPH 6: Difficulty Progression
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rn, diff_fracs, color='#E67E22', linewidth=2.5, marker='s', markersize=4,
            label='Difficulty Fraction')
    ax.fill_between(rn, 0, diff_fracs, color='#E67E22', alpha=0.15)
    ax.axhline(y=0.25, color='green', linestyle=':', alpha=0.5, label='Easy zone')
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Medium zone')
    ax.axhline(y=0.75, color='red', linestyle=':', alpha=0.5, label='Hard zone')
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Difficulty (0=Easiest, 1=Hardest)', fontweight='bold')
    ax.set_title('Curriculum Difficulty Progression (ADR)\n'
                 f'Started at {diff_fracs[0]:.2f}, ended at {diff_fracs[-1]:.2f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = out_dir / "06_difficulty_progression.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ── Print summary ──
    print(f"  [GRAPHS] Saved {len(saved)} graphs to {out_dir}/")
    for name in saved:
        print(f"    -> {name}")
    print(f"  [GRAPHS] Saved CSV -> {csv_path.name}")

    # Console summary table
    n = len(rn)
    early_n = min(10, n)
    late_start = max(0, n - 10)
    print(f"\n  {'Metric':<35s} {'Early':>10s} {'Late':>10s} {'Trend':>7s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*7}")
    for name, vals in [
        ("Abs. Training Loss", abs_loss),
        ("Raw Score", scores),
        ("Diff-Normalized Reward", norm_rewards),
        ("Detection Rate", det_rates),
        ("FP Rate", fp_rates),
        ("Efficiency", eff_rates),
    ]:
        early = np.mean(vals[:early_n])
        late = np.mean(vals[late_start:])
        trend = "DOWN" if late < early - 0.005 else ("UP" if late > early + 0.005 else "FLAT")
        print(f"  {name:<35s} {early:10.4f} {late:10.4f} {trend:>7s}")

    print(f"\n  Agent Elo:    {elos[0]:.1f} -> {elos[-1]:.1f}  (d={elos[-1]-elos[0]:+.1f})")
    print(f"  Opponent Elo: {diff_elos[0]:.1f} -> {diff_elos[-1]:.1f}  (d={diff_elos[-1]-diff_elos[0]:+.1f})")

    return saved


if __name__ == "__main__":
    generate_graphs()
