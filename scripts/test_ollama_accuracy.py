"""Full Ollama LLM accuracy test with automatic graph generation.

Runs the LLM agent on the easy task using ALL LLM calls (no heuristic),
saves detailed results to output/, and generates performance graphs.
"""
import os, sys, time, json
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force Ollama as primary
os.environ["API_BASE_URL"] = "http://localhost:11434/v1"
os.environ["API_KEY"] = "ollama"
os.environ["MODEL_NAME"] = "qwen3.5:cloud"

from server.firewall_environment import FirewallEnvironment, ACTIONS
from server.graders import grade_stats
from inference import InferenceAgent

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run_task(task="easy", max_llm_calls=999999):
    # Temporarily override the environment config for testing so we get a realistic 
    # mix of malicious and benign traffic instead of mostly benign ALLOWs.
    from server.firewall_environment import TASK_CONFIGS
    TASK_CONFIGS[task]["benign_ratio"] = 0.50
    TASK_CONFIGS[task]["threat_probability"] = 0.40
    TASK_CONFIGS[task]["false_flag_prob"] = 0.20  # Make benign traffic look slightly suspicious

    # Use a random seed so every test run is different and unpredictable
    import random
    env = FirewallEnvironment(seed=random.randint(1, 10000))
    agent = InferenceAgent()
    state = env.reset(task=task)

    done = False
    steps = 0
    llm_calls = 0
    llm_actions = []
    cumulative_correct = []
    cumulative_scores = []
    t0 = time.time()

    while not done:
        sid = state.get("focus_session_id")
        action = 0

        if sid:
            session_data = env.evaluate_session(sid)
            threat_intel = env.get_threat_intelligence()
            is_malicious = env.pending_sessions.get(
                sid, env.inspected_sessions.get(sid, {})
            ).get("metadata", {}).get("malicious", False)

            if llm_calls < max_llm_calls:
                t1 = time.time()
                action = agent.get_action(session_data, threat_intel)
                dt = time.time() - t1
                llm_calls += 1

                is_correct = (
                    (is_malicious and action in [1, 2, 3, 4, 5])
                    or (not is_malicious and action == 0)
                )

                llm_actions.append({
                    "step": steps,
                    "call": llm_calls,
                    "action": action,
                    "action_name": ACTIONS[action],
                    "malicious": is_malicious,
                    "correct": is_correct,
                    "time": round(dt, 1),
                })

                # Track cumulative accuracy
                correct_so_far = sum(1 for a in llm_actions if a["correct"])
                cumulative_correct.append(correct_so_far / len(llm_actions))

                print("  LLM #{:3d}: {:12s}  mal={:<5s}  correct={:<5s}  acc={:.1%}  ({:.1f}s)".format(
                    llm_calls, ACTIONS[action], str(is_malicious), str(is_correct),
                    cumulative_correct[-1], dt))
            else:
                action = agent._heuristic_action(session_data, threat_intel)

        result = env.step_single(action)
        done = bool(result["done"])
        state = result["state"]
        steps += 1

    elapsed = time.time() - t0
    stats = env.get_network_stats()
    grade = grade_stats(task, stats)

    return {
        "task": task,
        "score": grade["score"],
        "passed": grade["passed"],
        "threshold": grade["threshold"],
        "detection_rate": stats["detection_rate"],
        "false_positive_rate": stats["false_positive_rate"],
        "efficiency": stats["efficiency"],
        "steps": steps,
        "llm_calls": llm_calls,
        "llm_actions": llm_actions,
        "cumulative_accuracy": cumulative_correct,
        "elapsed": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


def generate_llm_graphs(result, out_dir):
    """Generate individual performance graphs from LLM test results."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    actions_data = result["llm_actions"]
    if not actions_data:
        print("  [GRAPHS] No LLM actions to graph.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calls = [a["call"] for a in actions_data]
    times = [a["time"] for a in actions_data]
    correct = [a["correct"] for a in actions_data]
    cum_acc = result["cumulative_accuracy"]
    is_mal = [a["malicious"] for a in actions_data]
    action_ids = [a["action"] for a in actions_data]

    w = min(10, len(calls) // 3 + 1)

    saved = []
    plt.rcParams.update({
        'figure.facecolor': '#FAFAFA', 'axes.facecolor': '#FFFFFF',
        'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11,
    })

    # ── Graph 1: Cumulative Accuracy ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(calls, cum_acc, color='#2ECC71', linewidth=2.5, label='Cumulative Accuracy')
    roll_acc = pd.Series([1 if c else 0 for c in correct]).rolling(
        window=w, min_periods=1).mean().tolist()
    ax.plot(calls, roll_acc, color='#27AE60', linewidth=2, linestyle='--',
            label=f'Rolling Accuracy (w={w})')
    ax.axhline(y=0.75, color='orange', linestyle=':', alpha=0.6, label='75% target')
    ax.set_xlabel('LLM Call #', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    final_acc = cum_acc[-1] if cum_acc else 0
    ax.set_title(f'LLM Decision Accuracy Over Time\n'
                 f'Final: {final_acc:.1%} correct out of {len(calls)} calls',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = out_dir / "llm_01_accuracy.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ── Graph 2: Response Time ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ['#E74C3C' if not c else '#2ECC71' for c in correct]
    ax.bar(calls, times, color=colors_bar, alpha=0.6, width=1.0)
    avg_time = np.mean(times)
    ax.axhline(y=avg_time, color='#3498DB', linestyle='--', linewidth=2,
               label=f'Average ({avg_time:.1f}s)')
    ax.set_xlabel('LLM Call #', fontweight='bold')
    ax.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax.set_title(f'LLM Response Time per Call\n'
                 f'Green = correct, Red = incorrect | Avg: {avg_time:.1f}s',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = out_dir / "llm_02_response_time.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ── Graph 3: Action Distribution ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Overall action distribution
    action_names = [a["action_name"] for a in actions_data]
    unique_actions = sorted(set(action_names))
    counts = [action_names.count(a) for a in unique_actions]
    bars = ax1.barh(unique_actions, counts, color='#3498DB', alpha=0.7)
    ax1.set_xlabel('Count', fontweight='bold')
    ax1.set_title('Action Distribution (All Calls)', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontweight='bold')

    # Accuracy by action type
    action_acc = {}
    for a in actions_data:
        name = a["action_name"]
        if name not in action_acc:
            action_acc[name] = {"correct": 0, "total": 0}
        action_acc[name]["total"] += 1
        if a["correct"]:
            action_acc[name]["correct"] += 1

    names = sorted(action_acc.keys())
    accs = [action_acc[n]["correct"] / action_acc[n]["total"] for n in names]
    bar_colors = ['#2ECC71' if a >= 0.7 else '#E67E22' if a >= 0.5 else '#E74C3C' for a in accs]
    bars2 = ax2.barh(names, accs, color=bar_colors, alpha=0.7)
    ax2.set_xlabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy by Action Type', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.1)
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.0%}', va='center', fontweight='bold')

    plt.tight_layout()
    p = out_dir / "llm_03_action_distribution.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    # ── Graph 4: Confusion Matrix (Malicious vs Benign) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    tp = sum(1 for a in actions_data if a["malicious"] and a["action"] != 0)
    fn = sum(1 for a in actions_data if a["malicious"] and a["action"] == 0)
    fp = sum(1 for a in actions_data if not a["malicious"] and a["action"] != 0)
    tn = sum(1 for a in actions_data if not a["malicious"] and a["action"] == 0)

    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Benign', 'Predicted Malicious'])
    ax.set_yticklabels(['Actually Benign', 'Actually Malicious'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    color='white' if matrix[i, j] > matrix.max() * 0.6 else 'black')
    ax.set_title(f'LLM Confusion Matrix\n'
                 f'Precision: {tp/(tp+fp):.0%}  |  Recall: {tp/(tp+fn):.0%}' if (tp+fp) > 0 and (tp+fn) > 0 else 'LLM Confusion Matrix',
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    p = out_dir / "llm_04_confusion_matrix.png"
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    saved.append(p.name)

    print(f"  [GRAPHS] Saved {len(saved)} LLM graphs to {out_dir}/")
    for name in saved:
        print(f"    -> {name}")
    return saved


def main():
    print()
    print("=" * 60)
    print("  OLLAMA qwen3.5:cloud -- FULL ACCURACY TEST (easy task)")
    print("  All steps using LLM (no heuristic fallback)")
    print("=" * 60)
    print()

    # Connectivity check
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    try:
        r = client.chat.completions.create(
            model="qwen3.5:cloud",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=200, timeout=30
        )
        content = r.choices[0].message.content or ""
        extra = getattr(r.choices[0].message, "model_extra", {}) or {}
        reasoning = extra.get("reasoning", "")
        print(f"  Connected! (content={repr(content[:30])}, reasoning={len(reasoning)} chars)")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Proceeding with 0 LLM calls (Heuristic only) due to connection error.")
        result = run_task(max_llm_calls=0)
    else:
        print(f"\n  Running easy task fully using LLMs (no heuristic)...\n")
        result = run_task(max_llm_calls=999999)

    # Print results
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n  {'=' * 50}")
    print(f"  RESULT: {status}")
    print(f"  Score:     {result['score']:.4f}  (threshold: {result['threshold']})")
    print(f"  Detection: {result['detection_rate']:.3f}")
    print(f"  FP Rate:   {result['false_positive_rate']:.3f}")
    print(f"  Efficiency:{result['efficiency']:.3f}")
    print(f"  Steps:     {result['steps']}")
    print(f"  LLM calls: {result['llm_calls']}")
    print(f"  Time:      {result['elapsed']:.1f}s")

    llm = result["llm_actions"]
    if llm:
        correct = sum(1 for a in llm if a["correct"])
        print(f"\n  LLM Decision Quality:")
        print(f"    Correct: {correct}/{len(llm)} ({100*correct/len(llm):.0f}%)")
        action_dist = {}
        for a in llm:
            action_dist[a["action_name"]] = action_dist.get(a["action_name"], 0) + 1
        print(f"    Actions: {action_dist}")
        avg_time = sum(a["time"] for a in llm) / len(llm)
        print(f"    Avg time per call: {avg_time:.1f}s")

    # Save results JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "llm_test_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Generate graphs
    print()
    print("+" + "-" * 58 + "+")
    print("|  GENERATING LLM PERFORMANCE GRAPHS" + " " * 23 + "|")
    print("+" + "-" * 58 + "+")
    print()
    try:
        generate_llm_graphs(result, OUTPUT_DIR)
    except Exception as e:
        print(f"  [GRAPHS] Warning: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
