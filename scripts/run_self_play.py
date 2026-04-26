"""Run self-play training with adaptive curriculum.

Usage:
    python scripts/run_self_play.py                    # Default: 30 rounds
    python scripts/run_self_play.py --rounds 50        # Custom round count
    python scripts/run_self_play.py --output results.json  # Custom output path
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.self_play.arena import SelfPlayArena
from server.baseline.heuristic_agent import heuristic_policy


def main():
    parser = argparse.ArgumentParser(
        description="Run self-play training with adaptive curriculum"
    )
    parser.add_argument(
        "--rounds", type=int, default=30,
        help="Number of training rounds (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.78,
        help="Pass threshold for mastery gating (default: 0.78)"
    )
    parser.add_argument(
        "--mastery-window", type=int, default=3,
        help="Consecutive passes needed for difficulty advance (default: 3)"
    )
    parser.add_argument(
        "--output", type=str, default="self_play_results.json",
        help="Output JSON path (default: self_play_results.json)"
    )
    parser.add_argument(
        "--no-graphs", action="store_true",
        help="Skip graph generation after training"
    )
    args = parser.parse_args()

    print()
    print("+" + "=" * 70 + "+")
    print("|  AI FIREWALL — SELF-PLAY ADAPTIVE CURRICULUM TRAINING" + " " * 16 + "|")
    print("+" + "=" * 70 + "+")
    print()
    print("  Config:")
    print("    Rounds:         {}".format(args.rounds))
    print("    Seed:           {}".format(args.seed))
    print("    Pass threshold: {}".format(args.threshold))
    print("    Mastery window: {} consecutive passes".format(args.mastery_window))
    print("    Output:         {}".format(args.output))
    print("    Policy:         heuristic (8-rule baseline)")

    t0 = time.time()

    arena = SelfPlayArena(
        seed=args.seed,
        mastery_window=args.mastery_window,
        pass_threshold=args.threshold,
    )

    results = arena.train(
        policy=heuristic_policy,
        num_rounds=args.rounds,
        verbose=True,
    )

    # Save results
    output_path = Path(args.output)
    arena.save_history(output_path)

    total_time = time.time() - t0
    print("  Results saved to: {}".format(output_path.resolve()))
    print("  Total training time: {:.1f}s".format(total_time))

    # Final assessment
    if results:
        final_elo = results[-1].elo
        start_elo = results[0].elo - results[0].elo_delta
        growth = final_elo - start_elo
        pass_rate = sum(1 for r in results if r.passed) / len(results)

        print()
        if growth > 50:
            print("  Agent showed SIGNIFICANT skill growth ({:+.0f} Elo)".format(growth))
        elif growth > 0:
            print("  Agent showed MODERATE skill growth ({:+.0f} Elo)".format(growth))
        else:
            print("  Agent did NOT improve ({:+.0f} Elo) — policy may need updating".format(growth))

        if pass_rate > 0.8:
            print("  Pass rate {:.0%} — agent handles adaptive curriculum well".format(pass_rate))
        elif pass_rate > 0.5:
            print("  Pass rate {:.0%} — room for policy improvement".format(pass_rate))
        else:
            print("  Pass rate {:.0%} — agent struggles with generated challenges".format(pass_rate))

    # ── Generate performance graphs ──
    if not args.no_graphs:
        print()
        print("+" + "-" * 70 + "+")
        print("|  GENERATING PERFORMANCE GRAPHS" + " " * 39 + "|")
        print("+" + "-" * 70 + "+")
        print()
        try:
            from scripts.generate_performance_matrix import generate_graphs
            generate_graphs(
                input_json=str(output_path.resolve()),
                output_dir=str(Path("output").resolve()),
            )
        except Exception as e:
            print("  [GRAPHS] Warning: Could not generate graphs: {}".format(e))
    else:
        print("  Skipping graph generation (--no-graphs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

