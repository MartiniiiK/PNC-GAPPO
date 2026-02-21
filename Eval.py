import argparse
import os
import re

import torch

from Config import Config
from Agent import PPOAgent
from Environment import PN_Environment
from PN_Generator import StructuredPNGenerator
from Utils import (
    ensure_dirs,
    evaluate_on_benchmark,
    evaluate_on_random_pns,
    summarize_eval_rows,
    save_eval_rows_csv,
)


def _find_latest_checkpoint() -> str:
    """优先按 model_epXXXX.pth 的 episode 数挑最大，否则按 mtime。"""
    model_dir = Config.TRAINING_MODELS_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    files = [f for f in os.listdir(model_dir) if f.lower().endswith(".pth")]
    if not files:
        raise FileNotFoundError(f"No .pth checkpoints found in: {model_dir}")

    def _ep_num(fname: str) -> int:
        m = re.search(r"model_ep(\d+)\.pth$", fname)
        return int(m.group(1)) if m else -1

    files.sort(
        key=lambda f: (_ep_num(f), os.path.getmtime(os.path.join(model_dir, f))),
        reverse=True,
    )
    return os.path.join(model_dir, files[0])


def main():
    parser = argparse.ArgumentParser(description="Greedy evaluation for PN_GAT_PPO (random PN + PNfiles benchmarks).")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint (.pth). Default: latest in Models.")

    parser.add_argument("--bench", action="store_true", help="Run greedy eval on Config.TRAIN_PN_LIST (PNfiles).")
    parser.add_argument("--no-bench", action="store_true", help="Disable benchmark eval.")

    parser.add_argument("--random-pns", type=int, default=int(getattr(Config, "FINAL_EVAL_RANDOM_PN_NUM", 20)))
    parser.add_argument("--seed", type=int, default=int(getattr(Config, "FINAL_EVAL_RANDOM_SEED", 0)))
    parser.add_argument("--export-random", action="store_true", help="Export generated random PNs to files for reproducibility.")
    parser.add_argument("--export-dir", type=str, default=str(getattr(Config, "FINAL_EVAL_RANDOM_EXPORT_DIR", os.path.join(Config.TRAINING_RESULT_DIR, "FinalEval_RandomPNs"))))

    parser.add_argument("--gen-stage", type=int, default=None, help="Optional curriculum stage index to configure generator (e.g., 0).")
    args = parser.parse_args()

    ensure_dirs()

    model_path = args.model or _find_latest_checkpoint()
    print(f">>> Loading checkpoint: {model_path}")

    agent = PPOAgent()
    state = torch.load(model_path, map_location=Config.DEVICE)
    agent.net.load_state_dict(state)

    # generator (optional stage config)
    gen = StructuredPNGenerator()
    if args.gen_stage is not None:
        idx = int(args.gen_stage)
        if idx < 0 or idx >= len(Config.CURRICULUM_SCHEDULE):
            raise ValueError(f"gen_stage out of range: {idx}, len={len(Config.CURRICULUM_SCHEDULE)}")
        gen.configure(Config.CURRICULUM_SCHEDULE[idx])
        print(f">>> Generator configured to curriculum stage idx={idx} name={Config.CURRICULUM_SCHEDULE[idx].get('name')}")

    run_bench = bool(args.bench) or (not bool(args.no_bench))
    stem = os.path.splitext(os.path.basename(model_path))[0]

    if run_bench:
        print("\n>>> Greedy Eval on PNfiles Benchmarks (Config.TRAIN_PN_LIST)")
        val_envs = [PN_Environment(pn_name=name) for name in Config.TRAIN_PN_LIST]
        bench_results = evaluate_on_benchmark(agent, val_envs)
        bench_rows = []
        for name, res in bench_results.items():
            status = str(res.get("status", ""))
            bench_rows.append(
                {
                    "pn_name": name,
                    "status": status,
                    "success": 1 if status == "SUCCESS" else 0,
                    "deadlock": 1 if status == "DEADLOCK" else 0,
                    "timeout": 1 if status == "TIMEOUT" else 0,
                    "steps": int(res.get("steps", 0)),
                    "makespan": res.get("makespan", None),
                    "reward": float(res.get("reward", 0.0)),
                }
            )
        bench_sum = summarize_eval_rows(bench_rows)
        print(
            f">>> Benchmarks Summary: n={bench_sum['n']} | "
            f"succ={bench_sum['success_rate']:.3f} dead={bench_sum['deadlock_rate']:.3f} tout={bench_sum['timeout_rate']:.3f} | "
            f"avg_ms(success)={bench_sum['avg_makespan_success']}"
        )
        save_eval_rows_csv(bench_rows, os.path.join(Config.TRAINING_RESULT_DIR, f"greedy_eval_{stem}_benchmarks.csv"))

    if int(args.random_pns) > 0:
        print("\n>>> Greedy Eval on Random Generated PNs")
        export_dir = args.export_dir if bool(args.export_random) else None
        rand_rows = evaluate_on_random_pns(
            agent,
            gen,
            n_pns=int(args.random_pns),
            seed=int(args.seed),
            export_dir=export_dir,
        )
        rand_sum = summarize_eval_rows(rand_rows)
        print(
            f">>> RandomPN Summary: n={rand_sum['n']} | "
            f"succ={rand_sum['success_rate']:.3f} dead={rand_sum['deadlock_rate']:.3f} tout={rand_sum['timeout_rate']:.3f} | "
            f"avg_ms(success)={rand_sum['avg_makespan_success']}"
        )
        save_eval_rows_csv(rand_rows, os.path.join(Config.TRAINING_RESULT_DIR, f"greedy_eval_{stem}_random_pns.csv"))
        if export_dir:
            print(f">>> RandomPNs exported to: {export_dir}")


if __name__ == "__main__":
    main()

