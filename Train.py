import numpy as np
import time
from Config import Config
from Environment import PN_Environment
from Agent import PPOAgent
from Utils import (
    ensure_dirs,
    evaluate_on_benchmark,
    evaluate_on_random_pns,
    save_training_log,
    save_benchmark_log,
    plot_training_status,
    plot_benchmark_result,
    summarize_eval_rows,
    save_eval_rows_csv,
    sci_str
)
import torch
import os
import random
from PN_Generator import RandomPNGenerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

def main():
    ensure_dirs()
    generator = RandomPNGenerator()
    pn_pool = None
    agent = PPOAgent()
    scheduler = CosineAnnealingLR(agent.optimizer, T_max=Config.EPISODES, eta_min=0)
    val_envs = [PN_Environment(pn_name=name) for name in Config.TRAIN_PN_LIST]
    train_history = {
        'episode': [],
        'reward': [],
        'success': [],
        'deadlock': [],
        'avg_value': [],
        'loss_total': [],
        'loss_policy': [],
        'loss_value': [],
        'loss_entropy': [],
        'approx_kl': [],
        'clip_frac': [],
        'explained_var': [],
        'entropy_coef': [],
    }
    network_mode = getattr(Config, "AC_NET_TYPE", "MLP")
    print(f"Starting PPO Training with {network_mode}")

    last_loss_dict = {
        'total': 0.0,
        'policy': 0.0,
        'value': 0.0,
        'entropy': 0.0,
        'approx_kl': 0.0,
        'clip_frac': 0.0,
        'explained_var': 0.0,
        'entropy_coef': float(getattr(Config, "ENTROPY_COEF", 0.0)),
    }

    best_ms_per_bench = {}

    train_mode = getattr(Config, "TRAIN_MODE", "random_gen")
    fixed_pn_pool = getattr(Config, "FIXED_PN_POOL", Config.TRAIN_PN_LIST[:3])
    single_pn_name = getattr(Config, "SINGLE_INSTANCE_PN", Config.TRAIN_PN_LIST[0])

    if train_mode == "single_instance":
        print(f"Train mode: single_instance - Training on single PN: {single_pn_name}")
        single_env = PN_Environment(pn_name=single_pn_name)
    elif train_mode == "fixed_pool":
        print(f"Train mode: fixed_pool - Selecting from fixed PN pool: {fixed_pn_pool}")
        fixed_envs = {name: PN_Environment(pn_name=name) for name in fixed_pn_pool}
    elif train_mode == "gen_pool":
        from PN_Generator import GeneratorPNPool
        pn_pool = GeneratorPNPool(generator)
        pn_pool.initialize()
        print("Train mode: gen_pool - Maintaining generator PN pool")
        print(pn_pool.describe())
    else:
        print("Train mode: random_gen - Random PN generation per episode")

    for episode in range(1, Config.EPISODES + 1):
        if train_mode == "single_instance":
            env = single_env
            pn_instance = env.pn
            pn_name = single_pn_name
        elif train_mode == "fixed_pool":
            pn_name = random.choice(list(fixed_envs.keys()))
            env = fixed_envs[pn_name]
            pn_instance = env.pn
        elif train_mode == "gen_pool":
            assert pn_pool is not None
            pn_instance, _ = pn_pool.sample(episode)
            env = PN_Environment(pn_instance=pn_instance)
            pn_name = getattr(pn_instance, "name", "GenPoolPN")
        else:
            pn_instance = generator.generate()
            env = PN_Environment(pn_instance=pn_instance)
            pn_name = getattr(pn_instance, "name", "RandPN")
        state, action_mask = env.reset()

        ep_reward = 0
        ep_values = []
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state, action_mask, is_training=True)
            next_state, next_action_mask, reward, done, info = env.step(action)

            agent.store(state, action_mask, action, reward, next_state, done, log_prob, value)

            ep_reward += reward
            ep_values.append(value)
            state = next_state
            action_mask = next_action_mask

        loss_dict = agent.learn()
        if loss_dict is not None:
            last_loss_dict = loss_dict

        avg_value = np.mean(ep_values) if ep_values else 0.0
        is_success = info.get('success', False)
        is_deadlock = info.get('deadlock', False)

        train_history['episode'].append(episode)
        train_history['reward'].append(ep_reward)
        train_history['success'].append(1 if is_success else 0)
        train_history['deadlock'].append(1 if is_deadlock else 0)
        train_history['avg_value'].append(avg_value)
        train_history['loss_total'].append(last_loss_dict['total'])
        train_history['loss_policy'].append(last_loss_dict['policy'])
        train_history['loss_value'].append(last_loss_dict['value'])
        train_history['loss_entropy'].append(last_loss_dict['entropy'])
        train_history['approx_kl'].append(last_loss_dict.get('approx_kl', 0.0))
        train_history['clip_frac'].append(last_loss_dict.get('clip_frac', 0.0))
        train_history['explained_var'].append(last_loss_dict.get('explained_var', 0.0))
        train_history['entropy_coef'].append(last_loss_dict.get('entropy_coef', float(getattr(Config, "ENTROPY_COEF", 0.0))))

        status = "SUCCESS" if is_success else "DEADLOCK" if is_deadlock else "TIMEOUT"
        final_steps = info.get('steps', Config.MAX_STEPS_PER_EPISODE)
        final_makespan = info.get('makespan', 0.0)
        pn_display = pn_name
        print(f"Episode {episode:<5} | {pn_display:<18} | {status:<8} | Step:{final_steps:<5} | Msp:{final_makespan:<8.1f} | Reward: {ep_reward:<6.1f} | Val: {avg_value:<5.2f} | "
                f"Total_Loss: {sci_str(last_loss_dict['total'])} | Policy_Loss: {sci_str(last_loss_dict['policy'])} | "
                f"Value_Loss: {sci_str(last_loss_dict['value'])} | Entropy_Loss: {sci_str(last_loss_dict['entropy'])} | "
                f"KL: {last_loss_dict.get('approx_kl', 0.0):.4f} | ClipFrac: {last_loss_dict.get('clip_frac', 0.0):.2f} | "
                f"EV: {last_loss_dict.get('explained_var', 0.0):.2f} | EntC: {last_loss_dict.get('entropy_coef', 0.0):.4g}")

        if getattr(agent.optimizer, "_step_count", 0) > 0:
            scheduler.step()
        if episode % 10 == 0:
            save_training_log(train_history)

        if episode % Config.EVAL_INTERVAL == 0:
            print(f"\n>>> Validating on Benchmarks (Ep {episode})...")

            bench_results = evaluate_on_benchmark(agent, val_envs)

            for name, res in bench_results.items():
                status = str(res.get("status", "SUCCESS" if res.get("success") else "DEADLOCK"))
                ms = res.get("makespan", None)

                save_benchmark_log(name, episode, status, ms)
                plot_benchmark_result(name)

                best_tag = ""
                if status == "SUCCESS" and ms is not None:
                    prev_best = best_ms_per_bench.get(name, float("inf"))
                    if ms < prev_best:
                        best_ms_per_bench[name] = ms
                        best_path = os.path.join(Config.TRAINING_MODELS_DIR, f"model_best_{name}.pth")
                        agent.net.eval()
                        torch.save(agent.net.state_dict(), best_path)
                        agent.net.train()
                        best_tag = f" ★ NEW BEST (prev={prev_best:.1f})" if prev_best < float("inf") else " ★ FIRST BEST"

                ms_str = f"{ms:.1f}" if ms is not None else "---"
                print(f"    [{name:<15}] -> {status:<10} | Makespan: {ms_str}{best_tag}")

            plot_training_status(train_history)
            print(">>> Validation & Plotting Done.\n")

    try:
        save_training_log(train_history)
    except Exception as e:
        print(f"\n[WARN] Failed to save training_log.csv at end: {e}")
    if bool(getattr(Config, "SAVE_TRAIN_HISTORY_PT", True)):
        try:
            torch.save(train_history, os.path.join(Config.TRAINING_RESULT_DIR, "train_history.pt"))
        except Exception as e:
            print(f"[WARN] Failed to save train_history.pt: {e}")

    final_model_path = os.path.join(Config.TRAINING_MODELS_DIR, f"model_ep{Config.EPISODES}.pth")
    try:
        torch.save(agent.net.state_dict(), final_model_path)
        print(f"\n>>> Saved final model: {final_model_path}")
    except Exception as e:
        print(f"\n[WARN] Failed to save final model: {e}")

    if bool(getattr(Config, "FINAL_EVAL_ENABLED", True)):
        print("\n>>> Final Greedy Evaluation (Upper-Bound Estimation)")

        export_dir = None
        if bool(getattr(Config, "FINAL_EVAL_RANDOM_EXPORT", False)):
            export_dir = str(getattr(Config, "FINAL_EVAL_RANDOM_EXPORT_DIR", os.path.join(Config.TRAINING_RESULT_DIR, "FinalEval_RandomPNs")))
        rand_rows = evaluate_on_random_pns(
            agent,
            generator,
            n_pns=int(getattr(Config, "FINAL_EVAL_RANDOM_PN_NUM", 20)),
            seed=int(getattr(Config, "FINAL_EVAL_RANDOM_SEED", 0)),
            export_dir=export_dir,
        )
        rand_sum = summarize_eval_rows(rand_rows)
        print(
            f">>> RandomPN Summary: n={rand_sum['n']} | "
            f"succ={rand_sum['success_rate']:.3f} dead={rand_sum['deadlock_rate']:.3f} tout={rand_sum['timeout_rate']:.3f} | "
            f"avg_ms(success)={rand_sum['avg_makespan_success']}"
        )
        save_eval_rows_csv(rand_rows, os.path.join(Config.TRAINING_RESULT_DIR, "final_greedy_eval_random_pns.csv"))

        if export_dir:
            print(f">>> RandomPNs exported to: {export_dir}")

        if bool(getattr(Config, "FINAL_EVAL_BENCHMARK_ENABLED", False)):
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
            save_eval_rows_csv(bench_rows, os.path.join(Config.TRAINING_RESULT_DIR, "final_greedy_eval_benchmarks.csv"))

    if best_ms_per_bench:
        print(f"\n>>> Per-Benchmark Best Makespan Summary:")
        for name, ms in sorted(best_ms_per_bench.items()):
            print(f"    {name:<25} -> Best Msp: {ms:.1f}  (model: model_best_{name}.pth)")

if __name__ == "__main__":
    main()
