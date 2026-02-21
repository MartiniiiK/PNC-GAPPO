import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Config import Config
import torch
import seaborn as sns
from typing import Dict, List, Optional, Any

def ensure_dirs():
    os.makedirs(Config.TRAINING_RESULT_DIR, exist_ok=True)
    os.makedirs(Config.EVAL_RESULT_DIR, exist_ok=True)
    os.makedirs(Config.TRAINING_MODELS_DIR, exist_ok=True)


def get_output_paths(file_prefix):
    if '_' in file_prefix:
        main_folder_name, sub_folder_name = file_prefix.rsplit('_', 1)
    else:
        main_folder_name = file_prefix
        sub_folder_name = "Run"

    result_path = os.path.join(Config.RESULT_DIR, main_folder_name, sub_folder_name)
    model_path = os.path.join(Config.MODEL_DIR, main_folder_name, sub_folder_name)

    return result_path, model_path


def evaluate_on_benchmark(agent, val_envs):
    results = {}
    
    print(f"\n{'-'*80}")
    print(f"{'Benchmark Name':<25} | {'Status':<10} | {'Steps':<8} | {'Makespan':<10} | {'Reward':<10}")
    print(f"{'-'*80}")
    
    for env in val_envs:
        state, action_mask = env.reset()
        done = False
        current_ms = 0
        ep_reward = 0
        info = {}
        while not done:
            action, _, _ = agent.select_action(state, action_mask, is_training=False)
            
            next_state, next_action_mask, reward, done, info = env.step(action)
            current_ms += info.get('makespan_increment', 0)
            state = next_state
            action_mask = next_action_mask
            ep_reward += reward
            
        status_str = "SUCCESS" if info.get('success') else "DEADLOCK" if info.get('deadlock') else "TIMEOUT"
        final_steps = info.get('steps', 0)
        step_display = status_str if status_str == "TIMEOUT" else final_steps
        final_makespan = info.get('makespan', 0.0)
 
        print(f"{env.name[:25]:<25} | {status_str:<10} | {str(step_display):<8} | {final_makespan:<10.1f} | {ep_reward:<10.1f}")
        results[env.name] = {
            "success": info.get('success', False),
            "makespan": final_makespan if info.get('success') else None, 
            "reward": ep_reward,
            "status": status_str,
            "steps": final_steps
       }
            
    return results

def save_training_log(history_dict):
    df = pd.DataFrame(history_dict)
    df.to_csv(os.path.join(Config.TRAINING_RESULT_DIR, "training_log.csv"), index=False)

def save_benchmark_log(benchmark_name, episode, status, makespan):
    file_path = os.path.join(Config.EVAL_RESULT_DIR, f"{benchmark_name}.csv")
    new_data = {
        'episode': [episode],
        'status': [status],
        'makespan': [makespan if makespan is not None else -1]
    }
    new_df = pd.DataFrame(new_data)
    if not os.path.exists(file_path):
        new_df.to_csv(file_path, index=False)
    else:
        new_df.to_csv(file_path, mode='a', header=False, index=False)

def plot_training_status(history):
    df = pd.DataFrame(history)
    if len(df) < Config.EVAL_INTERVAL: return
    df['group_idx'] = (df['episode'] - 1) // Config.EVAL_INTERVAL + 1
    episodes = df['episode']
    
    grouped = df.groupby('group_idx').agg({
        'reward': 'mean',          
        'deadlock': 'sum',         
        'success': 'mean',         
        'episode': 'max'           
    }).reset_index()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams['font.family'] = Config.FONT_FAMILY
    colors = sns.color_palette("deep")
    fig, axs = plt.subplots(4, 2, figsize=(20, 30)) 
    fig.suptitle(f'Training Dynamics (Aggregated per {Config.EVAL_INTERVAL} Eps)', fontsize=24, fontweight='bold', y=0.98)
    ax1 = axs[0, 0] 
    sns.lineplot(data=grouped, x='group_idx', y='reward', marker='o', markersize=8, linewidth=3, color=colors[2], ax=ax1) # Green
    ax1.fill_between(grouped['group_idx'], grouped['reward'], grouped['reward'].min(), color=colors[2], alpha=0.1)
    ax1.set_ylabel('Avg Reward (Group)', fontweight='bold', color=colors[2])
    ax1.set_title('Macro: Average Reward per Group', fontsize=14)
    ax1.set_xlabel('')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax2 = axs[0, 1]
    ax2.bar(grouped['group_idx'], grouped['deadlock'], color='#c0392b', alpha=0.6, width=0.5, label='Deadlock Count')
    for idx, row in grouped.iterrows():
        if row.deadlock > 0:
            ax2.text(row.group_idx, row.deadlock + 0.2, f"{int(row.deadlock)}", color='#c0392b', ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel('Deadlock Count', fontweight='bold', color='#c0392b')
    ax2.set_ylim(0, Config.EVAL_INTERVAL + 2)
    ax2.grid(False)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(grouped['group_idx'], grouped['success'], color='#2980b9', marker='D', markersize=6, linewidth=2.5, linestyle='-', label='Success Rate')
    ax2_twin.set_ylabel('Success Rate (Group)', fontweight='bold', color='#2980b9')
    ax2_twin.set_ylim(0, 1.1)
    ax2_twin.grid(True, linestyle=':', alpha=0.5)
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    ax2.set_title('Macro: Deadlock Count vs Success Rate', fontsize=14)
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('')
    ax3 = axs[1, 0]
    reward_series = df['reward']
    window = max(10, len(reward_series)//50)
    smooth_reward = reward_series.rolling(window=window, min_periods=1).mean()
    ax3.plot(episodes, reward_series, color='lightgray', alpha=0.4, label='Raw', linewidth=1)
    ax3.plot(episodes, smooth_reward, color='tab:blue', linewidth=2, label=f'Smoothed (w={window})')
    
    ax3.set_title('Micro: Episode Reward (Step-by-Step)', fontsize=14)
    ax3.set_ylabel('Reward', fontweight='bold', color='tab:blue')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('')
    ax4 = axs[1, 1]
    if 'success' in df.columns:
        succ_series = df['success']
        roll_succ = succ_series.rolling(window=100, min_periods=1).mean()
        
        ax4.plot(episodes, roll_succ, color='tab:green', linewidth=2.5)
        ax4.fill_between(episodes, roll_succ, 0, color='tab:green', alpha=0.1)
        
        ax4.set_title('Micro: Rolling Success Rate (Last 100 Eps)', fontsize=14)
        ax4.set_ylabel('Success Rate', fontweight='bold', color='tab:green')
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('')
    ax5 = axs[2, 0]
    window = max(5, len(df)//50)
    s_val = df['loss_value'].rolling(window).mean()
    s_pol = df['loss_policy'].rolling(window).mean()
    s_ent = df['loss_entropy'].rolling(window).mean()
    ln1 = ax5.plot(episodes, s_val, color='#e67e22', label='Value Loss', linewidth=2)
    ln2 = ax5.plot(episodes, s_pol, color='#2980b9', label='Policy Loss', linewidth=1.5, linestyle='-')
    ln3 = ax5.plot(episodes, s_ent, color='#27ae60', label='Entropy Loss', linewidth=1.5, linestyle=':')
    
    ax5.set_ylabel('Loss', fontweight='bold', color='#2c3e50')
    ax5.tick_params(axis='y', labelcolor='#2c3e50')
    ax5.grid(True, linestyle='--', alpha=0.3)
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='upper center', ncol=3, fontsize=10)
    ax5.set_title(f'Internal: PPO Loss Breakdown', fontsize=14)
    ax6 = axs[2, 1]
    s_v_est = df['avg_value'].rolling(window).mean()
    ax6.plot(episodes, df['avg_value'], color='gray', alpha=0.15)
    ax6.plot(episodes, s_v_est, color='#8e44ad', linewidth=2)
    ax6.set_ylabel('Avg State Value', color='#8e44ad', fontweight='bold')
    ax6.set_title(f'Internal: Critic Value Estimate', fontsize=14)
    ax7 = axs[3, 0]
    if 'approx_kl' in df.columns:
        s_kl = df['approx_kl'].rolling(window).mean()
        ax7.plot(episodes, s_kl, color='#2980b9', linewidth=2, label='Approx KL')
        target_kl = getattr(Config, "TARGET_KL", None)
        if target_kl is not None and float(target_kl) > 0:
            ax7.axhline(y=float(target_kl), color='#2980b9', linestyle='--', alpha=0.6, linewidth=1.5, label=f'Target KL={float(target_kl):.3g}')
        ax7.set_ylabel('Approx KL', color='#2980b9', fontweight='bold')
        ax7.tick_params(axis='y', labelcolor='#2980b9')

    ax7_twin = ax7.twinx()
    if 'clip_frac' in df.columns:
        s_cf = df['clip_frac'].rolling(window).mean()
        ax7_twin.plot(episodes, s_cf, color='#e67e22', linewidth=2, linestyle='-', label='ClipFrac')
        ax7_twin.set_ylabel('Clip Fraction', color='#e67e22', fontweight='bold')
        ax7_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax7_twin.set_ylim(0, 1.05)
    ax7.set_title('Diag: Approx KL & Clip Fraction', fontsize=14)
    ax7.grid(True, linestyle='--', alpha=0.3)
    lines_1, labels_1 = ax7.get_legend_handles_labels()
    lines_2, labels_2 = ax7_twin.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax7.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10)
    ax8 = axs[3, 1]
    if 'explained_var' in df.columns:
        s_ev = df['explained_var'].rolling(window).mean()
        ax8.plot(episodes, s_ev, color='#27ae60', linewidth=2.5, label='Explained Var')
        ax8.set_ylabel('Explained Variance', color='#27ae60', fontweight='bold')
        ax8.tick_params(axis='y', labelcolor='#27ae60')
        ax8.set_ylim(-1.05, 1.05)
        ax8.axhline(y=0.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    ax8_twin = ax8.twinx()
    if 'entropy_coef' in df.columns:
        s_ec = df['entropy_coef'].rolling(window).mean()
        ax8_twin.plot(episodes, s_ec, color='#8e44ad', linewidth=2, linestyle=':', label='Entropy Coef')
        ax8_twin.set_ylabel('Entropy Coef', color='#8e44ad', fontweight='bold')
        ax8_twin.tick_params(axis='y', labelcolor='#8e44ad')

    ax8.set_title('Diag: Explained Var & Entropy Coef', fontsize=14)
    ax8.grid(True, linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax8.get_legend_handles_labels()
    lines_2, labels_2 = ax8_twin.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax8.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.TRAINING_RESULT_DIR, "training_process.png"), dpi=Config.PLOT_DPI)
    plt.close()

def plot_benchmark_result(benchmark_name):
    file_path = os.path.join(Config.EVAL_RESULT_DIR, f"{benchmark_name}.csv")
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)
    if len(df) < 2: return
    episodes = df['episode']
    makespans = df['makespan']
    statuses = df['status']

    # 提取有效 Makespan (非死锁)
    valid_mask = (statuses == 'SUCCESS')
    valid_eps = episodes[valid_mask]
    valid_ms = makespans[valid_mask]
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    width = (episodes.iloc[-1] - episodes.iloc[0]) / len(episodes) * 0.6 if len(episodes) > 1 else 10
    ax.bar(valid_eps, valid_ms, width=width, color='skyblue', alpha=0.6, label='Makespan (Bar)')
    ax.plot(valid_eps, valid_ms, color='navy', marker='o', markersize=4, linewidth=2, linestyle='-', label='Trend (Line)')
    deadlock_mask = (statuses == 'DEADLOCK')
    if deadlock_mask.any():
        deadlock_eps = episodes[deadlock_mask]
        y_pos = valid_ms.max() * 1.05 if not valid_ms.empty else 100
        ax.scatter(deadlock_eps, [y_pos]*len(deadlock_eps), color='tab:red', marker='X', s=100, label='Deadlock', zorder=5)
        for dep in deadlock_eps:
            ax.axvline(x=dep, color='tab:red', alpha=0.1, linestyle='--')
    ax.set_title(f'Evaluation History: {benchmark_name}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Makespan')
    ax.legend(loc='upper right', frameon=True, framealpha=1.0)
    if not valid_ms.empty:
        best_ms = valid_ms.min()
        ax.axhline(y=best_ms, color='gold', linestyle='--', linewidth=2, label=f'Best: {best_ms}')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.EVAL_RESULT_DIR, f"{benchmark_name}_history.png"), dpi=Config.PLOT_DPI)
    plt.close()
    
    print(f"Updated plot for {benchmark_name}")


def _set_global_seeds(seed: int):
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _episode_rollout_greedy(agent, env) -> Dict[str, Any]:
    state, action_mask = env.reset()
    done = False
    ep_reward = 0.0
    info = {}
    while not done:
        action, _, _ = agent.select_action(state, action_mask, is_training=False)
        next_state, next_action_mask, reward, done, info = env.step(action)
        state = next_state
        action_mask = next_action_mask
        ep_reward += float(reward)

    status_str = "SUCCESS" if info.get("success") else "DEADLOCK" if info.get("deadlock") else "TIMEOUT"
    return {
        "pn_name": getattr(env, "name", "PN"),
        "status": status_str,
        "success": 1 if info.get("success") else 0,
        "deadlock": 1 if info.get("deadlock") else 0,
        "timeout": 1 if info.get("timeout") else 0,
        "steps": int(info.get("steps", 0)),
        "makespan": float(info.get("makespan", 0.0)),
        "reward": float(ep_reward),
    }


def evaluate_on_random_pns(
    agent,
    generator,
    n_pns: int,
    seed: int = 0,
    export_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from Environment import PN_Environment

    n_pns = int(n_pns)
    if n_pns <= 0:
        return []

    _set_global_seeds(seed)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(100, n_pns * 50)
    while len(rows) < n_pns and attempts < max_attempts:
        attempts += 1
        pn = generator.generate()
        try:
            env = PN_Environment(pn_instance=pn)
        except Exception:
            continue

        if export_dir and hasattr(pn, "save_to_files"):
            safe_name = str(getattr(pn, "name", "RandPN"))
            file_prefix = f"EvalRand_{len(rows):03d}_{safe_name}"
            try:
                pn.save_to_files(export_dir, file_prefix=file_prefix)
            except Exception:
                pass

        rows.append(_episode_rollout_greedy(agent, env))

    return rows


def summarize_eval_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "success_rate": 0.0, "deadlock_rate": 0.0, "timeout_rate": 0.0, "avg_makespan_success": None}

    n = len(rows)
    succ = sum(int(r.get("success", 0)) for r in rows)
    dead = sum(int(r.get("deadlock", 0)) for r in rows)
    tout = sum(int(r.get("timeout", 0)) for r in rows)
    ms_list = [float(r["makespan"]) for r in rows if int(r.get("success", 0)) == 1]
    avg_ms = float(np.mean(ms_list)) if ms_list else None
    return {
        "n": int(n),
        "success_rate": float(succ) / float(n),
        "deadlock_rate": float(dead) / float(n),
        "timeout_rate": float(tout) / float(n),
        "avg_makespan_success": avg_ms,
    }


def save_eval_rows_csv(rows: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    
    
from collections import deque
from Config import Config

class CurriculumManager:
    def __init__(self, generator):
        self.generator = generator
        self.stages = Config.CURRICULUM_SCHEDULE
        self.current_idx = 0
        
        # 记录最近 N 场的胜负 (1=Success, 0=Fail)
        self.history = deque(maxlen=Config.CURRICULUM_WINDOW_SIZE)
        self._apply_stage(0)

    def _apply_stage(self, idx):
        stage_conf = self.stages[idx]
        self.generator.configure(stage_conf)
        self.history.clear()
        self.current_idx = idx

    def record_result(self, is_success):
        self.history.append(1 if is_success else 0)
        
        current_stage = self.stages[self.current_idx]
        if len(self.history) < self.history.maxlen:
            return False, current_stage['name']
        win_rate = sum(self.history) / len(self.history)
        threshold = current_stage['upgrade_threshold']
        if self.current_idx < len(self.stages) - 1 and win_rate >= threshold:
            self.current_idx += 1
            print(f"\n*** [LEVEL UP] Win Rate {win_rate:.2f} >= {threshold}. Promoting! ***")
            self._apply_stage(self.current_idx)
            return True, self.stages[self.current_idx]['name']
            
        return False, current_stage['name']

    @property
    def current_stage_name(self):
        return self.stages[self.current_idx]['name']
    
    
def sci_str(x, prec=2):
    if x == 0:
        return "0"
    import math
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    return f"{mant:.{prec}f}×10^{exp}"


def _assign_place_to_job(pn) -> List[int]:
    N_P, N_T = int(pn.N_P), int(pn.N_T)
    job_of = [-1] * N_P
    start_places = [p for p in range(N_P)
                    if pn.StartM[p] > 0 and pn.GoalM[p] == 0]

    for job_id, sp in enumerate(start_places):
        queue = [sp]
        visited = set()
        while queue:
            p = queue.pop(0)
            if p in visited:
                continue
            visited.add(p)
            job_of[p] = job_id
            for t in range(N_T):
                if pn.LMINUS[p, t] > 0:
                    for p2 in range(N_P):
                        if pn.LPLUS[p2, t] > 0:
                            if pn.StartM[p2] > 0 and pn.GoalM[p2] > 0 and pn.StartM[p2] == pn.GoalM[p2]:
                                continue
                            if p2 not in visited:
                                queue.append(p2)
    return job_of


