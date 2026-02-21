import numpy as np
import random
import re
import time
import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PN import PN
from Config import Config


class RandomPNGenerator:
    def __init__(self):
        self.min_job = Config.GEN_MIN_JOB
        self.max_job = Config.GEN_MAX_JOB
        self.min_op = Config.GEN_MIN_OP
        self.max_op = Config.GEN_MAX_OP
        self.min_res = Config.GEN_MIN_RES
        self.max_res = Config.GEN_MAX_RES
        self.min_time = Config.GEN_MIN_TIME
        self.max_time = Config.GEN_MAX_TIME
        self.min_lot = Config.GEN_MIN_LOT
        self.max_lot = Config.GEN_MAX_LOT
        
        self.ratio_alt = getattr(Config, "GEN_RATIO_ALT", 0.0)
        self.min_multi_r = getattr(Config, "GEN_MIN_MULTI_R", 1)
        self.max_multi_r = getattr(Config, "GEN_MAX_MULTI_R", 1)

        self.variable_capacity = getattr(Config, "GEN_VARIABLE_CAPACITY", False)
        self.max_attempts = 200

    def configure(self, stage_conf: dict):
        if "n_job_range" in stage_conf:
            self.min_job, self.max_job = stage_conf["n_job_range"]
        if "n_op_range" in stage_conf:
            self.min_op, self.max_op = stage_conf["n_op_range"]
        if "n_res_range" in stage_conf:
            self.min_res, self.max_res = stage_conf["n_res_range"]
        if "lot_range" in stage_conf:
            self.min_lot, self.max_lot = stage_conf["lot_range"]
        if "ratio_alt" in stage_conf:
            self.ratio_alt = stage_conf["ratio_alt"]

    def generate(self, export_dir=None, file_prefix=None):
        last_err = None
        for attempt in range(self.max_attempts):
            try:
                ratio_alt_eff = self._effective_ratio_alt(attempt)
                max_job_eff, max_op_eff = self._effective_max_job_op(attempt)

                spec = self._generate_problem_spec(
                    ratio_alt=ratio_alt_eff,
                    max_job=max_job_eff,
                    max_op=max_op_eff,
                )
                pn = self._compile_atomic_swap(spec)
                pn.n_jobs = spec["n_jobs"]
                pn.n_resources = spec["n_resources"]
                pn.n_ops = spec["total_main_ops"]
                pn.n_alt_ops = spec["total_alt_ops"]
                if pn.N_P > Config.MAX_PLACES or pn.N_T > Config.MAX_TRANSITIONS:
                    raise ValueError(
                        f"Generated PN too large: P={pn.N_P} (max {Config.MAX_PLACES}), "
                        f"T={pn.N_T} (max {Config.MAX_TRANSITIONS})."
                    )

                if export_dir:
                    safe_prefix = self._build_safe_prefix(pn, file_prefix)
                    pn.save_to_files(export_dir, safe_prefix)
                return pn
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"Failed to generate PN within {self.max_attempts} attempts. Last error: {last_err}"
        )

    def _effective_ratio_alt(self, attempt: int) -> float:
        if attempt < 60:
            return float(self.ratio_alt)
        if attempt < 120:
            return float(min(self.ratio_alt, 0.15))
        return 0.0

    def _effective_max_job_op(self, attempt: int):
        max_job_eff = self.max_job
        max_op_eff = self.max_op

        if attempt >= 120:
            max_job_eff = max(self.min_job, self.max_job - 1)
            max_op_eff = max(self.min_op, self.max_op - 1)
        if attempt >= 160:
            max_job_eff = max(self.min_job, max_job_eff - 1)
            max_op_eff = max(self.min_op, max_op_eff - 1)

        return max_job_eff, max_op_eff

    def _generate_problem_spec(self, ratio_alt: float, max_job: int, max_op: int):
        n_jobs = random.randint(self.min_job, max_job)
        n_resources = random.randint(self.min_res, self.max_res)
        if self.variable_capacity:
            res_caps = [random.choice([1, 1, 1, 2]) for _ in range(n_resources)]
        else:
            res_caps = [1 for _ in range(n_resources)]

        jobs = []
        total_main_ops = 0
        total_alt_ops = 0

        for jid in range(n_jobs):
            lot_size = random.randint(self.min_lot, self.max_lot)
            n_ops = random.randint(self.min_op, max_op)
            total_main_ops += n_ops

            segments = []
            last_main_res = None

            for _ in range(n_ops):
                main_opt = self._sample_option(
                    n_resources,
                    forbid_res_set=last_main_res,
                )
                options = [main_opt]
                if random.random() < ratio_alt:
                    alt_opt = self._sample_option(
                        n_resources,
                        forbid_res_set=main_opt["res_ids"],
                    )
                    if alt_opt["res_ids"] != main_opt["res_ids"]:
                        options.append(alt_opt)
                        total_alt_ops += 1

                segments.append(options)
                last_main_res = main_opt["res_ids"]

            jobs.append({"id": jid, "lot": lot_size, "segments": segments})
        
        return {
            "n_jobs": n_jobs,
            "n_resources": n_resources,
            "res_caps": res_caps,
            "jobs": jobs,
            "total_main_ops": total_main_ops,
            "total_alt_ops": total_alt_ops,
        }

    def _sample_option(self, n_resources: int, forbid_res_set=None):
        k_min = max(1, int(self.min_multi_r))
        k_max = max(k_min, int(self.max_multi_r))
        k = random.randint(k_min, min(k_max, n_resources))
        for _ in range(20):
            res_ids = tuple(sorted(random.sample(range(n_resources), k)))
            if forbid_res_set is None or res_ids != forbid_res_set:
                break

        proc_time = random.randint(self.min_time, self.max_time)
        return {"res_ids": res_ids, "time": proc_time}

    def _compile_atomic_swap(self, spec):
        n_resources = spec["n_resources"]
        res_caps = spec["res_caps"]
        jobs = spec["jobs"]

        places = []      # list of dict
        transitions = []
        res_pid = {}
        for rid in range(n_resources):
            pid = len(places)
            places.append({"type": "resource", "cap": int(res_caps[rid])})
            res_pid[rid] = pid

        for job in jobs:
            lot = int(job["lot"])
            segments = job["segments"]

            start_pid = len(places)
            places.append({"type": "start", "lot": lot})
            end_pid = len(places)
            places.append({"type": "end", "lot": lot})

            prev_places = [{"pid": start_pid, "held": frozenset()}]

            for seg in segments:
                curr_places = []
                for opt in seg:
                    curr_pid = len(places)
                    places.append({"type": "op", "time": int(opt["time"])})

                    curr_held = frozenset(opt["res_ids"])
                    curr_places.append({"pid": curr_pid, "held": curr_held})

                    for prev in prev_places:
                        prev_pid = prev["pid"]
                        prev_held = prev["held"]

                        req = curr_held.difference(prev_held)     # 新申请
                        rel = prev_held.difference(curr_held)     # 释放不再需要

                        in_places = [prev_pid] + [res_pid[r] for r in sorted(req)]
                        out_places = [curr_pid] + [res_pid[r] for r in sorted(rel)]

                        transitions.append({"in": in_places, "out": out_places})

                prev_places = curr_places

            for prev in prev_places:
                prev_pid = prev["pid"]
                prev_held = prev["held"]
                rel_all = sorted(prev_held)
                transitions.append(
                    {
                        "in": [prev_pid],
                        "out": [end_pid] + [res_pid[r] for r in rel_all],
                    }
                )

        n_p = len(places)
        n_t = len(transitions)
        
        LPLUS = np.zeros((n_p, n_t), dtype=np.int32)
        LMINUS = np.zeros((n_p, n_t), dtype=np.int32)
        StartM = np.zeros(n_p, dtype=np.int32)
        GoalM = np.zeros(n_p, dtype=np.int32)
        OpTimes = np.zeros(n_p, dtype=np.float32)
        
        for p_idx, p in enumerate(places):
            p_type = p["type"]
            if p_type == "resource":
                cap = int(p["cap"])
                StartM[p_idx] = cap
                GoalM[p_idx] = cap
            elif p_type == "start":
                StartM[p_idx] = int(p["lot"])
            elif p_type == "end":
                GoalM[p_idx] = int(p["lot"])
            elif p_type == "op":
                OpTimes[p_idx] = float(p["time"])

        for t_idx, t in enumerate(transitions):
            for p_in in t["in"]:
                LMINUS[p_in, t_idx] += 1
            for p_out in t["out"]:
                LPLUS[p_out, t_idx] += 1

        non_res_non_end = [i for i, p in enumerate(places) if p["type"] not in ("resource", "end")]
        res_only = [i for i, p in enumerate(places) if p["type"] == "resource"]
        end_only = [i for i, p in enumerate(places) if p["type"] == "end"]
        perm = non_res_non_end + res_only + end_only
        if perm != list(range(n_p)):
            LPLUS = LPLUS[perm, :]
            LMINUS = LMINUS[perm, :]
            StartM = StartM[perm]
            GoalM = GoalM[perm]
            OpTimes = OpTimes[perm]
            
        pn = PN(file_prefix=None)
        pn.name = f"RandPN_{spec['n_jobs']}J_{spec['total_main_ops']}O_{spec['n_resources']}R"
        pn.N_P = n_p
        pn.N_T = n_t
        pn.LPLUS = LPLUS
        pn.LMINUS = LMINUS
        pn.StartM = StartM
        pn.GoalM = GoalM
        pn.operationTimes = OpTimes
        pn.N_P_R = 0
        maxTokP_R = 0
        maxTokP_Start = 0
        for i in range(n_p):
            if StartM[i] != 0 and GoalM[i] != 0 and StartM[i] == GoalM[i]:
                pn.N_P_R += 1
                if maxTokP_R < StartM[i]:
                    maxTokP_R = int(StartM[i])
            elif StartM[i] != 0 and GoalM[i] == 0:
                if maxTokP_Start < StartM[i]:
                    maxTokP_Start = int(StartM[i])
        pn.MAX_TOK_PA = maxTokP_Start if maxTokP_Start <= maxTokP_R else maxTokP_R
        if pn.MAX_TOK_PA == 0:
            pn.MAX_TOK_PA = 1

        pn._cached_edge_index = None
        return pn

    def _build_safe_prefix(self, pn, file_prefix):
        base = file_prefix if file_prefix else getattr(pn, "name", "") or "PN"
        base = base.strip().replace("\\", "_").replace("/", "_")
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", base, flags=re.ASCII)
        base = re.sub(r"_+", "_", base).strip("_")
        if not base:
            base = "PN"
        if len(base) > 64:
            base = base[:64].rstrip("_")
        if not file_prefix:
            suffix = f"{int(time.time())}_{random.randint(1000, 9999)}"
            base = f"{base}_{suffix}"
        return base


class GeneratorPNPool:
    def __init__(self, generator: RandomPNGenerator):
        self.generator = generator
        self.pool: list[PN] = []
        self.pool_size = 0
        self._derive_info = {}
        self._pool_version = 1
        self.refresh_interval = int(getattr(Config, "GEN_POOL_REFRESH_INTERVAL", 200))
        self.refresh_ratio = float(getattr(Config, "GEN_POOL_REFRESH_RATIO", 0.10))
        self.export = bool(getattr(Config, "GEN_POOL_EXPORT", False))
        self.export_dir = getattr(Config, "GEN_POOL_EXPORT_DIR", "Training_result/GenPoolPNs")
        fixed_size = getattr(Config, "GEN_POOL_SIZE", None)
        if fixed_size is not None:
            try:
                self.pool_size = int(fixed_size)
            except Exception:
                self.pool_size = 0

        if not self.pool_size or self.pool_size <= 0:
            target_t = int(getattr(Config, "GEN_POOL_TARGET_TRANSITIONS_PER_PN", 1000))
            target_t = max(1, target_t)
            n_by_buffer = int(Config.BUFFER_SIZE // target_t)

            target_visits = getattr(Config, "GEN_POOL_TARGET_EPISODE_VISITS_PER_PN", None)
            if target_visits is None:
                if self.refresh_interval > 0:
                    approx = int(round(float(getattr(Config, "EPISODES", 0)) / float(self.refresh_interval)))
                else:
                    approx = 15
                target_visits = max(10, min(30, approx if approx > 0 else 15))
            else:
                try:
                    target_visits = int(target_visits)
                except Exception:
                    target_visits = 15
            target_visits = max(1, int(target_visits))

            if self.refresh_interval > 0 and self.refresh_ratio > 0:
                expected_life_eps = float(self.refresh_interval) / float(self.refresh_ratio)
                n_by_life = int(expected_life_eps // float(target_visits))
            else:
                expected_life_eps = float("inf")
                n_by_life = 10**9

            n = int(min(n_by_buffer, n_by_life))
            n_min = int(getattr(Config, "GEN_POOL_MIN_SIZE", 20))
            n_max = int(getattr(Config, "GEN_POOL_MAX_SIZE", 60))
            n = max(n_min, n)
            n = min(n_max, n)
            self.pool_size = max(1, n)
            self._derive_info = {
                "n_by_buffer": int(n_by_buffer),
                "target_transitions_per_pn": int(target_t),
                "n_by_life": int(n_by_life),
                "target_episode_visits_per_pn": int(target_visits),
                "expected_life_eps": float(expected_life_eps) if expected_life_eps != float("inf") else float("inf"),
            }

        self._initialized = False
        self.last_pool_idx = None
        self._pool_version = 1

    def _pool_version_for_episode(self, episode: int) -> int:
        if self.refresh_interval <= 0 or self.refresh_ratio <= 0:
            return 1
        ep = int(max(0, episode))
        return 1 + (ep // int(self.refresh_interval))

    def get_replace_cnt(self) -> int:
        if self.refresh_ratio <= 0:
            return 0
        return max(1, int(round(self.pool_size * self.refresh_ratio)))

    def describe(self) -> str:
        lines = []
        lines.append("[gen_pool]")
        lines.append(f"- pool_size (N)     : {self.pool_size}")
        lines.append(f"- refresh_interval  : {self.refresh_interval} eps")
        lines.append(f"- refresh_ratio     : {self.refresh_ratio:.2f}  (replace {self.get_replace_cnt()} per refresh)")
        lines.append(f"- export_pool_pn    : {self.export}  ({self.export_dir})")
        lines.append("- name_format       : PoolXXX__RandYYY  (XXX=pool version, YYY=PN index in pool)")

        if self._derive_info:
            n_by_buffer = self._derive_info.get("n_by_buffer", None)
            n_by_life = self._derive_info.get("n_by_life", None)
            t_t = self._derive_info.get("target_transitions_per_pn", None)
            v_t = self._derive_info.get("target_episode_visits_per_pn", None)
            life = self._derive_info.get("expected_life_eps", None)
            lines.append("")
            lines.append("[N derivation] (take min then clamp to [GEN_POOL_MIN_SIZE, GEN_POOL_MAX_SIZE])")
            lines.append(f"- by_buffer  : N <= BUFFER_SIZE/{t_t} => {n_by_buffer}")
            if life is not None and life != float('inf'):
                lines.append(f"- by_life    : N <= (refresh_interval/refresh_ratio)/{v_t} => {n_by_life}  (life~{life:.1f} eps)")
                exp_visits = float(life) / float(max(1, self.pool_size))
                lines.append(f"- exp_visits : ~{exp_visits:.1f} episodes per PN before replacement")
            else:
                lines.append("- by_life    : refresh disabled -> no extra constraint")

        return "\n".join(lines)

    def _make_one(self, pool_idx: int, tag: str, pool_version: int):
        if pool_idx < 0 or pool_idx >= self.pool_size:
            pool_idx = int(max(0, min(self.pool_size - 1, pool_idx)))
        slot_no = int(pool_idx) + 1
        file_prefix = None
        if self.export:
            file_prefix = f"Pool{int(pool_version):03d}__Rand{slot_no:03d}_{tag}"

        pn = self.generator.generate(
            export_dir=self.export_dir if self.export else None,
            file_prefix=file_prefix,
        )
        try:
            pn.base_name = getattr(pn, "name", "PN")
            pn.pool_slot = int(slot_no)
        except Exception:
            pass
        return pn

    def initialize(self):
        if self._initialized:
            return
        self._pool_version = 1
        self.pool = [self._make_one(i, tag="init", pool_version=self._pool_version) for i in range(self.pool_size)]
        self._initialized = True

    def maybe_refresh(self, episode: int):
        if not self._initialized:
            self.initialize()

        if self.refresh_interval <= 0 or self.refresh_ratio <= 0:
            return
        if episode <= 0:
            return
        if episode % self.refresh_interval != 0:
            return
        self._pool_version = self._pool_version_for_episode(int(episode))

        replace_cnt = max(1, int(round(self.pool_size * self.refresh_ratio)))
        replace_cnt = min(self.pool_size, replace_cnt)
        idxs = random.sample(range(self.pool_size), k=replace_cnt)
        for idx in idxs:
            self.pool[idx] = self._make_one(idx, tag=f"ep{episode:05d}", pool_version=self._pool_version)

    def sample(self, episode: int):
        if not self._initialized:
            self.initialize()

        self.maybe_refresh(int(episode))

        pool_idx = random.randrange(self.pool_size)
        self.last_pool_idx = pool_idx
        slot_no = int(pool_idx) + 1
        pool_version = int(self._pool_version_for_episode(int(episode)))
        pn = self.pool[pool_idx]
        try:
            pn.pool_version = int(pool_version)
            pn.name = f"Pool{pool_version:03d}__Rand{slot_no:03d}"
        except Exception:
            pass
        return pn, pool_idx

def test_generator_difficulty(n_tests=100, use_random_policy=True, use_serial_policy=False):
    from Environment import PN_Environment

    generator = RandomPNGenerator()

    deadlock_count = 0
    success_count = 0
    timeout_count = 0
    total_steps = []
    makespans = []

    if use_serial_policy:
        policy_name = "Serial (one job at a time)"
    elif use_random_policy:
        policy_name = "Random"
    else:
        policy_name = "Greedy (first choice)"

    print("=" * 60)
    print(f"Generator difficulty test (n_tests={n_tests})")
    print(f"Policy: {policy_name}")
    print("=" * 60)
    print("\nCurrent generator parameters:")
    print(f"  Jobs: {Config.GEN_MIN_JOB}-{Config.GEN_MAX_JOB}")
    print(f"  Ops:  {Config.GEN_MIN_OP}-{Config.GEN_MAX_OP}")
    print(f"  Res:  {Config.GEN_MIN_RES}-{Config.GEN_MAX_RES}")
    print(f"  Lot:  {Config.GEN_MIN_LOT}-{Config.GEN_MAX_LOT}")
    print(f"  Ratio Alt: {getattr(Config, 'GEN_RATIO_ALT', 0.0)}")
    print(f"  Multi-Res per Op: {getattr(Config, 'GEN_MIN_MULTI_R', 1)}-{getattr(Config, 'GEN_MAX_MULTI_R', 1)}")
    print(f"  Variable Capacity: {getattr(Config, 'GEN_VARIABLE_CAPACITY', False)}")
    print(f"  Compile: Atomic-Swap (no-buffer), alt closed, multi-resource enabled")
    print()

    for i in range(n_tests):
        try:
            # pn = generator.generate(
            #     export_dir="Training_result",
            #     file_prefix=f"RandPN_Ep{i}"
            # )
            pn = generator.generate()
            env = PN_Environment(pn_instance=pn)

            state, action_mask = env.reset()
            done = False
            steps = 0
            max_steps = Config.MAX_STEPS_PER_EPISODE
            info = {}
            early_deadlock = False

            while not done and steps < max_steps:
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) == 0:
                    early_deadlock = True
                    break

                if use_serial_policy:
                    action = _select_serial_action(valid_actions, env)
                elif use_random_policy:
                    action = random.choice(valid_actions.tolist())
                else:
                    action = int(valid_actions[0])

                state, action_mask, reward, done, info = env.step(action)
                steps += 1

            total_steps.append(steps)

            if early_deadlock or info.get("deadlock", False):
                deadlock_count += 1
            elif info.get("success", False):
                success_count += 1
                makespans.append(info.get("makespan", 0))
            elif steps >= max_steps:
                timeout_count += 1

            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{n_tests} | Success: {success_count} | Deadlock: {deadlock_count} | Timeout: {timeout_count}")

        except Exception as e:
            print(f"Test {i+1} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    deadlock_rate = deadlock_count / n_tests * 100
    success_rate = success_count / n_tests * 100
    timeout_rate = timeout_count / n_tests * 100

    print(f"\nTotal tests:   {n_tests}")
    print(f"Success:       {success_count} ({success_rate:.1f}%)")
    print(f"Deadlock:      {deadlock_count} ({deadlock_rate:.1f}%)")
    print(f"Timeout:       {timeout_count} ({timeout_rate:.1f}%)")

    if total_steps:
        print(f"\nAvg steps:     {np.mean(total_steps):.1f}")
        print(f"Max steps:     {np.max(total_steps)}")
        print(f"Min steps:     {np.min(total_steps)}")

    if makespans:
        print(f"\nSuccess case Makespan:")
        print(f"  Avg:        {np.mean(makespans):.1f}")
        print(f"  Max:        {np.max(makespans):.1f}")
        print(f"  Min:        {np.min(makespans):.1f}")

    print("\n" + "=" * 60)
    print("Difficulty Assessment")
    print("=" * 60)

    if deadlock_rate < 10:
        difficulty = "Too easy"
        suggestion = "Suggestion: Reduce resources / Increase ops / Increase multi-resource or ratio_alt"
    elif deadlock_rate < 30:
        difficulty = "Slightly easy"
        suggestion = "Suggestion: Increase multi-resource or reduce resources"
    elif deadlock_rate <= 50:
        difficulty = "OK"
        suggestion = "Current difficulty suitable for training"
    elif deadlock_rate <= 70:
        difficulty = "Slightly hard"
        suggestion = "Suggestion: Increase resources / Reduce multi-resource or ratio_alt"
    else:
        difficulty = "Too hard"
        suggestion = "Suggestion: Increase resources / Reduce ops / Reduce multi-resource or disable alt"

    print(f"\nDifficulty: {difficulty}")
    print(f"Deadlock rate: {deadlock_rate:.1f}% (target: 30%~50%)")
    print(suggestion)

    return {
        "success_rate": success_rate,
        "deadlock_rate": deadlock_rate,
        "timeout_rate": timeout_rate,
        "avg_steps": float(np.mean(total_steps)) if total_steps else 0.0,
        "avg_makespan": float(np.mean(makespans)) if makespans else 0.0,
    }


def _select_serial_action(valid_actions, env):
    valid_list = valid_actions.tolist() if hasattr(valid_actions, "tolist") else list(valid_actions)

    N_P = env.pn.N_P
    non_start_actions = []
    start_actions = []

    for action in valid_list:
        t_idx = int(action)

        input_places = []
        for p in range(N_P):
            if env.pn.LMINUS[p, t_idx] > 0:
                input_places.append(p)

        is_start_transition = False
        for p in input_places:
            if env.pn.StartM[p] > 0 and env.pn.GoalM[p] == 0:
                is_start_transition = True
                break

        if is_start_transition:
            start_actions.append(t_idx)
        else:
            non_start_actions.append(t_idx)

    if non_start_actions:
        return random.choice(non_start_actions)
    if start_actions:
        return start_actions[0]
    return valid_list[0]


StructuredPNGenerator = RandomPNGenerator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PN generator difficulty")
    parser.add_argument("-n", "--n_tests", type=int, default=100, help="Number of PN tests (default: 100)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy policy instead of random")
    parser.add_argument("--serial", action="store_true", help="Use serial policy (verify deadlock avoidability)")

    args = parser.parse_args()

    test_generator_difficulty(
        n_tests=args.n_tests,
        use_random_policy=not args.greedy and not args.serial,
        use_serial_policy=args.serial,
    )

