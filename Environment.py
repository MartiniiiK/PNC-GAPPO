import numpy as np
from Config import Config
from PN import PN


class PN_Environment:

    def __init__(self, pn_name=None, pn_instance=None):
        if pn_instance is not None:
            self.pn = pn_instance
            self.name = getattr(pn_instance, "name", "PN")
        elif pn_name is not None:
            self.pn = PN(pn_name)
            self.name = pn_name
        else:
            self.pn = PN(Config.TRAIN_PN_LIST[0])
            self.name = Config.TRAIN_PN_LIST[0]

        self.state_M = None
        self.state_R = None
        self.current_step = 0
        self.current_makespan = 0.0
        self._context_cache = None

        self._validate_pn_size()

    def _validate_pn_size(self):
        if self.pn.N_P > Config.MAX_PLACES:
            raise ValueError(
                f"PN '{self.name}' has {self.pn.N_P} places, exceeds MAX_PLACES={Config.MAX_PLACES}. "
                f"Please increase Config.MAX_PLACES or use a smaller PN."
            )
        if self.pn.N_T > Config.MAX_TRANSITIONS:
            raise ValueError(
                f"PN '{self.name}' has {self.pn.N_T} transitions, exceeds MAX_TRANSITIONS={Config.MAX_TRANSITIONS}. "
                f"Please increase Config.MAX_TRANSITIONS or use a smaller PN."
            )

    def reset(self):
        self.state_M, self.state_R = self.pn.get_initial_state()
        self.current_step = 0
        self.current_makespan = 0.0
        self._context_cache = self._build_context()
        return self._get_obs(), self.get_action_mask()

    def _build_context(self):
        PAD = float(Config.PAD_VALUE)
        N_P = int(self.pn.N_P)
        N_T = int(self.pn.N_T)

        start_m = np.full(Config.MAX_PLACES, PAD, dtype=np.float32)
        goal_m = np.full(Config.MAX_PLACES, PAD, dtype=np.float32)
        op_times = np.full(Config.MAX_PLACES, PAD, dtype=np.float32)

        start_m[:N_P] = self.pn.StartM.astype(np.float32)
        goal_m[:N_P] = self.pn.GoalM.astype(np.float32)
        op_times[:N_P] = self.pn.operationTimes.astype(np.float32)

        lminus = np.zeros((Config.MAX_PLACES, Config.MAX_TRANSITIONS), dtype=np.float32)
        lplus = np.zeros((Config.MAX_PLACES, Config.MAX_TRANSITIONS), dtype=np.float32)
        lminus[:N_P, :N_T] = self.pn.LMINUS.astype(np.float32)
        lplus[:N_P, :N_T] = self.pn.LPLUS.astype(np.float32)

        t_exists = np.zeros(Config.MAX_TRANSITIONS, dtype=np.float32)
        t_exists[:N_T] = 1.0

        return {
            "start_m": start_m,
            "goal_m": goal_m,
            "op_times": op_times,
            "lminus": lminus,
            "lplus": lplus,
            "t_exists": t_exists,
        }

    def _get_obs(self):
        PAD = float(Config.PAD_VALUE)
        N_P = int(self.pn.N_P)

        m_padded = np.full(Config.MAX_PLACES, PAD, dtype=np.float32)
        m_padded[:N_P] = self.state_M.astype(np.float32)

        r_padded = np.full((Config.MAX_PLACES, Config.MAX_TOKENS_PER_PLACE), PAD, dtype=np.float32)
        r_padded[:N_P, :] = 0.0
        for p in range(N_P):
            token_count = int(self.state_M[p])
            if token_count <= 0:
                continue
            max_valid = min(token_count, self.state_R.shape[1])
            valid_r = self.state_R[p, :max_valid]
            fill_len = min(len(valid_r), Config.MAX_TOKENS_PER_PLACE)
            r_padded[p, :fill_len] = valid_r[:fill_len].astype(np.float32)

        mask = np.zeros(Config.MAX_PLACES, dtype=np.float32)
        mask[:N_P] = 1.0

        dynamic_state = np.concatenate([m_padded, r_padded.flatten(), mask]).astype(np.float32)
        return (dynamic_state, self._context_cache)

    def get_action_mask(self):
        enabled = self.pn.get_enabled_transitions(self.state_M)
        action_mask = np.zeros(Config.MAX_TRANSITIONS, dtype=np.float32)
        action_mask[: self.pn.N_T] = enabled.astype(np.float32)
        return action_mask

    def step(self, action: int):
        self.current_step += 1
        scale = float(getattr(Config, "REWARD_SCALE", 1.0))

        enabled = self.pn.get_enabled_transitions(self.state_M)
        if action >= len(enabled) or enabled[action] == 0:
            info = {
                "deadlock": True,
                "invalid_action": True,
                "success": False,
                "timeout": False,
                "steps": self.current_step,
                "makespan_increment": 0.0,
                "makespan": self.current_makespan,
            }
            reward = float(Config.REWARD_DEADLOCK) * scale
            return self._get_obs(), self.get_action_mask(), float(reward), True, info

        new_M, new_R, delta_time = self.pn.step(self.state_M, self.state_R, int(action))
        self.state_M = new_M
        self.state_R = new_R

        reward = -float(delta_time)
        self.current_makespan += float(delta_time)

        done = False
        info = {
            "deadlock": False,
            "success": False,
            "timeout": False,
            "steps": self.current_step,
            "makespan_increment": float(delta_time),
            "makespan": self.current_makespan,
        }

        if self.pn.is_goal(self.state_M, self.state_R):
            done = True
            info["success"] = True
            reward += float(getattr(Config, "REWARD_SUCCESS_BONUS", 0))
        else:
            next_enabled = self.pn.get_enabled_transitions(self.state_M)
            if self.pn.is_deadlock(next_enabled):
                done = True
                info["deadlock"] = True
                reward = float(Config.REWARD_DEADLOCK)

        if self.current_step >= Config.MAX_STEPS_PER_EPISODE:
            done = True
            info["timeout"] = True
            reward = float(Config.REWARD_DEADLOCK)
        reward = float(reward) * scale
        return self._get_obs(), self.get_action_mask(), float(reward), bool(done), info

