import os
import numpy as np
import re
from Config import Config
import torch

class PN:
    def __init__(self, file_prefix=None):
        self.file_prefix = file_prefix
        self.init_file = os.path.join(Config.PN_FILES_DIR, f"{file_prefix}_init.txt")
        self.matrix_file = os.path.join(Config.PN_FILES_DIR, f"{file_prefix}_matrix.txt")

        self.N_P = 0
        self.N_T = 0
        self.N_P_R = 0
        self.MAX_TOK_PA = 0
        self.StartM = None
        self.GoalM = None
        self.operationTimes = None
        self.LPLUS = None
        self.LMINUS = None
        self._cached_edge_index = None
        if self.file_prefix is not None:
            self.init_file = os.path.join(Config.PN_FILES_DIR, f"{file_prefix}_init.txt")
            self.matrix_file = os.path.join(Config.PN_FILES_DIR, f"{file_prefix}_matrix.txt")
            self._load_data()

    def save_to_files(self, output_dir: str, file_prefix: str):
        os.makedirs(output_dir, exist_ok=True)

        self.file_prefix = file_prefix
        self.init_file = os.path.join(output_dir, f"{file_prefix}_init.txt")
        self.matrix_file = os.path.join(output_dir, f"{file_prefix}_matrix.txt")

        def _fmt_number(val):
            f = float(val)
            if f.is_integer():
                return str(int(f))
            return f"{f:.6g}"
        with open(self.init_file, "w") as f:
            f.write(" ".join(_fmt_number(v) for v in self.StartM) + "\n")
            f.write(" ".join(_fmt_number(v) for v in self.operationTimes) + "\n")
            f.write(" ".join(_fmt_number(v) for v in self.GoalM) + "\n")
        with open(self.matrix_file, "w") as f:
            for t in range(int(self.N_T)):
                row_vals = []
                for p in range(int(self.N_P)):
                    plus = int(self.LPLUS[p, t]) if self.LPLUS is not None else 0
                    minus = int(self.LMINUS[p, t]) if self.LMINUS is not None else 0
                    if plus > 0 and minus > 0:
                        val = plus - minus
                    elif plus > 0:
                        val = plus
                    elif minus > 0:
                        val = -minus
                    else:
                        val = 0
                    row_vals.append(str(val))
                f.write(" ".join(row_vals) + "\n")
    def _load_data(self):
        try:
            with open(self.init_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find {self.init_file}. Please check path.")
        words = re.split(r'\s+', lines[0])
        self.N_P = len(words)
        self.StartM = np.array([int(w) for w in words], dtype=np.int32)
        words = re.split(r'\s+', lines[1])
        self.operationTimes = np.array([int(w) for w in words], dtype=np.int32)
        words = re.split(r'\s+', lines[2])
        self.GoalM = np.array([int(w) for w in words], dtype=np.int32)
        maxTokP_R = 0
        maxTokP_Start = 0

        for i in range(self.N_P):
            if self.StartM[i] != 0 and self.GoalM[i] != 0 and self.StartM[i] == self.GoalM[i]:
                self.N_P_R += 1
                if maxTokP_R < self.StartM[i]:
                    maxTokP_R = self.StartM[i]
            elif self.StartM[i] != 0 and maxTokP_Start < self.StartM[i]:
                maxTokP_Start = self.StartM[i]

        self.MAX_TOK_PA = maxTokP_Start if maxTokP_Start <= maxTokP_R else maxTokP_R
        if self.MAX_TOK_PA == 0: self.MAX_TOK_PA = 1
        try:
            with open(self.matrix_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find {self.matrix_file}")
        self.N_T = len(lines)

        temp_matrix = np.zeros((self.N_T, self.N_P), dtype=np.int32)
        for i, line in enumerate(lines):
            words = re.split(r'\s+', line)
            temp_matrix[i, :] = [int(w) for w in words[:self.N_P]]
        self.LPLUS = np.zeros((self.N_P, self.N_T), dtype=np.int32)
        self.LMINUS = np.zeros((self.N_P, self.N_T), dtype=np.int32)

        for t in range(self.N_T):
            for p in range(self.N_P):
                val = temp_matrix[t, p]
                if val >= 1:
                    self.LPLUS[p, t] = val
                else:
                    self.LPLUS[p, t] = 0

                if val <= -1:
                    self.LMINUS[p, t] = -val
                else:
                    self.LMINUS[p, t] = 0

    def get_initial_state(self):
        StartR = np.zeros((self.N_P, self.MAX_TOK_PA), dtype=np.float32)
        return self.StartM.copy(), StartR.copy()

    def get_enabled_transitions(self, M):
        enabled = np.ones(self.N_T, dtype=np.int32)
        for t in range(self.N_T):
            for p in range(self.N_P):
                if self.LMINUS[p, t] > 0 and M[p] < self.LMINUS[p, t]:
                    enabled[t] = 0
                    break
        return enabled

    def step(self, M, R, t_idx):
        delt = 0.0
        for p in range(self.N_P):
            if self.LMINUS[p, t_idx] > 0 and self.operationTimes[p] > 0:
                token_idx = M[p] - 1
                if token_idx >= 0 and token_idx < self.MAX_TOK_PA:
                    if delt < R[p, token_idx]:
                        delt = R[p, token_idx]
        MZ = np.copy(M)
        for p in range(self.N_P):
            if self.LMINUS[p, t_idx] > 0:
                MZ[p] = M[p] - self.LMINUS[p, t_idx]
        MF = np.copy(MZ)
        for p in range(self.N_P):
            if self.LPLUS[p, t_idx] > 0:
                MF[p] = MZ[p] + self.LPLUS[p, t_idx]
        RZ = np.copy(R)
        RZ = RZ - delt
        RZ[RZ < 0] = 0
        RF = np.copy(RZ)
        for p in range(self.N_P):
            if self.LPLUS[p, t_idx] > 0 and self.operationTimes[p] > 0:
                RF[p, 1:] = RZ[p, :-1]
                RF[p, 0] = self.operationTimes[p]
        return MF, RF, delt

    def is_goal(self, M, R):
        if not np.array_equal(M, self.GoalM):
            return False
        if np.any(R > 0):
            return False

        return True

    def is_deadlock(self, enabled_transitions):
        return np.sum(enabled_transitions) == 0

    def get_graph_structure(self):
        if getattr(self, '_cached_edge_index', None) is not None:
            return self._cached_edge_index
        p_idx_minus, t_idx_minus = np.nonzero(self.LMINUS)
        src_minus = p_idx_minus
        dst_minus = t_idx_minus + self.N_P
        p_idx_plus, t_idx_plus = np.nonzero(self.LPLUS)
        src_plus = t_idx_plus + self.N_P
        dst_plus = p_idx_plus
        all_src = np.concatenate([src_minus, src_plus])
        all_dst = np.concatenate([dst_minus, dst_plus])
        edges_np = np.stack([all_src, all_dst], axis=0)
        edge_index = torch.tensor(edges_np, dtype=torch.long)
        self._cached_edge_index = edge_index
        
        return edge_index