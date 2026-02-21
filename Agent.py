import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from Config import Config
from Network import build_actor_critic_network


@dataclass
class Transition:
    state: Tuple[np.ndarray, Dict[str, np.ndarray]]
    action_mask: np.ndarray
    action: int
    reward: float
    next_state: Tuple[np.ndarray, Dict[str, np.ndarray]]
    done: bool
    old_log_prob: float
    value: float


class PPOAgent:
    def __init__(self):
        self.net = build_actor_critic_network().to(Config.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=Config.LEARNING_RATE)
        self.buffer: List[Transition] = []
        self.update_count: int = 0

        self.use_adaptive_entropy: bool = bool(
            getattr(Config, "ADAPTIVE_ENTROPY", False)
        )
        if self.use_adaptive_entropy:
            init_alpha = float(
                getattr(
                    Config,
                    "ENTROPY_COEF_INIT",
                    getattr(Config, "ENTROPY_COEF", 0.01),
                )
            )
            init_alpha = max(init_alpha, 1e-6)
            self.log_alpha = torch.nn.Parameter(
                torch.log(torch.tensor(init_alpha, dtype=torch.float32, device=Config.DEVICE))
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=float(getattr(Config, "ENTROPY_LR", 1e-3)),
            )
            self.target_entropy: float = float(
                getattr(Config, "TARGET_ENTROPY", 0.8)
            )
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None

    def _get_entropy_coef(self) -> float:
        init_c = float(getattr(Config, "ENTROPY_COEF", 0.01))
        min_c = float(getattr(Config, "ENTROPY_COEF_MIN", init_c))
        anneal_updates = int(getattr(Config, "ENTROPY_ANNEAL_UPDATES", 0))
        if anneal_updates > 0:
            frac = min(1.0, float(self.update_count) / float(anneal_updates))
            c = init_c + frac * (min_c - init_c)
        else:
            c = init_c
        return float(max(min_c, c))

    def _stack_context(self, ctx_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        start_m = np.stack([c["start_m"] for c in ctx_list]).astype(np.float32)
        goal_m = np.stack([c["goal_m"] for c in ctx_list]).astype(np.float32)
        op_times = np.stack([c["op_times"] for c in ctx_list]).astype(np.float32)
        lminus = np.stack([c["lminus"] for c in ctx_list]).astype(np.float32)
        lplus = np.stack([c["lplus"] for c in ctx_list]).astype(np.float32)
        t_exists = np.stack([c["t_exists"] for c in ctx_list]).astype(np.float32)
        return {
            "start_m": torch.tensor(start_m, dtype=torch.float32, device=Config.DEVICE),
            "goal_m": torch.tensor(goal_m, dtype=torch.float32, device=Config.DEVICE),
            "op_times": torch.tensor(op_times, dtype=torch.float32, device=Config.DEVICE),
            "lminus": torch.tensor(lminus, dtype=torch.float32, device=Config.DEVICE),
            "lplus": torch.tensor(lplus, dtype=torch.float32, device=Config.DEVICE),
            "t_exists": torch.tensor(t_exists, dtype=torch.float32, device=Config.DEVICE),
        }

    def select_action(self, state, action_mask, is_training: bool = True):
        dyn_state, ctx = state

        valid_actions = np.where(np.asarray(action_mask) > 0)[0]
        if valid_actions.size == 0:
            return 0, 0.0, 0.0

        self.net.eval()
        with torch.no_grad():
            s = torch.tensor(dyn_state, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            ctx_batch = self._stack_context([ctx])
            am = torch.tensor(action_mask, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)

            logits, values = self.net(s, ctx_batch, am)

            if is_training:
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                log_prob_t = dist.log_prob(action_t)
            else:
                action_t = logits.argmax(dim=1)
                log_prob_t = torch.tensor([0.0], device=Config.DEVICE)

            action = int(action_t.item())
            log_prob = float(log_prob_t.item())
            value = float(values.item())

        self.net.train()
        return action, log_prob, value

    def store(self, state, action_mask, action, reward, next_state, done, old_log_prob, value):
        self.buffer.append(
            Transition(
                state=state,
                action_mask=action_mask,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
                old_log_prob=float(old_log_prob),
                value=float(value),
            )
        )

    def compute_gae(self, next_value: float):
        gamma = float(Config.GAMMA)
        lam = float(Config.GAE_LAMBDA)

        advantages: List[float] = []
        gae = 0.0

        values = [t.value for t in self.buffer]
        values.append(float(next_value))

        for i in reversed(range(len(self.buffer))):
            t = self.buffer[i]
            delta = t.reward + gamma * values[i + 1] * (1.0 - float(t.done)) - values[i]
            gae = delta + gamma * lam * (1.0 - float(t.done)) * gae
            advantages.insert(0, gae)

        adv = torch.tensor(advantages, dtype=torch.float32, device=Config.DEVICE)
        vals = torch.tensor(values[:-1], dtype=torch.float32, device=Config.DEVICE)
        rets = adv + vals

        ep_start = 0
        for i in range(len(self.buffer)):
            if self.buffer[i].done or i == len(self.buffer) - 1:
                ep_end = i + 1
                ep_adv = adv[ep_start:ep_end]
                if ep_adv.numel() > 1:
                    adv[ep_start:ep_end] = (ep_adv - ep_adv.mean()) / (ep_adv.std() + 1e-8)
                else:
                    adv[ep_start:ep_end] = 0.0
                ep_start = ep_end

        return adv, rets

    def learn(self):
        if len(self.buffer) < int(Config.ROLLOUT_STEPS):
            return None

        self.update_count += 1

        last = self.buffer[-1]
        if not last.done:
            with torch.no_grad():
                dyn_next, ctx_next = last.next_state
                s_next = torch.tensor(dyn_next, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
                ctx_b = self._stack_context([ctx_next])
                logits_next, v_next = self.net(s_next, ctx_b, action_mask=None)
                next_value = float(v_next.item())
        else:
            next_value = 0.0

        advantages, returns = self.compute_gae(next_value=next_value)

        old_log_probs = torch.tensor([t.old_log_prob for t in self.buffer], dtype=torch.float32, device=Config.DEVICE)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.int64, device=Config.DEVICE)
        old_values = torch.tensor([t.value for t in self.buffer], dtype=torch.float32, device=Config.DEVICE)

        dyn_states = np.stack([t.state[0] for t in self.buffer]).astype(np.float32)
        ctx_states = [t.state[1] for t in self.buffer]
        action_masks = np.stack([t.action_mask for t in self.buffer]).astype(np.float32)

        states_t = torch.tensor(dyn_states, dtype=torch.float32, device=Config.DEVICE)
        ctx_batch = self._stack_context(ctx_states)
        action_masks_t = torch.tensor(action_masks, dtype=torch.float32, device=Config.DEVICE)

        row_ok = (action_masks_t.sum(dim=1) > 0)
        if row_ok.sum().item() < int(Config.PPO_BATCH_SIZE):
            self.buffer.clear()
            return None
        if not bool(row_ok.all().item()):
            states_t = states_t[row_ok]
            action_masks_t = action_masks_t[row_ok]
            actions = actions[row_ok]
            old_log_probs = old_log_probs[row_ok]
            old_values = old_values[row_ok]
            advantages = advantages[row_ok]
            returns = returns[row_ok]
            ctx_batch = {k: v[row_ok] for k, v in ctx_batch.items()}
        n = int(states_t.shape[0])
        idx = np.arange(n)

        loss_stats = {
            "policy": [],
            "value": [],
            "entropy": [],
            "total": [],
            "approx_kl": [],
            "clip_frac": [],
        }

        if self.use_adaptive_entropy and self.log_alpha is not None:
            entropy_coef = float(self.log_alpha.exp().item())
        else:
            entropy_coef = self._get_entropy_coef()

        self.net.train()
        for _ in range(int(Config.PPO_EPOCHS)):
            np.random.shuffle(idx)
            for start in range(0, n, int(Config.PPO_BATCH_SIZE)):
                end = min(n, start + int(Config.PPO_BATCH_SIZE))
                b = idx[start:end]

                mb_states = states_t[b]
                mb_ctx = {k: v[b] for k, v in ctx_batch.items()}
                mb_am = action_masks_t[b]
                mb_actions = actions[b]
                mb_adv = advantages[b]
                mb_ret = returns[b]
                mb_old_logp = old_log_probs[b]
                mb_old_v = old_values[b]

                logits, values = self.net(mb_states, mb_ctx, mb_am)

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - float(Config.CLIP_EPS_POLICY), 1.0 + float(Config.CLIP_EPS_POLICY)) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                approx_kl = (mb_old_logp - new_logp).mean()
                clip_eps = float(Config.CLIP_EPS_POLICY)
                clip_frac = ((ratio > (1.0 + clip_eps)) | (ratio < (1.0 - clip_eps))).float().mean()

                v_clip = mb_old_v + torch.clamp(values - mb_old_v, -float(Config.CLIP_EPS_VALUE), float(Config.CLIP_EPS_VALUE))
                v_loss1 = F.smooth_l1_loss(values, mb_ret, reduction="none")
                v_loss2 = F.smooth_l1_loss(v_clip, mb_ret, reduction="none")
                value_loss = float(Config.VALUE_LOSS_COEF) * torch.max(v_loss1, v_loss2).mean()

                entropy_loss = -float(entropy_coef) * entropy

                if self.use_adaptive_entropy and self.alpha_optimizer is not None:
                    alpha_loss = -self.log_alpha * (entropy.detach() - self.target_entropy)
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    with torch.no_grad():
                        min_c = max(float(getattr(Config, "ENTROPY_COEF_MIN", 1e-6)), 1e-6)
                        alpha = self.log_alpha.exp().clamp(min=min_c)
                        self.log_alpha.data.copy_(alpha.log())
                        entropy_coef = float(alpha.item())

                total_loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(Config.MAX_GRAD_NORM))
                self.optimizer.step()

                loss_stats["policy"].append(float(policy_loss.item()))
                loss_stats["value"].append(float(value_loss.item()))
                loss_stats["entropy"].append(float(entropy_loss.item()))
                loss_stats["total"].append(float(total_loss.item()))
                loss_stats["approx_kl"].append(float(approx_kl.item()))
                loss_stats["clip_frac"].append(float(clip_frac.item()))

        with torch.no_grad():
            _, v_pred = self.net(states_t, ctx_batch, action_masks_t)
            y = returns
            var_y = torch.var(y)
            if float(var_y.item()) < 1e-8:
                explained_var = 0.0
            else:
                explained_var = 1.0 - float(torch.var(y - v_pred).item()) / float(var_y.item())

        self.buffer.clear()
        out = {k: (sum(v) / max(1, len(v))) for k, v in loss_stats.items()}
        out["explained_var"] = float(explained_var)
        out["entropy_coef"] = float(entropy_coef)
        return out
