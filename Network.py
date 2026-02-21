import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple
from Config import Config


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if mask.dim() == x.dim() - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=dim, keepdim=False).clamp_min(1.0)
    num = (x * mask).sum(dim=dim, keepdim=False)
    return num / denom


class TransitionConditionedActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.P = int(Config.MAX_PLACES)
        self.T = int(Config.MAX_TRANSITIONS)
        self.K = int(Config.MAX_TOKENS_PER_PLACE)
        self.place_in_dim = 1 + self.K + 3 + 4

        self.place_emb_dim = int(getattr(Config, "AC_PLACE_EMB_DIM", 128))
        hidden = int(getattr(Config, "AC_HIDDEN_DIM", 256))
        dropout = float(getattr(Config, "AC_DROPOUT", 0.0))

        def mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 2):
            layers = []
            d = in_dim
            for _ in range(max(1, int(n_layers)) - 1):
                layers.append(nn.Linear(d, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        self.place_encoder = nn.Sequential(
            nn.Linear(self.place_in_dim, self.place_emb_dim),
            nn.LayerNorm(self.place_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.place_emb_dim, self.place_emb_dim),
            nn.LayerNorm(self.place_emb_dim),
            nn.GELU(),
        )

        trans_in_dim = self.place_emb_dim * 3
        self.actor_mlp = mlp(trans_in_dim, 1, hidden, n_layers=3)
        self.value_mlp = mlp(self.place_emb_dim, 1, hidden, n_layers=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        dyn_state: torch.Tensor,
        ctx: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dyn_state.dim() == 1:
            dyn_state = dyn_state.unsqueeze(0)

        B = dyn_state.shape[0]
        P, T, K = self.P, self.T, self.K
        m = dyn_state[:, 0:P]
        r_flat = dyn_state[:, P : P + P * K]
        r = r_flat.view(B, P, K)
        place_mask = dyn_state[:, P + P * K : P + P * K + P]
        start_m = ctx["start_m"]
        goal_m = ctx["goal_m"]
        op_times = ctx["op_times"]
        lminus = ctx["lminus"]  # [B,P,T]
        lplus = ctx["lplus"]    # [B,P,T]
        t_exists = ctx["t_exists"]
        is_res = ((start_m > 0) & (goal_m > 0) & (start_m == goal_m)).to(m.dtype)
        is_start = ((start_m > 0) & (goal_m == 0)).to(m.dtype)
        is_end = ((start_m == 0) & (goal_m > 0)).to(m.dtype)
        is_op = (op_times > 0).to(m.dtype)
        type_oh = torch.stack([is_res, is_start, is_end, is_op], dim=-1)  # [B,P,4]

        place_feat = torch.cat(
            [
                m.unsqueeze(-1),             # [B,P,1]
                r,                           # [B,P,K]
                start_m.unsqueeze(-1),       # [B,P,1]
                goal_m.unsqueeze(-1),        # [B,P,1]
                op_times.unsqueeze(-1),      # [B,P,1]
                type_oh,                     # [B,P,4]
            ],
            dim=-1,
        )
        place_feat = place_feat * place_mask.unsqueeze(-1)

        place_emb = self.place_encoder(place_feat)
        lminus_t = lminus.transpose(1, 2)
        lplus_t = lplus.transpose(1, 2)
        pre_count = lminus_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        post_count = lplus_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pre_emb = torch.bmm(lminus_t, place_emb) / pre_count
        post_emb = torch.bmm(lplus_t, place_emb) / post_count
        global_emb = _masked_mean(place_emb, place_mask, dim=1)
        global_expand = global_emb.unsqueeze(1).expand(-1, T, -1)
        trans_feat = torch.cat([pre_emb, post_emb, global_expand], dim=-1)
        logits = self.actor_mlp(trans_feat).squeeze(-1)
        logits = logits.masked_fill(t_exists <= 0, -1e9)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)

        values = self.value_mlp(global_emb).squeeze(-1)
        return logits, values


class _DenseBipartiteGATUpdate(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        ff_hidden: int,
        dropout: float,
        edge_beta_init: float,
    ):
        super().__init__()

        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = 1.0 / math.sqrt(max(1.0, float(self.head_dim)))

        proj_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(self.emb_dim, proj_dim, bias=False)
        self.k_pre = nn.Linear(self.emb_dim, proj_dim, bias=False)
        self.v_pre = nn.Linear(self.emb_dim, proj_dim, bias=False)
        self.k_post = nn.Linear(self.emb_dim, proj_dim, bias=False)
        self.v_post = nn.Linear(self.emb_dim, proj_dim, bias=False)
        self.out_proj = nn.Linear(2 * proj_dim, self.emb_dim, bias=True)
        self.edge_beta_pre = nn.Parameter(torch.ones(self.num_heads) * float(edge_beta_init))
        self.edge_beta_post = nn.Parameter(torch.ones(self.num_heads) * float(edge_beta_init))

        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(self.emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.emb_dim, int(ff_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(int(ff_hidden), self.emb_dim),
        )
        self.ln2 = nn.LayerNorm(self.emb_dim)

    def _attend(
        self,
        q: torch.Tensor,  # [B,Nq,H,D]
        k: torch.Tensor,  # [B,Nk,H,D]
        v: torch.Tensor,  # [B,Nk,H,D]
        edge_mask: torch.Tensor,  # [B,Nq,Nk] bool
        edge_weight: torch.Tensor,  # [B,Nq,Nk] float
        edge_beta: torch.Tensor,  # [H]
    ) -> torch.Tensor:
        # scores: [B,H,Nq,Nk]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale

        if edge_weight is not None:
            w = torch.clamp(edge_weight, min=0.0)
            bias = torch.log1p(w).unsqueeze(1)  # [B,1,Nq,Nk]
            scores = scores + edge_beta.view(1, -1, 1, 1) * bias

        # mask edges
        scores = scores.masked_fill(~edge_mask.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * edge_mask.unsqueeze(1).to(dtype=attn.dtype)
        attn = self.dropout(attn)

        msg = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        return msg

    def forward(
        self,
        query_emb: torch.Tensor,  # [B,Nq,E]
        key_emb: torch.Tensor,  # [B,Nk,E]
        mask_pre: torch.Tensor,  # [B,Nq,Nk] bool
        w_pre: torch.Tensor,  # [B,Nq,Nk]
        mask_post: torch.Tensor,  # [B,Nq,Nk] bool
        w_post: torch.Tensor,  # [B,Nq,Nk]
        query_exists: Optional[torch.Tensor],  # [B,Nq] float
    ) -> torch.Tensor:
        B, Nq, _ = query_emb.shape
        Nk = key_emb.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query_emb).view(B, Nq, H, D)
        k_pre = self.k_pre(key_emb).view(B, Nk, H, D)
        v_pre = self.v_pre(key_emb).view(B, Nk, H, D)
        k_post = self.k_post(key_emb).view(B, Nk, H, D)
        v_post = self.v_post(key_emb).view(B, Nk, H, D)

        msg_pre = self._attend(q, k_pre, v_pre, mask_pre, w_pre, self.edge_beta_pre)
        msg_post = self._attend(q, k_post, v_post, mask_post, w_post, self.edge_beta_post)

        msg = torch.cat([msg_pre, msg_post], dim=-1).reshape(B, Nq, 2 * H * D)
        delta = self.out_proj(msg)  # [B,Nq,E]

        if query_exists is not None:
            delta = delta * query_exists.unsqueeze(-1)

        x = self.ln1(query_emb + self.dropout(delta))
        ff = self.ff(x)
        if query_exists is not None:
            ff = ff * query_exists.unsqueeze(-1)
        x = self.ln2(x + self.dropout(ff))
        return x


class TransitionConditionedActorCriticGATNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.P = int(Config.MAX_PLACES)
        self.T = int(Config.MAX_TRANSITIONS)
        self.K = int(Config.MAX_TOKENS_PER_PLACE)

        self.place_in_dim = 1 + self.K + 3 + 4
        self.emb_dim = int(getattr(Config, "AC_PLACE_EMB_DIM", 128))
        hidden = int(getattr(Config, "AC_HIDDEN_DIM", 256))
        dropout = float(getattr(Config, "AC_DROPOUT", 0.0))

        heads = int(getattr(Config, "GAT_NUM_HEADS", 4))
        head_dim = int(getattr(Config, "GAT_HEAD_DIM", 32))
        layers = int(getattr(Config, "GAT_NUM_LAYERS", 2))
        ff_hidden = int(getattr(Config, "GAT_FF_HIDDEN", hidden))
        gat_dropout = float(getattr(Config, "GAT_DROPOUT", dropout))
        beta_init = float(getattr(Config, "GAT_EDGE_BETA_INIT", 0.1))

        if heads * head_dim != self.emb_dim:
            raise ValueError(
                f"GAT config mismatch: GAT_NUM_HEADS({heads})*GAT_HEAD_DIM({head_dim}) "
                f"must equal AC_PLACE_EMB_DIM({self.emb_dim})."
            )

        def mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 2):
            layers_ = []
            d = in_dim
            for _ in range(max(1, int(n_layers)) - 1):
                layers_.append(nn.Linear(d, hidden_dim))
                layers_.append(nn.LayerNorm(hidden_dim))
                layers_.append(nn.GELU())
                if dropout > 0:
                    layers_.append(nn.Dropout(dropout))
                d = hidden_dim
            layers_.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers_)

        self.place_encoder = nn.Sequential(
            nn.Linear(self.place_in_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU(),
        )

        self.trans_init = mlp(self.emb_dim * 3, self.emb_dim, hidden, n_layers=2)

        self.layers = int(max(0, layers))
        self.trans_updates = nn.ModuleList(
            [
                _DenseBipartiteGATUpdate(
                    emb_dim=self.emb_dim,
                    num_heads=heads,
                    head_dim=head_dim,
                    ff_hidden=ff_hidden,
                    dropout=gat_dropout,
                    edge_beta_init=beta_init,
                )
                for _ in range(self.layers)
            ]
        )
        self.place_updates = nn.ModuleList(
            [
                _DenseBipartiteGATUpdate(
                    emb_dim=self.emb_dim,
                    num_heads=heads,
                    head_dim=head_dim,
                    ff_hidden=ff_hidden,
                    dropout=gat_dropout,
                    edge_beta_init=beta_init,
                )
                for _ in range(self.layers)
            ]
        )

        self.actor_head = mlp(self.emb_dim * 2, 1, hidden, n_layers=3)
        self.value_head = mlp(self.emb_dim, 1, hidden, n_layers=2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dyn_state: torch.Tensor, ctx: Dict[str, torch.Tensor], action_mask: Optional[torch.Tensor] = None):
        if dyn_state.dim() == 1:
            dyn_state = dyn_state.unsqueeze(0)

        B = dyn_state.shape[0]
        P, T, K = self.P, self.T, self.K

        # dynamic parse
        m = dyn_state[:, 0:P]
        r_flat = dyn_state[:, P : P + P * K]
        r = r_flat.view(B, P, K)
        place_mask = dyn_state[:, P + P * K : P + P * K + P]
        start_m = ctx["start_m"]
        goal_m = ctx["goal_m"]
        op_times = ctx["op_times"]
        lminus = ctx["lminus"]  # [B,P,T]
        lplus = ctx["lplus"]    # [B,P,T]
        t_exists = ctx["t_exists"]
        is_res = ((start_m > 0) & (goal_m > 0) & (start_m == goal_m)).to(m.dtype)
        is_start = ((start_m > 0) & (goal_m == 0)).to(m.dtype)
        is_end = ((start_m == 0) & (goal_m > 0)).to(m.dtype)
        is_op = (op_times > 0).to(m.dtype)
        type_oh = torch.stack([is_res, is_start, is_end, is_op], dim=-1)  # [B,P,4]

        place_feat = torch.cat(
            [m.unsqueeze(-1), r,
             start_m.unsqueeze(-1), goal_m.unsqueeze(-1),
             op_times.unsqueeze(-1), type_oh],
            dim=-1,
        )
        place_feat = place_feat * place_mask.unsqueeze(-1)
        place_emb = self.place_encoder(place_feat)  # [B,P,E]

        global_emb = _masked_mean(place_emb, place_mask, dim=1)  # [B,E]
        global_expand = global_emb.unsqueeze(1).expand(-1, T, -1)  # [B,T,E]


        lminus_t = lminus.transpose(1, 2)  # [B,T,P]
        lplus_t = lplus.transpose(1, 2)
        pre_count = lminus_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        post_count = lplus_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pre_mean = torch.bmm(lminus_t, place_emb) / pre_count
        post_mean = torch.bmm(lplus_t, place_emb) / post_count
        trans_emb = self.trans_init(torch.cat([pre_mean, post_mean, global_expand], dim=-1))
        trans_emb = trans_emb * t_exists.unsqueeze(-1)
        place_exists = (place_mask > 0.0)
        trans_exists = (t_exists > 0.0)

        mask_pre_tp = (lminus.transpose(1, 2) > 0) & place_exists.unsqueeze(1) & trans_exists.unsqueeze(-1)
        mask_post_tp = (lplus.transpose(1, 2) > 0) & place_exists.unsqueeze(1) & trans_exists.unsqueeze(-1)
        w_pre_tp = lminus.transpose(1, 2)
        w_post_tp = lplus.transpose(1, 2)
        mask_pre_pt = (lminus > 0) & place_exists.unsqueeze(-1) & trans_exists.unsqueeze(1)
        mask_post_pt = (lplus > 0) & place_exists.unsqueeze(-1) & trans_exists.unsqueeze(1)
        w_pre_pt = lminus
        w_post_pt = lplus
        for i in range(self.layers):
            trans_emb = self.trans_updates[i](
                query_emb=trans_emb,
                key_emb=place_emb,
                mask_pre=mask_pre_tp,
                w_pre=w_pre_tp,
                mask_post=mask_post_tp,
                w_post=w_post_tp,
                query_exists=t_exists,
            )
            place_emb = self.place_updates[i](
                query_emb=place_emb,
                key_emb=trans_emb,
                mask_pre=mask_pre_pt,
                w_pre=w_pre_pt,
                mask_post=mask_post_pt,
                w_post=w_post_pt,
                query_exists=place_mask,
            )
            trans_emb = trans_emb * t_exists.unsqueeze(-1)
            place_emb = place_emb * place_mask.unsqueeze(-1)

        global_emb = _masked_mean(place_emb, place_mask, dim=1)
        global_expand = global_emb.unsqueeze(1).expand(-1, T, -1)
        actor_in = torch.cat([trans_emb, global_expand], dim=-1)
        logits = self.actor_head(actor_in).squeeze(-1)
        logits = logits.masked_fill(t_exists <= 0, -1e9)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)

        values = self.value_head(global_emb).squeeze(-1)
        return logits, values


def build_actor_critic_network() -> nn.Module:
    net_type = str(getattr(Config, "AC_NET_TYPE", "mlp")).lower()
    if net_type in ("gat", "attention"):
        return TransitionConditionedActorCriticGATNetwork()
    return TransitionConditionedActorCriticNetwork()

