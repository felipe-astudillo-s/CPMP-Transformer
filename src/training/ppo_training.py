import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Categorical

from generation.adapters import EnrichedStackMatrix5DAdapter
from models.cpmp_transformer_v9 import CPMPTransformer


class PPOActorCritic(CPMPTransformer):
    """
    CPMPTransformer + value head para PPO actor-critic.
    El policy head (forward) es idéntico al padre.
    Usar forward_ppo(S, X) para obtener logits + valor en un solo forward pass.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = self.d_model
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )
        # Weights pequeños para que el valor no domine al inicio del entrenamiento
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward_ppo(self, S, X):
        """
        Forward pass completo que retorna (logits, value).
        Backbone idéntico a forward() pero además calcula V(s) via mean-pool del x_global.
        """
        batch_size, S_len, H_tensor, C_dim = S.shape
        device = S.device

        # --- Intra-stack attention (igual que CPMPTransformer.forward) ---
        padding_mask = (S == -1).all(dim=-1)
        x = self.input_projection(S.float())
        x = torch.where(padding_mask.unsqueeze(-1), self.empty_embed, x)
        x = x.view(batch_size * S_len, H_tensor, self.d_model)
        x = self.pos_encoder(x)

        padding_mask_flat = padding_mask.view(batch_size * S_len, H_tensor)
        cls_tokens = self.cls_token.expand(batch_size * S_len, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        cls_mask = torch.zeros(batch_size * S_len, 1, dtype=torch.bool, device=device)
        full_padding_mask = torch.cat([cls_mask, padding_mask_flat], dim=1)
        x = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask)
        stack_vertical_info = x[:, 0, :].view(batch_size, S_len, self.d_model)

        # --- Fusion + inter-stack attention ---
        x_external = self.x_projection(X)
        combined = torch.cat([stack_vertical_info, x_external], dim=-1)
        stack_emb = self.fusion_norm(self.fusion_layer(combined))

        stack_pad_mask = (S == -1).all(dim=-1).all(dim=-1)
        x_global = self.inter_stack_attention(stack_emb, src_key_padding_mask=stack_pad_mask)

        # --- Value head: mean-pool sobre pilas no-padding ---
        non_pad = ~stack_pad_mask                                       # (batch, S_len) bool
        x_pooled = (x_global * non_pad.unsqueeze(-1).float()).sum(dim=1) / \
                   non_pad.float().sum(dim=1, keepdim=True).clamp(min=1)
        value = self.value_head(x_pooled).squeeze(-1)                   # (batch,)

        # --- Policy head (igual que CPMPTransformer.forward) ---
        q_origin = self.origin_proj(x_global)
        k_dest   = self.dest_proj(x_global)
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model ** 0.5)

        mask_diag       = torch.eye(S_len, device=device).bool().unsqueeze(0)
        is_origin_empty = (S == -1).all(dim=-1).all(dim=2)
        free_spaces     = X[:, :, 4] - X[:, :, 3]
        is_dest_full    = (free_spaces <= 0)

        invalid = (mask_diag
                   | is_origin_empty.unsqueeze(2).expand(-1, -1, S_len)
                   | is_dest_full.unsqueeze(1).expand(-1, S_len, -1))
        logits_matrix = logits_matrix.masked_fill(invalid, -1e4)

        idx       = torch.arange(S_len, device=device)
        flat_mask = (idx.view(-1, 1) != idx.view(1, -1))
        logits    = logits_matrix[:, flat_mask]

        return logits, value


class PPOTrainer:
    """
    Fine-tunes PPOActorCritic usando PPO con:
      - Reward por paso: n_sorted_después - n_sorted_antes (señal densa)
      - Reward episódico: -total_steps si resuelve, -(max_steps*2) si no
      - GAE para estimación de ventajas
      - Objective clippeado PPO (clip_eps)
      - Value function loss (MSE)
      - Regularización SL opcional para evitar catastrofic forgetting
      - Rollout vectorizado (mismo grupo S×H en un solo forward pass por timestep)
      - Checkpointing completo
    """

    def __init__(
        self,
        model,
        device,
        learning_rate=1e-5,
        clip_eps=0.2,
        n_ppo_epochs=4,
        mini_batch_size=256,
        value_coeff=0.5,
        entropy_coeff=0.01,
        gamma=0.99,
        lam=0.95,
        step_reward_coeff=1.0,
        sl_coeff=0.3,
        sl_coeff_min=0.15,
        sl_coeff_decay=0.85,
        max_steps=100,
        grad_clip=1.0,
    ):
        self.model          = model.to(device)
        self.device         = device
        self.optimizer      = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.clip_eps       = clip_eps
        self.n_ppo_epochs   = n_ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coeff    = value_coeff
        self.entropy_coeff  = entropy_coeff
        self.gamma          = gamma
        self.lam            = lam
        self.step_reward_coeff = step_reward_coeff
        self.sl_coeff       = sl_coeff
        self.sl_coeff_min   = sl_coeff_min
        self.sl_coeff_decay = sl_coeff_decay
        self.max_steps      = max_steps
        self.grad_clip      = grad_clip
        self.epoch          = 0
        self.step           = 0

    @staticmethod
    def _n_sorted(layout):
        return len(layout.stacks) - layout.unsorted_stacks

    def rollout_batch(self, layouts):
        """
        Colecta trayectorias PPO para un batch homogéneo (mismo S_len y H).
        Usa torch.no_grad() — los gradientes se recomputan en ppo_update.

        Retorna lista de dicts con:
          states_S, states_X, actions, old_log_probs, advantages, returns, solved, total_reward
        """
        n = len(layouts)
        layouts = [copy.deepcopy(l) for l in layouts]
        H     = layouts[0].H
        S_len = len(layouts[0].stacks)

        traj_S        = [[] for _ in range(n)]
        traj_X        = [[] for _ in range(n)]
        traj_actions  = [[] for _ in range(n)]
        traj_lp       = [[] for _ in range(n)]   # log_probs (float)
        traj_vals     = [[] for _ in range(n)]   # values (float)
        traj_rewards  = [[] for _ in range(n)]
        traj_dones    = [[] for _ in range(n)]

        solved_status  = [False] * n
        n_sorted_prev  = [self._n_sorted(layouts[i]) for i in range(n)]
        active         = list(range(n))

        self.model.eval()

        with torch.no_grad():
            for _ in range(self.max_steps):
                still_active = [i for i in active if not layouts[i].is_sorted()]
                if not still_active:
                    break

                S_list, X_list = [], []
                for i in still_active:
                    S_np, X_np = EnrichedStackMatrix5DAdapter.layout_2_vec(layouts[i], H)
                    S_list.append(torch.tensor(S_np, dtype=torch.float32))
                    X_list.append(torch.tensor(X_np, dtype=torch.float32))

                S_batch = torch.stack(S_list).to(self.device)
                X_batch = torch.stack(X_list).to(self.device)

                logits_batch, vals_batch = self.model.forward_ppo(S_batch, X_batch)
                dist    = Categorical(logits=logits_batch)
                actions = dist.sample()
                lp      = dist.log_prob(actions)

                for j, i in enumerate(still_active):
                    # Guardar estado como tensor CPU (sin GPU memory durante rollout)
                    traj_S[i].append(S_list[j])
                    traj_X[i].append(X_list[j])
                    traj_actions[i].append(actions[j].item())
                    traj_lp[i].append(lp[j].item())
                    traj_vals[i].append(vals_batch[j].item())

                    idx = actions[j].item()
                    src = idx // (S_len - 1)
                    r   = idx % (S_len - 1)
                    dst = r if r < src else r + 1
                    layouts[i].move(src, dst)

                    # Reward por paso: progreso en pilas ordenadas
                    n_sorted_now = self._n_sorted(layouts[i])
                    step_r = self.step_reward_coeff * (n_sorted_now - n_sorted_prev[i])
                    n_sorted_prev[i] = n_sorted_now

                    done = layouts[i].is_sorted() or (layouts[i].steps >= self.max_steps)
                    if done:
                        solved_status[i] = layouts[i].is_sorted()
                        final_r = (
                            -float(layouts[i].steps)
                            if solved_status[i]
                            else -float(self.max_steps * 2)
                        )
                        traj_rewards[i].append(step_r + final_r)
                    else:
                        traj_rewards[i].append(step_r)
                    traj_dones[i].append(done)

                active = [i for i in still_active if not traj_dones[i][-1]]

        # Calcular GAE por instancia y empaquetar trayectorias
        trajectories = []
        for i in range(n):
            T = len(traj_rewards[i])
            if T == 0:
                continue

            rewards = traj_rewards[i]
            values  = traj_vals[i]
            dones   = traj_dones[i]

            # GAE: recorrido reverso
            advantages = []
            gae = 0.0
            for t in reversed(range(T)):
                next_v = values[t + 1] if (t + 1 < T and not dones[t]) else 0.0
                delta  = rewards[t] + self.gamma * next_v - values[t]
                gae    = delta + self.gamma * self.lam * (0.0 if dones[t] else gae)
                advantages.insert(0, gae)

            returns = [adv + val for adv, val in zip(advantages, values)]

            trajectories.append({
                'states_S':    traj_S[i],
                'states_X':    traj_X[i],
                'actions':     traj_actions[i],
                'old_log_probs': traj_lp[i],
                'advantages':  advantages,
                'returns':     returns,
                'solved':      solved_status[i],
                'total_reward': sum(rewards),
            })

        return trajectories

    def ppo_update(self, trajectories, sl_batch=None):
        """
        Actualización PPO sobre las trayectorias colectadas.
        Ejecuta n_ppo_epochs pasadas con mini-batches aleatorios.
        """
        if not trajectories:
            return {}

        # Aplanar todas las transiciones del batch (mismo S×H → stack directo)
        all_S        = torch.stack([s for t in trajectories for s in t['states_S']])
        all_X        = torch.stack([x for t in trajectories for x in t['states_X']])
        all_actions  = torch.tensor(
            [a for t in trajectories for a in t['actions']], dtype=torch.long)
        all_old_lp   = torch.tensor(
            [lp for t in trajectories for lp in t['old_log_probs']], dtype=torch.float32)
        all_adv      = torch.tensor(
            [a for t in trajectories for a in t['advantages']], dtype=torch.float32)
        all_returns  = torch.tensor(
            [r for t in trajectories for r in t['returns']], dtype=torch.float32)

        # Normalizar ventajas globalmente
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

        # Mover todo a GPU de una vez (tamaños pequeños, < 50 MB)
        all_S       = all_S.to(self.device)
        all_X       = all_X.to(self.device)
        all_actions = all_actions.to(self.device)
        all_old_lp  = all_old_lp.to(self.device)
        all_adv     = all_adv.to(self.device)
        all_returns = all_returns.to(self.device)

        N = len(all_actions)
        stats = defaultdict(list)

        self.model.train()

        for _ in range(self.n_ppo_epochs):
            perm = torch.randperm(N, device=self.device)

            for start in range(0, N, self.mini_batch_size):
                idx = perm[start:start + self.mini_batch_size]

                logits_new, vals_new = self.model.forward_ppo(all_S[idx], all_X[idx])
                dist_new = Categorical(logits=logits_new)
                new_lp   = dist_new.log_prob(all_actions[idx])
                entropy  = dist_new.entropy().mean()

                # Objective clippeado PPO
                ratio = (new_lp - all_old_lp[idx]).exp()
                adv_mb = all_adv[idx]
                L_clip = torch.min(
                    ratio * adv_mb,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb,
                ).mean()
                policy_loss = -L_clip

                # Value loss (MSE con los retornos GAE)
                value_loss = F.mse_loss(vals_new, all_returns[idx])

                total_loss = (policy_loss
                              + self.value_coeff * value_loss
                              - self.entropy_coeff * entropy)

                # Regularización SL
                if sl_batch is not None and self.sl_coeff > 0:
                    *inputs, y_batch = [t.to(self.device, non_blocking=True) for t in sl_batch]
                    logits_sl = self.model(*inputs)
                    y_norm = y_batch.float()
                    y_norm = y_norm / (y_norm.sum(dim=1, keepdim=True) + 1e-8)
                    sl_loss = F.cross_entropy(logits_sl, y_norm)
                    total_loss = total_loss + self.sl_coeff * sl_loss
                    stats['sl_loss'].append(sl_loss.item())

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.step += 1

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.item())
                stats['clip_ratio'].append((ratio > 1 + self.clip_eps).float().mean().item()
                                           + (ratio < 1 - self.clip_eps).float().mean().item())

        stats['mean_reward']   = [float(np.mean([t['total_reward'] for t in trajectories]))]
        stats['solved_ratio']  = [float(np.mean([t['solved'] for t in trajectories]))]

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train_epoch(self, layouts, sl_loader=None, batch_size=128, print_every=5):
        """
        Un epoch completo PPO.
        Agrupa layouts por (S_len, H), hace rollout vectorizado por grupo/batch,
        y luego actualiza con ppo_update.
        """
        self.epoch += 1

        # Agrupar por (S_len, H) para rollout vectorizado sin padding
        groups = defaultdict(list)
        for layout in layouts:
            groups[(len(layout.stacks), layout.H)].append(layout)

        all_batches = []
        for group in groups.values():
            idx = np.random.permutation(len(group))
            for start in range(0, len(group), batch_size):
                all_batches.append([group[int(i)] for i in idx[start:start + batch_size]])

        np.random.shuffle(all_batches)

        sl_iter = iter(sl_loader) if sl_loader is not None else None
        epoch_stats = defaultdict(list)
        n_batches = len(all_batches)

        for batch_idx, batch in enumerate(all_batches):
            sl_batch = None
            if sl_iter is not None:
                try:
                    sl_batch = next(sl_iter)
                except StopIteration:
                    sl_iter = iter(sl_loader)
                    sl_batch = next(sl_iter)

            trajectories = self.rollout_batch(batch)
            stats = self.ppo_update(trajectories, sl_batch)

            for k, v in stats.items():
                epoch_stats[k].append(v)

            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == n_batches:
                w = min(print_every, len(epoch_stats.get('mean_reward', [1])))
                recent = {k: float(np.mean(v[-w:])) for k, v in epoch_stats.items() if v}
                print(
                    f"  Epoch {self.epoch} | batch {batch_idx + 1:3d}/{n_batches} | "
                    f"reward: {recent.get('mean_reward', 0):7.2f} | "
                    f"solved: {recent.get('solved_ratio', 0) * 100:5.1f}% | "
                    f"ent: {recent.get('entropy', 0):.3f} | "
                    f"clip%: {recent.get('clip_ratio', 0) * 100:4.1f}% | "
                    f"vf_loss: {recent.get('value_loss', 0):.4f}",
                    flush=True,
                )

        # Decay sl_coeff una vez por epoch
        self.sl_coeff = max(self.sl_coeff * self.sl_coeff_decay, self.sl_coeff_min)

        return {k: float(np.mean(v)) for k, v in epoch_stats.items() if v}

    def evaluate(self, layouts, max_steps=None):
        """
        Evaluación greedy determinística (igual que RLTrainer.evaluate).
        Solo usa el policy head (forward), no el value head.
        """
        max_steps = max_steps or self.max_steps
        self.model.eval()
        solved = 0
        steps_list = []

        with torch.no_grad():
            for layout in layouts:
                layout  = copy.deepcopy(layout)
                H       = layout.H
                S_len   = len(layout.stacks)
                visited = set()

                for _ in range(max_steps):
                    if layout.is_sorted():
                        break
                    state_key = tuple(tuple(s) for s in layout.stacks)
                    if state_key in visited:
                        break
                    visited.add(state_key)

                    S_np, X_np = EnrichedStackMatrix5DAdapter.layout_2_vec(layout, H)
                    S_t = torch.tensor(S_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                    X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device).unsqueeze(0)

                    action = self.model(S_t, X_t)[0].argmax().item()
                    src = action // (S_len - 1)
                    r   = action % (S_len - 1)
                    dst = r if r < src else r + 1
                    layout.move(src, dst)

                if layout.is_sorted():
                    solved += 1
                    steps_list.append(layout.steps)

        return {
            'solve_rate': solved / len(layouts),
            'mean_steps': float(np.mean(steps_list)) if steps_list else float(max_steps),
        }

    def save_checkpoint(self, path):
        torch.save(
            {
                'model_state_dict':     self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'sl_coeff':             self.sl_coeff,
                'epoch':                self.epoch,
                'step':                 self.step,
                # Metadata de configuración
                'clip_eps':             self.clip_eps,
                'gamma':                self.gamma,
                'lam':                  self.lam,
            },
            path,
        )
        print(f'Checkpoint guardado → {path}')

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.sl_coeff = ckpt['sl_coeff']
        self.epoch    = ckpt['epoch']
        self.step     = ckpt['step']
        print(f'Checkpoint cargado ← {path} (epoch {self.epoch}, step {self.step})')
