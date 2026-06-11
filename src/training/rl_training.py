import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from generation.adapters import EnrichedStackMatrix5DAdapter


class RLTrainer:
    """
    Fine-tunes a pre-trained CPMPTransformer using REINFORCE with:
      - Reward: -total_steps si resuelve, -(max_steps*2) si no resuelve
      - Baseline: EMA del reward promedio por batch
      - Regularización: mezcla opcional de pérdida supervisada (CE)
      - Rollout vectorizado: instancias del mismo tamaño se procesan en paralelo
      - Checkpointing: guarda/carga estado completo del entrenamiento
    """

    def __init__(
        self,
        model,
        device,
        learning_rate=1e-5,
        entropy_coeff=0.01,
        sl_coeff=0.3,
        sl_coeff_min=0.15,
        sl_coeff_decay=0.85,
        baseline_decay=0.99,
        max_steps=50,
        grad_clip=1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.entropy_coeff = entropy_coeff
        self.sl_coeff = sl_coeff
        self.sl_coeff_min = sl_coeff_min
        self.sl_coeff_decay = sl_coeff_decay
        self.baseline = 0.0
        self.baseline_initialized = False
        self.baseline_decay = baseline_decay
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.epoch = 0
        self.step = 0

    def rollout_batch(self, layouts):
        """
        Rollout vectorizado: procesa un batch de layouts en paralelo.
        Todos los layouts deben tener el mismo número de pilas y altura (mismo config).

        En cada timestep, los layouts activos (no resueltos) se encodifican juntos
        en un único forward pass → la GPU procesa batch_size instancias a la vez
        en lugar de una por una.

        Retorna lista de (log_probs, entropies, reward) por instancia.
        """
        n = len(layouts)
        layouts = [copy.deepcopy(l) for l in layouts]
        H = layouts[0].H
        S_len = len(layouts[0].stacks)

        # log_probs y entropies acumulados por instancia
        inst_log_probs = [[] for _ in range(n)]
        inst_entropies = [[] for _ in range(n)]
        active = list(range(n))

        self.model.train()

        for _ in range(self.max_steps):
            active = [i for i in active if not layouts[i].is_sorted()]
            if not active:
                break

            # Encodificar todos los layouts activos en un solo batch
            S_list, X_list = [], []
            for i in active:
                S_np, X_np = EnrichedStackMatrix5DAdapter.layout_2_vec(layouts[i], H)
                S_list.append(torch.tensor(S_np, dtype=torch.float32))
                X_list.append(torch.tensor(X_np, dtype=torch.float32))

            S_batch = torch.stack(S_list).to(self.device)  # (n_active, S, H, 2)
            X_batch = torch.stack(X_list).to(self.device)  # (n_active, S, 5)

            logits_batch = self.model(S_batch, X_batch)    # (n_active, S*(S-1))
            probs_batch = F.softmax(logits_batch, dim=-1)

            dist = torch.distributions.Categorical(probs_batch)
            actions = dist.sample()          # (n_active,)
            log_probs = dist.log_prob(actions)  # (n_active,)
            entropies = dist.entropy()          # (n_active,)

            for j, i in enumerate(active):
                inst_log_probs[i].append(log_probs[j])
                inst_entropies[i].append(entropies[j])

                idx = actions[j].item()
                src = idx // (S_len - 1)
                r = idx % (S_len - 1)
                dst = r if r < src else r + 1
                layouts[i].move(src, dst)

        results = []
        for i in range(n):
            solved = layouts[i].is_sorted()
            reward = -float(layouts[i].steps) if solved else -float(self.max_steps * 2)
            if inst_log_probs[i]:
                results.append((
                    torch.stack(inst_log_probs[i]),
                    torch.stack(inst_entropies[i]),
                    reward,
                ))
            else:
                results.append((None, None, reward))

        return results

    def train_step(self, rollout_results, sl_batch=None):
        """
        Realiza una actualización REINFORCE dado los resultados de rollout_batch.
        rollout_results: lista de (log_probs, entropies, reward) por instancia.
        sl_batch: batch del dataloader supervisado (opcional, para regularización).
        """
        all_mean_log_probs = []
        all_mean_entropies = []
        all_rewards = []

        for log_probs, entropies, reward in rollout_results:
            if log_probs is not None:
                all_mean_log_probs.append(log_probs.mean())
                all_mean_entropies.append(entropies.mean())
                all_rewards.append(reward)

        if not all_mean_log_probs:
            return {}

        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
        mean_reward = rewards.mean().item()

        # Actualizar baseline EMA
        if not self.baseline_initialized:
            self.baseline = mean_reward
            self.baseline_initialized = True
        else:
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1 - self.baseline_decay) * mean_reward
            )

        advantages = rewards - self.baseline
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(all_mean_log_probs)
        entropies_t = torch.stack(all_mean_entropies)

        pg_loss = -(log_probs_t * advantages.detach()).mean()
        entropy_loss = -self.entropy_coeff * entropies_t.mean()
        total_loss = pg_loss + entropy_loss

        # Regularización supervisada
        sl_loss_val = 0.0
        if sl_batch is not None and self.sl_coeff > 0:
            *inputs, y_batch = [t.to(self.device, non_blocking=True) for t in sl_batch]
            logits_sl = self.model(*inputs)
            y_norm = y_batch.float()
            y_norm = y_norm / (y_norm.sum(dim=1, keepdim=True) + 1e-8)
            sl_loss = F.cross_entropy(logits_sl, y_norm)
            total_loss = total_loss + self.sl_coeff * sl_loss
            sl_loss_val = sl_loss.item()

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.step += 1

        n_solved = sum(1 for r in all_rewards if r > -(self.max_steps * 2))

        return {
            "pg_loss": pg_loss.item(),
            "entropy": entropies_t.mean().item(),
            "sl_loss": sl_loss_val,
            "mean_reward": mean_reward,
            "baseline": self.baseline,
            "solved_ratio": n_solved / len(all_rewards),
            "sl_coeff": self.sl_coeff,
        }

    def train_epoch(self, layouts, sl_loader=None, batch_size=32, print_every=10):
        """
        Un epoch completo de entrenamiento RL.
        Agrupa layouts por (S_len, H) para que cada batch sea homogéneo
        y pueda usar rollout vectorizado sin padding.
        """
        self.model.train()
        self.epoch += 1

        # Agrupar por config (S_len, H) para rollout vectorizado
        groups = defaultdict(list)
        for layout in layouts:
            key = (len(layout.stacks), layout.H)
            groups[key].append(layout)

        # Construir lista de batches homogéneos mezclados entre configs
        all_batches = []
        for key, group in groups.items():
            idx = np.random.permutation(len(group))
            for start in range(0, len(group), batch_size):
                batch = [group[int(i)] for i in idx[start:start + batch_size]]
                all_batches.append(batch)

        # Mezclar el orden de los batches entre configs
        np.random.shuffle(all_batches)

        sl_iter = iter(sl_loader) if sl_loader is not None else None
        epoch_stats = {k: [] for k in ["pg_loss", "entropy", "sl_loss", "mean_reward", "solved_ratio"]}
        n_batches = len(all_batches)

        for batch_idx, batch in enumerate(all_batches):
            sl_batch = None
            if sl_iter is not None:
                try:
                    sl_batch = next(sl_iter)
                except StopIteration:
                    sl_iter = iter(sl_loader)
                    sl_batch = next(sl_iter)

            rollout_results = self.rollout_batch(batch)
            stats = self.train_step(rollout_results, sl_batch)

            for k in epoch_stats:
                if k in stats:
                    epoch_stats[k].append(stats[k])

            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == n_batches:
                recent = {k: float(np.mean(v[-print_every:])) if v else 0.0 for k, v in epoch_stats.items()}
                print(
                    f"  Epoch {self.epoch} | batch {batch_idx+1:3d}/{n_batches} | "
                    f"reward: {recent['mean_reward']:6.2f} | "
                    f"solved: {recent['solved_ratio']*100:5.1f}% | "
                    f"entropy: {recent['entropy']:.4f}",
                    flush=True,
                )

        # Decay sl_coeff una vez por epoch (no por batch)
        self.sl_coeff = max(self.sl_coeff * self.sl_coeff_decay, self.sl_coeff_min)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_stats.items()}

    def evaluate(self, layouts, max_steps=None):
        """
        Evalúa el modelo con decodificación greedy (determinística).
        Retorna solve_rate y mean_steps sobre los layouts dados.
        """
        max_steps = max_steps or self.max_steps
        self.model.eval()
        solved = 0
        steps_list = []

        with torch.no_grad():
            for layout in layouts:
                layout = copy.deepcopy(layout)
                H = layout.H
                S_len = len(layout.stacks)
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
                    r = action % (S_len - 1)
                    dst = r if r < src else r + 1
                    layout.move(src, dst)

                if layout.is_sorted():
                    solved += 1
                    steps_list.append(layout.steps)

        return {
            "solve_rate": solved / len(layouts),
            "mean_steps": float(np.mean(steps_list)) if steps_list else float(max_steps),
        }

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "baseline": self.baseline,
                "baseline_initialized": self.baseline_initialized,
                "sl_coeff": self.sl_coeff,
                "epoch": self.epoch,
                "step": self.step,
            },
            path,
        )
        print(f"✅ Checkpoint guardado → {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.baseline = ckpt["baseline"]
        self.baseline_initialized = ckpt.get("baseline_initialized", True)
        self.sl_coeff = ckpt["sl_coeff"]
        self.epoch = ckpt["epoch"]
        self.step = ckpt["step"]
        print(f"📂 Checkpoint cargado ← {path} (epoch {self.epoch}, step {self.step})")
