import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.attention import CrossAttentionBlock
from models.base.transformer import Transformer
from generation.adapters import GPIAdapter

class CPMPTransformer(Transformer):
    def __init__(self, d_model=64, nhead=8, num_layers=4, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            GPIAdapter,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.nhead = nhead

        # 1. Embeddings de entrada
        self.group_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Linear(1, d_model)
        self.stack_query_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.empty_slot_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Bloques de Atención
        # Bloque inferior (Contenedores -> Pilas)
        self.state_cross_attn = CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
        
        # Bloques medios (Refinamiento de estados potenciales)
        self.source_refiner = CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
        self.target_refiner = CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
        
        # 3. Capas de Proyección (Linear en tu diagrama)
        self.source_projector = nn.Linear(d_model, 1)
        self.target_projector = nn.Linear(d_model, 1)

    def forward(self, G, P, I, S, H):
        """
        G: (B, N, 1) - Group values
        P: (B, N, 1) - Tier positions
        I: (B, N, 1) - Stack indices
        S: (B, 1) - Number of stacks
        H: (B, 1) - Maximum stack height
        """
        device = G.device

        # 1. Extraer valores numéricos de los tensores S y H
        S = int(S[0].item())
        H = int(H[0].item())
        batch_size = G.size(0)

        # --- FASE 1: Construcción de la Rejilla (S x H) ---
        # Aseguramos (B, N, 1) para evitar dimensiones extrañas
        if G.dim() == 2: G = G.unsqueeze(-1)
        if P.dim() == 2: P = P.unsqueeze(-1)
        if I.dim() == 2: I = I.unsqueeze(-1)

        containers_rep = self.group_embedding(G.float()) + self.pos_embedding(P.float())

        # Inicializar rejilla (B, S, H, d) con tokens vacíos
        # Usamos .to(dtype) para evitar errores de Float vs BFloat16
        grid = self.empty_slot_token.to(containers_rep.dtype).expand(batch_size, S, H, self.d_model).clone()

        # Mapear contenedores a sus posiciones exactas
        idx_b = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, G.size(1))
        idx_s = I.squeeze(-1).long()
        idx_h = P.squeeze(-1).long() # 0 es el TOP según tu convención

        grid[idx_b, idx_s, idx_h] = containers_rep
        
        # Aplanar para la atención: (Batch, S * H, d)
        full_kv = grid.view(batch_size, S * H, self.d_model)

        # --- FASE 2: Máscaras Base ---
        # Creamos la máscara que limita a cada pila a ver solo sus H slots
        mask_indices = torch.arange(S * H, device=device).view(1, 1, S * H)
        s_starts = torch.arange(S, device=device).view(1, S, 1) * H
        s_ends = s_starts + H
        base_mask = (mask_indices < s_starts) | (mask_indices >= s_ends)

        # EXPANDIR AL BATCH: (Batch, S, S*H)
        # Esto soluciona el IndexError al indexar muestras más allá de la primera
        base_mask_batched = base_mask.expand(batch_size, -1, -1)

        # --- FASE 3: Representaciones de Estado ---
        stacks_init = self.stack_query_embedding.expand(batch_size, S, -1)

        # 3.1 Current States (Estado actual de cada pila)
        current_stacks = self.state_cross_attn(
            stacks_init, full_kv, full_kv, 
            attn_mask=base_mask_batched.repeat_interleave(self.nhead, dim=0)
        )

        # 3.2 Source States (Pila i sin su TOP en P=0)
        source_mask = base_mask_batched.clone()
        top_global_indices = torch.arange(S, device=device) * H
        source_mask[:, torch.arange(S), top_global_indices] = True
        
        source_states = self.state_cross_attn(
            stacks_init, full_kv, full_kv, 
            attn_mask=source_mask.repeat_interleave(self.nhead, dim=0)
        )

        # 3.3 Target States (Pila j + TOP de pila i)
        S_range = torch.arange(S, device=device)
        i_idx, j_idx = torch.meshgrid(S_range, S_range, indexing='ij')
        mask_valid = i_idx != j_idx
        indices_i = i_idx[mask_valid] 
        indices_j = j_idx[mask_valid] 
        num_actions = indices_j.size(0)
        
        target_queries = stacks_init[:, indices_j, :]
        
        # Creamos target_mask heredando el tamaño del batch: (Batch, Acciones, S*H)
        target_mask = base_mask_batched[:, indices_j, :].clone()
        
        # Broadcasting para indexación avanzada segura en 3D
        b_idx = torch.arange(batch_size, device=device).view(-1, 1)   # (B, 1)
        a_idx = torch.arange(num_actions, device=device).view(1, -1) # (1, Acciones)
        src_slot_idx = (indices_i * H).view(1, -1)                   # (1, Acciones)
        
        # Desbloqueamos el slot 0 de la pila origen para la pila destino
        target_mask[b_idx, a_idx, src_slot_idx] = False
        
        target_states = self.state_cross_attn(
            target_queries, full_kv, full_kv, 
            attn_mask=target_mask.repeat_interleave(self.nhead, dim=0)
        )

        # --- FASE 4: Proyección de Scores ---
        source_refined = self.source_refiner(source_states, current_stacks, current_stacks)
        target_refined = self.target_refiner(target_states, current_stacks, current_stacks)

        s_scores = self.source_projector(source_refined) # (B, S, 1)
        t_scores = self.target_projector(target_refined) # (B, S*(S-1), 1)
        
        # "Right Join": Sumar puntaje de origen i con destino j
        final_scores = s_scores[:, indices_i, :] + t_scores 
        final_scores = final_scores.squeeze(-1) # (B, S*(S-1))

        # --- FASE 5: Enmascarar movimientos inválidos ---
        # 1. Contar cuántos contenedores hay en cada pila actualmente (B, S)
        # I tiene los índices de pila para cada uno de los N contenedores.
        counts = torch.zeros(batch_size, S, device=device)
        # Squeeze para pasar de (B, N, 1) a (B, N) y contar apariciones de cada stack ID
        counts.scatter_add_(1, I.squeeze(-1).long(), torch.ones_like(I.squeeze(-1), dtype=torch.float32))

        # 2. Identificar qué pilas están vacías o llenas en cada muestra del batch
        is_empty = (counts == 0) # (B, S) - Booleano
        is_full = (counts == H)  # (B, S) - Booleano

        # 3. Mapear estas condiciones a la lista de acciones (B, Acciones)
        # indices_i son los orígenes de cada acción, indices_j son los destinos
        invalid_mask = is_empty[:, indices_i] | is_full[:, indices_j]

        # 4. Aplicar el "menos infinito" a los movimientos prohibidos
        final_scores.masked_fill_(invalid_mask, -1e-9)
        
        return final_scores