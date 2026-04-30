import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer
from generation.adapters import StackMatrix3DAdapter

class CPMPTransformer(Transformer):
    def __init__(self, H, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            StackMatrix3DAdapter,
            H=H,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.H = H
        
        # 1. Proyección inicial
        self.input_projection = nn.Linear(1, d_model)
        self.empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        
        # 2. Atención DENTRO de cada pila (Vertical)
        # Tratamos las celdas de una pila como una secuencia
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers
        )
        
        # 3. Resumen de la pila (H * d_model -> d_model)
        self.stack_summary_layer = nn.Linear(H * d_model, d_model)
        
        # 4. Atención ENTRE pilas (Horizontal/Global)
        self.inter_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers
        )
        
        # 5. Cross-Attention para pares (Origen -> Destino)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 6. Cabeza de puntuación final
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, S):
        # S: (B, num_stacks, H)
        batch_size, S_len, H = S.shape
        device = S.device
        
        # --- PASO 1: Proyección y Embeddings ---
        padding_mask = (S == -1)
        x = self.input_projection(S.unsqueeze(-1)) # (B, S_len, H, d_model)
        x = torch.where(padding_mask.unsqueeze(-1), self.empty_embed, x)
        
        # --- PASO 2: Intra-Stack Attention ---
        # Combinamos Batch y Pilas para procesar todas las verticales en paralelo
        x = x.view(batch_size * S_len, H, self.d_model) 
        x = self.intra_stack_attention(x) # (B*S_len, H, d_model)
        
        # --- PASO 3: Flatten y Resumen ---
        x = x.view(batch_size, S_len, H * self.d_model)
        stack_embeddings = self.stack_summary_layer(x) # (B, S_len, d_model)
        
        # --- PASO 4: Inter-Stack Attention ---
        # Las pilas intercambian información entre sí
        x_global = self.inter_stack_attention(stack_embeddings) # (B, S_len, d_model)
        
        # --- PASO 5: Generar combinaciones (Origen, Destino) ---
        indices = torch.arange(S_len, device=device)
        pairs = torch.cartesian_prod(indices, indices)
        mask_not_same = pairs[:, 0] != pairs[:, 1]
        valid_pairs = pairs[mask_not_same] # (num_pairs, 2)
        
        origin_idx = valid_pairs[:, 0]
        dest_idx = valid_pairs[:, 1]
        
        x_origin = x_global[:, origin_idx, :]
        x_dest = x_global[:, dest_idx, :]
        
        # --- PASO 6: Cross Attention y Score ---
        attn_output, _ = self.cross_attention(query=x_origin, key=x_dest, value=x_dest)
        logits = self.score_head(attn_output).squeeze(-1) # (B, num_pairs)

        # --- PASO 7: Máscara de Acciones Inválidas ---
        
        # 1. ¿La pila de origen está vacía? 
        # (True si todos los elementos de la pila son -1)
        is_origin_empty = (S == -1).all(dim=2) # (B, S_len)
        
        # 2. ¿La pila de destino está llena?
        # (True si no hay ningún -1 en la pila)
        is_dest_full = (S != -1).all(dim=2) # (B, S_len)

        # Expandimos a los pares generados
        # batch_origin_empty: (B, num_pairs)
        batch_origin_empty = is_origin_empty[:, origin_idx]
        # batch_dest_full: (B, num_pairs)
        batch_dest_full = is_dest_full[:, dest_idx]

        # Combinamos ambas restricciones: si el origen está vacío O el destino está lleno
        invalid_action_mask = batch_origin_empty | batch_dest_full

        # Aplicamos la máscara: asignamos un valor muy bajo para que la probabilidad sea 0
        logits = logits.masked_fill(invalid_action_mask, -1e9)

        return logits