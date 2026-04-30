import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base.transformer import Transformer

# Importamos el NUEVO adaptador modular que acabamos de crear
from generation.adapters import EnrichedStackMatrix5DAdapter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class CPMPTransformer(Transformer):
    def __init__(self, H, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            EnrichedStackMatrix5DAdapter, # <--- Usamos el nuevo adaptador modular
            H=H,
            C_dim=C_dim,
            X_dim=X_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.X_dim = X_dim
        self.C_dim = C_dim
        
        self.input_projection = nn.Linear(C_dim, d_model)
        self.empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.x_projection = nn.Linear(X_dim, d_model)
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        self.inter_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers
        )
        
        self.origin_proj = nn.Linear(d_model, d_model)
        self.dest_proj = nn.Linear(d_model, d_model)

    def forward(self, S, X):
        batch_size, S_len, H_tensor, C_dim = S.shape 
        device = S.device
        
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
        
        stack_vertical_info = x[:, 0, :] 
        stack_vertical_info = stack_vertical_info.view(batch_size, S_len, self.d_model)
        
        x_external_info = self.x_projection(X) 
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        stack_embeddings = self.fusion_layer(combined) 
        stack_embeddings = self.fusion_norm(stack_embeddings)

        # ---> NUEVO: Creamos una máscara para identificar pilas enteramente "falsas" o rellenadas
        # Si todos los elementos de una pila son -1 (o si la altura es 0), es una pila fantasma
        stack_padding_mask = (S == -1).all(dim=-1).all(dim=-1) 

        # ---> NUEVO: Le pasamos esa máscara a la atención entre pilas
        x_global = self.inter_stack_attention(stack_embeddings, src_key_padding_mask=stack_padding_mask)
        
        q_origin = self.origin_proj(x_global)
        k_dest = self.dest_proj(x_global)
        
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model**0.5)

        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        is_origin_empty = (S == -1).all(dim=-1).all(dim=2) 
        
        # --- [MODIFICADO] LÓGICA DE TIGHT BOUNDING CON MATEMÁTICA EN TIEMPO REAL ---
        # Ahora el vector X tiene 5 dimensiones.
        # X[:, :, 3] -> Tamaño absoluto
        # X[:, :, 4] -> Altura máxima (H)
        
        size_absoluto = X[:, :, 3]
        H_max = X[:, :, 4]
        
        # PyTorch calcula los espacios libres eliminando la colinealidad en la entrada
        free_spaces = H_max - size_absoluto
        
        is_dest_full = (free_spaces <= 0)
        # ---------------------------------------------
        
        mask_origin = is_origin_empty.unsqueeze(2).expand(-1, -1, S_len)
        mask_dest = is_dest_full.unsqueeze(1).expand(-1, S_len, -1)
        
        invalid_action_mask = mask_diag | mask_origin | mask_dest
        
        logits_matrix = logits_matrix.masked_fill(invalid_action_mask, -1e4)
        
        indices = torch.arange(S_len, device=device)
        src_grid = indices.view(-1, 1).repeat(1, S_len)
        dst_grid = indices.view(1, -1).repeat(S_len, 1)
        mask_diag_flat = src_grid != dst_grid
        
        logits = logits_matrix[:, mask_diag_flat]

        return logits
