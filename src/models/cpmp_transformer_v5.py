import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer
from generation.adapters import EnrichedStackMatrix4DAdapter

class CPMPTransformer(Transformer):
    def __init__(self, H, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            EnrichedStackMatrix4DAdapter,
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
        self.H = H
        self.X_dim = X_dim
        self.C_dim = C_dim
        
        self.input_projection = nn.Linear(C_dim, d_model)
        self.empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.stack_summary_layer = nn.Linear(H * d_model, d_model)

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
        batch_size, S_len, H, C_dim = S.shape
        device = S.device
        
        padding_mask = (S == -1).all(dim=-1) 
        x = self.input_projection(S.float())
        x = torch.where(padding_mask.unsqueeze(-1), self.empty_embed, x)
        
        x = x.view(batch_size * S_len, H, self.d_model) 
        x = self.intra_stack_attention(x) 
        
        x = x.view(batch_size, S_len, H, self.d_model)
        x_flat = x.view(batch_size, S_len, H * self.d_model)
        stack_vertical_info = self.stack_summary_layer(x_flat) 
        
        x_external_info = self.x_projection(X) 
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        stack_embeddings = self.fusion_layer(combined) 
        stack_embeddings = self.fusion_norm(stack_embeddings)

        x_global = self.inter_stack_attention(stack_embeddings)
        
        q_origin = self.origin_proj(x_global)
        k_dest = self.dest_proj(x_global)
        
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model**0.5)

        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        
        is_origin_empty = (S == -1).all(dim=-1).all(dim=2) 
        is_dest_full = ~(S == -1).all(dim=-1).any(dim=2) 
        
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