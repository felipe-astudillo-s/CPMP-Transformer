import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim_multiplier, dropout):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim_multiplier * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim_multiplier * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, attn_mask=None):
        attn_out, _ = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
        )
        
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
    

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim_multiplier, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim_multiplier * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim_multiplier * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x