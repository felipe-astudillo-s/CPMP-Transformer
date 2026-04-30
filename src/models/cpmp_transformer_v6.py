import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.transformer import Transformer
from generation.adapters import EnrichedStackMatrix4DAdapter

class CPMPTransformer(Transformer):
    # [MODIFICADO] H ya no es un requerimiento estricto para la inicialización de las capas,
    # pero lo mantenemos en el __init__ por si el Adapter superior (super) aún lo requiere.
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
        self.X_dim = X_dim
        self.C_dim = C_dim
        
        self.input_projection = nn.Linear(C_dim, d_model)
        self.empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        
        # [NUEVO] Parámetro aprendible para el Classification Token [CLS]
        # Dimensiones: (1, 1, d_model) para poder expandirlo fácilmente al tamaño del batch
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # [ELIMINADO] self.stack_summary_layer = nn.Linear(H * d_model, d_model)
        # Ya no necesitamos aplanar la matriz espacialmente.

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
        # [MODIFICADO] Ahora extraemos H_tensor dinámicamente de la entrada actual
        batch_size, S_len, H_tensor, C_dim = S.shape 
        device = S.device
        
        padding_mask = (S == -1).all(dim=-1) 
        x = self.input_projection(S.float())
        x = torch.where(padding_mask.unsqueeze(-1), self.empty_embed, x)
        
        # Aplanar para procesar cada pila independientemente en el intra_stack_attention
        x = x.view(batch_size * S_len, H_tensor, self.d_model) 
        
        # [NUEVO] Aplanar también la máscara para que coincida con las dimensiones de x
        padding_mask_flat = padding_mask.view(batch_size * S_len, H_tensor)
        
        # --- [NUEVO] LÓGICA DEL TOKEN [CLS] ---
        # 1. Expandimos el token [CLS] para que haya uno por cada pila en el batch
        cls_tokens = self.cls_token.expand(batch_size * S_len, -1, -1)
        
        # 2. Concatenamos el token [CLS] al principio de cada secuencia de contenedores
        # La nueva longitud de secuencia será (1 + H_tensor)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 3. Creamos una máscara para el token [CLS] llena de Falses (nunca debe ser ignorado)
        cls_mask = torch.zeros(batch_size * S_len, 1, dtype=torch.bool, device=device)
        
        # 4. Unimos la máscara del token [CLS] con la máscara de padding original
        full_padding_mask = torch.cat([cls_mask, padding_mask_flat], dim=1)
        
        # 5. Pasamos la información por el Transformer pasándole explícitamente la máscara
        x = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask) 
        
        # 6. En lugar de aplanar todo, extraemos ÚNICAMENTE el vector resultante del token [CLS] (índice 0)
        stack_vertical_info = x[:, 0, :] # Shape: (batch_size * S_len, d_model)
        
        # --- FIN LÓGICA [CLS] ---
        
        # Volvemos a darle la forma (batch_size, S_len, d_model) para procesar las pilas globalmente
        stack_vertical_info = stack_vertical_info.view(batch_size, S_len, self.d_model)
        
        # Combinación con Stacks Features (X)
        x_external_info = self.x_projection(X) 
        combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
        stack_embeddings = self.fusion_layer(combined) 
        stack_embeddings = self.fusion_norm(stack_embeddings)

        # Atención global entre pilas
        x_global = self.inter_stack_attention(stack_embeddings)
        
        q_origin = self.origin_proj(x_global)
        k_dest = self.dest_proj(x_global)
        
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model**0.5)

        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        
        is_origin_empty = (S == -1).all(dim=-1).all(dim=2) 
        
        # [NOTA DE INVESTIGACIÓN]
        # Si decides crear los tensores basándote estrictamente en H_real, esta línea sigue 
        # funcionando perfectamente. Si decides implementar el "Tight Bounding", tendrás 
        # que reescribir esta línea para extraer el límite desde el tensor X.
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