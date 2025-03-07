from torch import nn

class MLPBlock(nn.Module):
  def __init__(self, embed_dim:int=768, ff_dim:int=3072, dropout:float=0.1):
    super().__init__()
    
    self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

    self.mlp = nn.Sequential(
        nn.Linear(in_features=embed_dim,
                  out_features=ff_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=ff_dim,
                  out_features=embed_dim),
        nn.Dropout(p=dropout) 
    )
  
  def forward(self, x):
    return self.mlp(self.layer_norm(x))