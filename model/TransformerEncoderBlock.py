from torch import nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .MLPBlock import MLPBlock

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim:int=768, num_heads:int=12, ff_dim:int=3072, dropout:float=0.1, atten_dropout:float=0.0)->None:
        super().__init__()

        self.msa = MultiHeadSelfAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=atten_dropout)
        
        self.mlp = MLPBlock(embed_dim=embed_dim,
                            ff_dim=ff_dim,
                            dropout=dropout)
        
    
    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x
