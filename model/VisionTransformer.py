from torch import nn
from .TransformerEncoderBlock import TransformerEncoderBlock
from .PatchEmbedding import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_encoder_layers:int=12,
                 embed_dim:int=768,
                 ff_dim:int=3072,
                 num_heads:int=12,
                 dropout:float=0.1,
                 atten_dropout:float=0.0,
                 num_classes:int=10) -> None:
        """
        Vision Transformer model. a.k.a. ViT.
        
        Args:
            img_size (int): Image size (assuming square image).
            in_channels (int): Number of input channels.
            patch_size (int): Patch size.
            num_encoder_layers (int): Number of encoder layers.
            embed_dim (int): Embedding dimension.
            ff_dim (int): Feedforward dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            atten_dropout (float): Attention dropout rate.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        
        assert img_size % patch_size == 0,  f"Image size must be divisible by patch size, image: {img_size}, patch size: {patch_size}"

        self.embedding_dropout = nn.Dropout(p=dropout)

        self.patch_embed = PatchEmbedding(in_channels=in_channels,
                                          patch_size=patch_size,
                                          embed_dim=embed_dim,
                                          img_size=img_size)
        
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    ff_dim=ff_dim,
                                    dropout=dropout,
                                    atten_dropout=atten_dropout) for _ in range(num_encoder_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.embedding_dropout(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0])
        return x
