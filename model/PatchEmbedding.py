import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3, patch_size:int=16, embed_dim:int=768, img_size:int=224) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embed_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
        self.num_patches = (img_size * img_size) // patch_size**2

        self.class_embedding = nn.Parameter(torch.randn(1,1,embed_dim), requires_grad=True)

        self.position_embedding = nn.Parameter(torch.randn(1,self.num_patches+1, embed_dim), requires_grad=True)


    def forward(self, x):
        batch_size, image_res = x.shape[0], x.shape[-1]
        assert image_res % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_res}, patch_size: {self.patch_size}"
        
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patcher(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x

        return x
