import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, feature_dim, seq_len, in_channels, embed_dim, arrgmnt = 'rgbrgb', region_size = 5, dimension_order = 'TR'):
        super().__init__()
        self.seq_len = seq_len
        self.dimension_order = dimension_order
        self.in_channels = in_channels
        self.region_size = region_size
        self.arrgmnt = arrgmnt
        self.dimension_order = dimension_order
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        dim_1 = self.seq_len if self.dimension_order == 'TR' else self.region_size
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + dim_1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.filter_region(x)
        x = self.define_dimension_order(x)
        x = self.get_feature_arrangement(x)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

    def get_feature_arrangement(self, x):
        B = x.size(0)
        dim_1 = self.seq_len if self.dimension_order == 'TR' else self.region_size
        dim_2 = self.region_size if self.dimension_order == 'TR' else self.seq_len

        if self.arrgmnt == 'rgbrgb':
          x = x.reshape(
              B,
              dim_1,
              dim_2 * self.in_channels
              )
        else:

          x = x.view(B, dim_1, dim_2, self.in_channels)  # [B, seq_len, region_size, channels]
          x = x.permute(0, 1, 3, 2)  # [B, seq_len, channels, region_size]
          x = x.reshape(B, dim_1, self.in_channels * dim_2)
        return x

    def filter_region(self, x):
      region_5_filter = torch.tensor([1, 3, 4, 5, 7])
      if self.region_size == 5:
        x = x[:, :, region_5_filter, :]
      return x

    def define_dimension_order(self, x):
      B = x.size(0)
      if self.dimension_order == 'RT':
        x = x.permute(0, 2, 1, 3)
      return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class HeadlessVisionTransformer(nn.Module):
    def __init__(self, feature_dim, seq_len, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate, region_size, feat_arrgmnt, data_structure):
        super().__init__()
        self.patch_embed = PatchEmbedding(feature_dim, seq_len, in_channels, embed_dim, feat_arrgmnt, region_size, data_structure)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return cls_token