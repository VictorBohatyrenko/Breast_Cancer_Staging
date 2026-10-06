import torch
from torch import nn


class CumulativeLSATransformerTier2(nn.Module):
    """
    Tier 2 v4: КУМУЛЯТИВНА Location-Sensitive Attention -- ближче до
    оригінального механізму (Chorowski et al./Tacotron 2), де LSA
    накопичує ВСЮ історію попередніх attention-розподілів (кумулятивна
    сума), а не лише "останній крок" (як у v2, де шар2 бачив тільки шар1).

    3 шари замість 2:
      Шар 1: звичайна self-attention.
      Шар 2: self-attention + LSA-bias від cum_attn = attn_w1.
      Шар 3: self-attention + LSA-bias від cum_attn = attn_w1 + attn_w2
             (кумулятивна сума, той самий принцип, що в Tacotron 2).
    """

    def __init__(self, D_feat, D_inner, n_class, n_heads=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        ffn_dim = D_inner * ffn_mult
        self.n_heads = n_heads
        self.D_inner = D_inner

        self.input_proj = nn.Linear(D_feat, D_inner)

        def make_layer():
            return nn.ModuleDict({
                'attn': nn.MultiheadAttention(D_inner, n_heads, dropout=dropout, batch_first=True),
                'norm_a': nn.LayerNorm(D_inner),
                'ffn': nn.Sequential(
                    nn.Linear(D_inner, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, D_inner)
                ),
                'norm_b': nn.LayerNorm(D_inner),
            })

        self.layer1 = make_layer()
        self.layer2 = make_layer()
        self.layer3 = make_layer()

        # окремі LSA-згортки для bias шару 2 (з cum=attn1) і шару 3 (з cum=attn1+attn2)
        self.lsa_conv_2 = nn.Conv1d(1, n_heads, kernel_size=3, padding=1)
        self.lsa_conv_3 = nn.Conv1d(1, n_heads, kernel_size=3, padding=1)

        self.pool_query = nn.Parameter(torch.zeros(1, 1, D_inner))
        nn.init.normal_(self.pool_query, std=1e-6)
        self.pool_attn = nn.MultiheadAttention(D_inner, n_heads, dropout=dropout, batch_first=True)

        self.classifier = nn.Linear(D_inner, n_class)

    def _lsa_bias(self, cum_attn, conv, B, M):
        """cum_attn: [B, M, M] -> адитивний attn_mask [B*n_heads, M, M]."""
        lsa_in = cum_attn.reshape(B * M, 1, M)
        bias = conv(lsa_in)  # [B*M, n_heads, M]
        bias = bias.reshape(B, M, self.n_heads, M).permute(0, 2, 1, 3)  # [B, n_heads, M, M]
        return bias.reshape(B * self.n_heads, M, M)

    def _run_layer(self, layer, x, attn_mask=None):
        attn_out, attn_w = layer['attn'](x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=True)
        x = layer['norm_a'](x + attn_out)
        x = layer['norm_b'](x + layer['ffn'](x))
        return x, attn_w

    def forward(self, x):
        """x: [1, M, D_feat] -> (slide_pred [1, n_class], attn_w3 [1, M, M] діагностика)"""
        B, M, _ = x.shape
        x = self.input_proj(x)

        # Шар 1 -- без bias
        x, attn_w1 = self._run_layer(self.layer1, x)

        # Шар 2 -- bias від cum_attn = attn_w1
        bias2 = self._lsa_bias(attn_w1, self.lsa_conv_2, B, M)
        x, attn_w2 = self._run_layer(self.layer2, x, attn_mask=bias2)

        # Шар 3 -- bias від cum_attn = attn_w1 + attn_w2 (КУМУЛЯТИВНО)
        cum_attn = attn_w1 + attn_w2
        bias3 = self._lsa_bias(cum_attn, self.lsa_conv_3, B, M)
        x, attn_w3 = self._run_layer(self.layer3, x, attn_mask=bias3)

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, x, x, need_weights=False)
        pooled = pooled.squeeze(1)

        slide_pred = self.classifier(pooled)
        return slide_pred, attn_w3
