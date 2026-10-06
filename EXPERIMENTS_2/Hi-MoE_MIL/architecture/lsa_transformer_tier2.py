import torch
from torch import nn


class LSATransformerTier2(nn.Module):
    """
    Tier 2 для DTFD-ASMIL v2: справжній 2-шаровий Transformer encoder
    (Q/K/V + FFN + residual + LayerNorm, на відміну від спрощеного
    MutiHeadAttention без FFN, яким користується решта нашого коду),
    плюс Location-Sensitive Attention (LSA, Chorowski et al./Tacotron 2)
    між ДВОМА ШАРАМИ замість класичного "між кроками часу" -- бо тут
    M елементів (псевдо-bags) обробляються ОДНОЧАСНО, немає природного
    попереднього часового кроку.

    LSA-адаптація: attention-карта ПЕРШОГО шару (куди він "подивився" --
    матриця M x M) проходить через 1D-згортку і стає адитивним bias для
    attention ДРУГОГО шару. Ідея та сама, що в оригінальному LSA -- дати
    моделі "пам'ять" про вже переглянуте, щоб заохотити рівномірніше
    покриття всіх M елементів, а не повторний фокус на тому самому.
    """

    def __init__(self, D_feat, D_inner, n_class, n_heads=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        ffn_dim = D_inner * ffn_mult
        self.n_heads = n_heads
        self.D_inner = D_inner

        self.input_proj = nn.Linear(D_feat, D_inner)

        # --- Шар 1: звичайна self-attention ---
        self.attn1 = nn.MultiheadAttention(D_inner, n_heads, dropout=dropout, batch_first=True)
        self.norm1a = nn.LayerNorm(D_inner)
        self.ffn1 = nn.Sequential(
            nn.Linear(D_inner, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, D_inner)
        )
        self.norm1b = nn.LayerNorm(D_inner)

        # --- LSA-міст: attn1 (M x M) -> per-head адитивний bias для attn2 ---
        # 1D-згортка по вимірності "keys" (останній вимір M x M матриці),
        # окремо для кожного query-рядка -- локальне згладжування "куди
        # вже дивились", те саме, що робить LSA у Tacotron 2 над попередніми
        # attention-розподілами, тут -- над attention-розподілом шару 1.
        self.lsa_conv = nn.Conv1d(1, n_heads, kernel_size=3, padding=1)

        # --- Шар 2: self-attention + LSA-bias, потім FFN ---
        self.attn2 = nn.MultiheadAttention(D_inner, n_heads, dropout=dropout, batch_first=True)
        self.norm2a = nn.LayerNorm(D_inner)
        self.ffn2 = nn.Sequential(
            nn.Linear(D_inner, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, D_inner)
        )
        self.norm2b = nn.LayerNorm(D_inner)

        # --- Slide-рівневий pooling: навчений query "читає" M елементів ---
        self.pool_query = nn.Parameter(torch.zeros(1, 1, D_inner))
        nn.init.normal_(self.pool_query, std=1e-6)
        self.pool_attn = nn.MultiheadAttention(D_inner, n_heads, dropout=dropout, batch_first=True)

        self.classifier = nn.Linear(D_inner, n_class)

    def forward(self, x):
        """
        x: [1, M, D_feat] -- M дистильованих векторів з Tier 1.
        Повертає: (slide_pred [1, n_class], attn2_weights [1, M, M] -- діагностика)
        """
        B, M, _ = x.shape
        x = self.input_proj(x)  # [1, M, D_inner]

        # --- Шар 1 ---
        attn_out1, attn_w1 = self.attn1(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1a(x + attn_out1)
        x = self.norm1b(x + self.ffn1(x))

        # --- LSA bias з attn_w1 [B, M, M] -> [B*n_heads, M, M] адитивний attn_mask ---
        lsa_in = attn_w1.reshape(B * M, 1, M)
        lsa_bias = self.lsa_conv(lsa_in)  # [B*M, n_heads, M]
        lsa_bias = lsa_bias.reshape(B, M, self.n_heads, M).permute(0, 2, 1, 3)  # [B, n_heads, M, M]
        attn_mask = lsa_bias.reshape(B * self.n_heads, M, M)

        # --- Шар 2 (з LSA-bias як адитивна attn_mask) ---
        attn_out2, attn_w2 = self.attn2(
            x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=True
        )
        x = self.norm2a(x + attn_out2)
        x = self.norm2b(x + self.ffn2(x))

        # --- Slide-рівневий pooling ---
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, x, x, need_weights=False)
        pooled = pooled.squeeze(1)  # [B, D_inner]

        slide_pred = self.classifier(pooled)
        return slide_pred, attn_w2
