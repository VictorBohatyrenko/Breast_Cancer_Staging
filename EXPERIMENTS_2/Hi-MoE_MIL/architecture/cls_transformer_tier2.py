import torch
from torch import nn


class CLSTransformerTier2(nn.Module):
    """
    Tier 2 для DTFD-ASMIL v3: замінює LSA-механізм (v2) на стандартний
    CLS-токен підхід (BERT/ViT) -- простіше, надійніше, добре перевірена
    практика замість "саморобного" LSA-мосту між шарами.

    Навчений CLS-токен додається на початок послідовності з M дистильованих
    векторів, проходить через стандартний nn.TransformerEncoder (звичайні
    Q/K/V + FFN шари, вбудовані в PyTorch, без жодної кастомної логіки),
    і його фінальний стан (після всіх шарів) використовується НАПРЯМУ як
    slide-рівнева репрезентація для класифікатора -- той самий принцип,
    що [CLS]-токен у BERT чи class-токен у ViT.
    """

    def __init__(self, D_feat, D_inner, n_class, n_heads=4, ffn_mult=4, dropout=0.1, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(D_feat, D_inner)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_inner))
        nn.init.normal_(self.cls_token, std=1e-6)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_inner, nhead=n_heads, dim_feedforward=D_inner * ffn_mult,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Linear(D_inner, n_class)

    def forward(self, x):
        """
        x: [1, M, D_feat] -- M дистильованих векторів з Tier 1.
        Повертає: (slide_pred [1, n_class], None -- немає окремої attention-карти
                   для діагностики, nn.TransformerEncoder не віддає ваги напряму).
        """
        B = x.shape[0]
        x = self.input_proj(x)  # [1, M, D_inner]

        cls = self.cls_token.expand(B, -1, -1)  # [1, 1, D_inner]
        x = torch.cat([cls, x], dim=1)  # [1, M+1, D_inner] -- CLS завжди на позиції 0

        x = self.encoder(x)  # [1, M+1, D_inner]

        cls_out = x[:, 0, :]  # фінальний стан CLS-токена -- [1, D_inner]
        slide_pred = self.classifier(cls_out)

        return slide_pred, None
