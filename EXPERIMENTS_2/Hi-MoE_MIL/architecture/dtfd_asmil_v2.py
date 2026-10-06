import torch
from torch import nn

from architecture.mytransformer import ACMIL_MYMHA
from architecture.lsa_transformer_tier2 import LSATransformerTier2


class DTFD_ASMIL_v2(nn.Module):
    """
    DTFD-ASMIL v2 -- підсумкова схема після серії експериментів:

    Tier 1 (БЕЗ ЗМІН від першої версії):
      - WSI ВИПАДКОВО (не просторово -- spatial-версія дала гірший
        результат, F1=0.516 проти 0.620 на random) розбивається на
        M=n_token псевдо-bags.
      - Кожен псевдо-bag незалежно проходить ACMIL_MYMHA(pool_mode='moe'),
        PPEG активний (реальні просторові тайли всередині одного
        псевдо-bag'а).
      - n_drop ЗАВЖДИ 0 у Tier 1 (потрібен фіксований розмір для
        конкатенації -- n_drop>0 дає змінну кількість branches у feats).
      - Дистиляція = КОНКАТЕНАЦІЯ всіх n_token branch-репрезентацій
        (НЕ max/усереднення, як в оригінальному DTFD-MIL).

    Tier 2 (НОВЕ -- LSATransformerTier2):
      - Замість спрощеного MutiHeadAttention (без FFN, один шар) --
        справжній 2-шаровий Transformer encoder з Q/K/V + FFN.
      - PPEG вимкнений (M елементів без реального просторового сусідства).
      - Location-Sensitive Attention між двома шарами: attention-карта
        шару 1 інформує шар 2 через навчену 1D-згортку.

    tier1_weight: РЕКОМЕНДОВАНЕ значення 0.05-0.1 (не 0, не 1.0) --
    діагностика показала: w=1.0 робить Tier1 "розумним" самостійно
    (F1=0.56), але ГІРШИЙ фінальний результат (0.620); w=0.0 -- Tier1
    "поламаний" сам по собі (F1=0.14), але КРАЩИЙ фінальний результат
    (0.646, хоч і не значуще на n=10). Мала ненульова вага -- компроміс:
    трохи стабілізує Tier1 без домінування над корисністю ознак для Tier 2.
    """

    def __init__(self, conf, n_token=8, M=8, tier2_n_heads=4, tier2_D_inner=None):
        super().__init__()
        self.M = M
        self.n_token = n_token

        # Tier 1: n_drop ЗАВЖДИ 0 (те саме обмеження, що й у v1 -- дивись
        # docstring вище і коментар у forward()).
        self.tier1 = ACMIL_MYMHA(
            conf, n_token=n_token, n_drop=0, pool_mode='moe', use_ppeg=True,
        )

        tier2_D_feat = n_token * conf.D_inner
        tier2_D_inner = tier2_D_inner if tier2_D_inner is not None else conf.D_inner

        self.tier2 = LSATransformerTier2(
            D_feat=tier2_D_feat, D_inner=tier2_D_inner, n_class=conf.n_class,
            n_heads=tier2_n_heads,
        )

    def _split_pseudo_bags(self, n_tiles, device):
        """Випадкове, рівномірне розбиття індексів тайлів на self.M груп
        (підтверджено емпірично кращим за просторове k-means-розбиття)."""
        perm = torch.randperm(n_tiles, device=device)
        m_eff = min(self.M, n_tiles)
        if m_eff < self.M:
            print(f"!!! УВАГА: n_tiles={n_tiles} < M={self.M}, використовую {m_eff} псевдо-bags")
        return torch.chunk(perm, m_eff)

    def forward(self, input, coords=None):
        """
        input: [1, N, D_feat] -- повний bag (усі тайли одного WSI).
        Повертає:
            final_slide_pred:  [1, n_class]
            tier1_slide_preds: [M, n_class]  -- для допоміжного (малий tier1_weight) loss
            tier2_attn: [1, M, M] -- attention другого шару Tier 2 (діагностика)
        """
        device = input.device
        n_tiles = input.shape[1]

        chunks = self._split_pseudo_bags(n_tiles, device)

        tier1_slide_preds = []
        distilled = []

        for idx in chunks:
            pseudo_bag = input[:, idx, :]
            sub_preds_j, slide_pred_j, attn_j, feats_j = self.tier1(
                pseudo_bag, coords=coords, return_feats=True
            )
            tier1_slide_preds.append(slide_pred_j)
            distilled.append(feats_j.reshape(1, -1))

        tier1_slide_preds = torch.cat(tier1_slide_preds, dim=0)  # [M, n_class]
        tier2_input = torch.cat(distilled, dim=0).unsqueeze(0)   # [1, M, n_token*D_inner]

        final_slide_pred, tier2_attn = self.tier2(tier2_input)

        return final_slide_pred, tier1_slide_preds, tier2_attn
