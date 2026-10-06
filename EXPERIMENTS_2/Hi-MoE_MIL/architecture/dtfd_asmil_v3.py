import torch
from torch import nn

from architecture.mytransformer import ACMIL_MYMHA
from architecture.cls_transformer_tier2 import CLSTransformerTier2


class DTFD_ASMIL_v3(nn.Module):
    """
    DTFD-ASMIL v3: те саме, що v2, АЛЕ Tier 2 -- CLSTransformerTier2
    (стандартний CLS-токен, БЕЗ LSA-мосту) замість LSATransformerTier2.

    Tier 1 БЕЗ ЗМІН від v1/v2 -- випадкове розбиття на M псевдо-bags,
    ACMIL_MYMHA(pool_mode='moe'), n_drop завжди 0, дистиляція = конкатенація.
    """

    def __init__(self, conf, n_token=8, M=8, tier2_n_heads=4, tier2_D_inner=None, tier2_n_layers=2):
        super().__init__()
        self.M = M
        self.n_token = n_token

        self.tier1 = ACMIL_MYMHA(
            conf, n_token=n_token, n_drop=0, pool_mode='moe', use_ppeg=True,
        )

        tier2_D_feat = n_token * conf.D_inner
        tier2_D_inner = tier2_D_inner if tier2_D_inner is not None else conf.D_inner

        self.tier2 = CLSTransformerTier2(
            D_feat=tier2_D_feat, D_inner=tier2_D_inner, n_class=conf.n_class,
            n_heads=tier2_n_heads, n_layers=tier2_n_layers,
        )

    def _split_pseudo_bags(self, n_tiles, device):
        perm = torch.randperm(n_tiles, device=device)
        m_eff = min(self.M, n_tiles)
        if m_eff < self.M:
            print(f"!!! УВАГА: n_tiles={n_tiles} < M={self.M}, використовую {m_eff} псевдо-bags")
        return torch.chunk(perm, m_eff)

    def forward(self, input, coords=None):
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

        final_slide_pred, tier2_attn = self.tier2(tier2_input)  # tier2_attn завжди None тут

        return final_slide_pred, tier1_slide_preds, tier2_attn
