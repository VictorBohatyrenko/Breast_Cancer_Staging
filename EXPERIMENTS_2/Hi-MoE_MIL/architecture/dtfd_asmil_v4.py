import torch
from torch import nn

from architecture.mytransformer import ACMIL_MYMHA
from architecture.cumulative_lsa_tier2 import CumulativeLSATransformerTier2


class DTFD_ASMIL_v4(nn.Module):
    """
    DTFD-ASMIL v4: Tier1 без змін (random split, ACMIL_MYMHA+MoE, n_drop=0,
    конкатенація) + Tier2 = CumulativeLSATransformerTier2 (3 шари, кумулятивна
    LSA) + НАВЧЕНИЙ tier1_weight = sigmoid(blend_logit), той самий трюк, що
    вже показав себе корисним у moe_blend -- замість фіксованого гіперпараметра,
    модель сама знаходить баланс між tier1_loss і tier2_loss під час тренування.

    Ініціалізація blend_logit -> alpha≈0.1 (той самий "безпечний старт", що
    й у moe_blend, і те саме значення tier1_weight=0.1, яке ми підібрали
    вручну раніше -- тепер модель може від нього відхилитись, якщо це
    виявиться корисним).
    """

    def __init__(self, conf, n_token=8, M=8, tier2_n_heads=4, tier2_D_inner=None):
        super().__init__()
        self.M = M
        self.n_token = n_token

        self.tier1 = ACMIL_MYMHA(
            conf, n_token=n_token, n_drop=0, pool_mode='moe', use_ppeg=True,
        )

        tier2_D_feat = n_token * conf.D_inner
        tier2_D_inner = tier2_D_inner if tier2_D_inner is not None else conf.D_inner

        self.tier2 = CumulativeLSATransformerTier2(
            D_feat=tier2_D_feat, D_inner=tier2_D_inner, n_class=conf.n_class,
            n_heads=tier2_n_heads,
        )

        # НАВЧЕНИЙ tier1_weight: sigmoid(-2.2) ≈ 0.10, той самий старт, що
        # й наше вручну підібране значення -- модель може відійти від нього,
        # якщо градієнт покаже, що це корисно.
        self.tier1_weight_logit = nn.Parameter(torch.tensor(-2.2))

    def get_tier1_weight(self):
        return torch.sigmoid(self.tier1_weight_logit)

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

        final_slide_pred, tier2_attn = self.tier2(tier2_input)

        return final_slide_pred, tier1_slide_preds, tier2_attn
