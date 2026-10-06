import torch
from torch import nn

from architecture.mytransformer import ACMIL_MYMHA
from utils.utils import Struct


class DTFD_ASMIL(nn.Module):
    """
    DTFD-MIL стиль (Zhang et al., CVPR 2022) поєднаний з нашим ASMIL+MoE на
    ОБОХ рівнях, замість звичайного attention-MIL, яким користується
    оригінальний DTFD-MIL.

    Tier 1: WSI випадково, рівномірно розбивається на M псевдо-bags. Кожен
    псевдо-bag незалежно проходить через ACMIL_MYMHA(pool_mode='moe') --
    дає M ОКРЕМИХ slide-рівневих передбачень (густа супервізія -- вирішує
    проблему "sub_preds ніколи не отримують прямого loss", яку ми бачили
    в звичайному ASMIL) і M наборів branch-репрезентацій (feats), отримані
    ОДНИМ forward-викликом (return_feats=True) -- важливо, бо --n_drop
    випадково відкидає гілки на кожному виклику; два окремі виклики дали б
    неузгоджені feats/slide_pred з різних підмножин гілок.

    Дистиляція: з кожного псевдо-bag'а конкатенуємо ВСІ n_token
    branch-репрезентації (feats) в один вектор розмірності n_token*D_inner
    -- НЕ одне усереднене/max-значення, як у оригінальному DTFD-MIL (MaxS/AFS).

    Tier 2: ОКРЕМА (не спільні ваги) модель ACMIL_MYMHA(pool_mode='moe'),
    БЕЗ PPEG (M дистильованих векторів не мають реального просторового
    сусідства -- вони з випадкового розбиття, не сітки), яка розглядає ці
    M дистильованих векторів як "тайли" свого власного MIL-bag'а, і видає
    фінальне slide-рівневе передбачення.
    """

    def __init__(self, conf, n_token=8, n_drop=0, M=8,
                 tier2_D_inner=None, tier2_n_token=None):
        super().__init__()
        self.M = M
        self.n_token = n_token

        # Tier 1: той самий формат входу, що й звичайний ASMIL (D_feat=conf.D_feat),
        # PPEG активний -- реальні просторові тайли з WSI.
        # ВАЖЛИВО: Tier 1 ЗАВЖДИ з n_drop=0, незалежно від переданого n_drop.
        # ACMIL_MYMHA.forward() при n_drop>0 і self.training=True виключає
        # випадкові n_drop гілок зі списку feats -- розмір конкатенованого
        # вектора був би різним щоразу (і різним для кожного псевдо-bag'а),
        # а Tier 2 очікує ФІКСОВАНИЙ розмір входу (n_token*D_inner). Якщо
        # захочете n_drop-регуляризацію тут -- знадобиться окрема правка
        # ACMIL_MYMHA (padding нулями замість виключення), не зроблено поки.
        if n_drop != 0:
            print(f"!!! УВАГА: n_drop={n_drop} проігноровано для Tier 1 в DTFD_ASMIL "
                  f"(несумісно з фіксованим розміром дистиляції), використовую n_drop=0")
        self.tier1 = ACMIL_MYMHA(
            conf, n_token=n_token, n_drop=0, pool_mode='moe', use_ppeg=True,
        )

        # Tier 2: вхід -- конкатенація n_token branch-репрезентацій з Tier 1.
        tier2_D_feat = n_token * conf.D_inner
        tier2_D_inner = tier2_D_inner if tier2_D_inner is not None else conf.D_inner
        tier2_n_token = tier2_n_token if tier2_n_token is not None else n_token

        tier2_conf = Struct(
            D_feat=tier2_D_feat,
            D_inner=tier2_D_inner,
            n_class=conf.n_class,
        )
        # PPEG вимкнений -- М псевдо-bags без реального просторового сусідства.
        self.tier2 = ACMIL_MYMHA(
            tier2_conf, n_token=tier2_n_token, n_drop=0, pool_mode='moe', use_ppeg=False,
        )

    def _split_pseudo_bags(self, n_tiles, device):
        """Випадкове, рівномірне (за розміром) розбиття індексів тайлів на
        self.M груп. torch.chunk дає M груп розміром floor(n/M) чи ceil(n/M)."""
        perm = torch.randperm(n_tiles, device=device)
        m_eff = min(self.M, n_tiles)
        if m_eff < self.M:
            # вкрай малоймовірно для реальних WSI (тисячі тайлів), але про всяк
            # випадок: не даємо впасти, просто менше псевдо-bags цього разу.
            print(f"!!! УВАГА: n_tiles={n_tiles} < M={self.M}, використовую {m_eff} псевдо-bags")
        return torch.chunk(perm, m_eff)

    def forward(self, input, coords=None):
        """
        input: [1, N, D_feat] -- повний bag (усі тайли одного WSI).
        Повертає:
            final_slide_pred:  [1, n_class]  -- фінальне передбачення (Tier 2)
            tier1_slide_preds: [M, n_class]  -- передбачення КОЖНОГО псевдо-bag'а
                                                 (Tier 1), для допоміжного/густого loss
            tier2_attn: attention Tier 2 (діагностика)
        """
        device = input.device
        n_tiles = input.shape[1]

        chunks = self._split_pseudo_bags(n_tiles, device)

        tier1_slide_preds = []
        distilled = []

        for idx in chunks:
            pseudo_bag = input[:, idx, :]  # [1, n_i, D_feat]

            # ОДИН виклик forward з return_feats=True -- узгоджені slide_pred_j
            # і feats_j з тим самим (якщо --n_drop>0) патерном відкидання гілок.
            sub_preds_j, slide_pred_j, attn_j, feats_j = self.tier1(
                pseudo_bag, coords=coords, return_feats=True
            )

            tier1_slide_preds.append(slide_pred_j)
            distilled.append(feats_j.reshape(1, -1))  # конкатенація n_token гілок -> [1, n_token*D_inner]

        tier1_slide_preds = torch.cat(tier1_slide_preds, dim=0)  # [M, n_class]
        tier2_input = torch.cat(distilled, dim=0).unsqueeze(0)  # [1, M, n_token*D_inner]

        sub_preds2, final_slide_pred, tier2_attn = self.tier2(tier2_input, coords=None)

        return final_slide_pred, tier1_slide_preds, tier2_attn