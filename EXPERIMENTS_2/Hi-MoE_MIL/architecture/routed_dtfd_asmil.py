import torch
from torch import nn


class RoutedDTFDASMIL(nn.Module):
    """
    Uncertainty-Cascaded DTFD-ASMIL.

    Ідея (валідована на BRACS і C17, n=10 сідів кожен, McNemar p=0.018/0.004
    на flagged-підмножині -- дивись uni2h_fixes_disagreement.py):
        1. Дешевий backbone (ViT-S/DINO, D_feat=384) обробляє КОЖЕН слайд.
        2. Його Tier1 (M=8 випадкових псевдо-bags, ACMIL_MYMHA+MoE) дає
           M незалежних оцінок класу. Якщо вони узгоджені (мода домінує)
           -- слайд "простий", повертаємо дешевий вердикт як є.
        3. Якщо bags РОЗХОДЯТЬСЯ (disagreement >= threshold) -- слайд
           ескалюється до дорогого backbone (UNI2h, D_feat=1536), чий
           вердикт замінює дешевий.

    Емпірично: на flagged-підмножині (~15-27% слайдів залежно від
    датасету/порогу) дорогий backbone дає значущий приріст accuracy
    (BRACS: 0.46->0.58, C17: 0.53->0.66); на unflagged -- ефект слабкий
    або відсутній. Тобто дорогий backbone викликається лише там, де він
    реально потрібен, а не на всьому датасеті (economics: 73-85% слайдів
    взагалі не потребують дорогого проходу).

    ВАЖЛИВО: cheap_model і expensive_model -- це ДВІ окремо натреновані
    DTFD_ASMIL_v2 моделі (звичайним Step3_WSI_classification_DTFD_ASMIL_v2.py,
    --pretrain medical_ssl і --pretrain UNI2h відповідно, на попередньо
    порахованих фічах ОБОХ backbone для того самого датасету). Цей клас їх
    НЕ тренує -- лише компонує вже готові ваги в inference-каскад. Спільне
    end-to-end дотренування (наприклад, differentiable routing через
    Gumbel-softmax поверх disagreement) -- можлива майбутня робота, тут не
    реалізовано.
    """

    def __init__(self, cheap_model: nn.Module, expensive_model: nn.Module, threshold: float = 0.1):
        super().__init__()
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        self.threshold = threshold

    @staticmethod
    def _disagreement(tier1_slide_preds: torch.Tensor) -> torch.Tensor:
        """tier1_slide_preds: [M, n_class] логіти M псевдо-bags одного слайду.
        Повертає скаляр [0, 1] -- частку bags, чий argmax відрізняється від
        моди (найчастішого класу серед bags)."""
        probs = torch.softmax(tier1_slide_preds, dim=-1)
        argmax = probs.argmax(dim=-1)
        mode = torch.mode(argmax).values
        return (argmax != mode).float().mean()

    def forward(self, cheap_feats: torch.Tensor, expensive_feats: torch.Tensor,
                coords_cheap=None, coords_expensive=None):
        """
        cheap_feats:     [1, N, D_feat_cheap]     -- фічі дешевого backbone
        expensive_feats: [1, N, D_feat_expensive]  -- фічі ДОРОГОГО backbone,
                          ТОГО Ж слайду (обидва набори мають бути заздалегідь
                          порахованими -- .pt файли з обох екстракцій)

        Повертає:
            slide_pred:     [1, n_class] -- фінальний вердикт (дешевий або дорогий)
            used_expensive: bool         -- чи довелось ескалювати
            disagreement:   float        -- сирий tier1_disagreement (діагностика)
        """
        cheap_pred, tier1_slide_preds, _ = self.cheap_model(cheap_feats, coords=coords_cheap)
        disagreement = self._disagreement(tier1_slide_preds).item()

        if disagreement < self.threshold:
            return cheap_pred, False, disagreement

        expensive_pred, _, _ = self.expensive_model(expensive_feats, coords=coords_expensive)
        return expensive_pred, True, disagreement

    def forward_lazy(self, cheap_feats: torch.Tensor, expensive_feats_fn, coords_cheap=None):
        """
        Той самий каскад, але дорогі фічі рахуються ЛІНИВО -- через
        expensive_feats_fn() (напр. виклик UNI2h-екстракції наживо для
        цього одного слайду), а не заздалегідь. Саме це дає реальну
        економію компуту в проді: дорогий backbone взагалі не запускається
        для ~75-85% слайдів (unflagged), а не просто "рахується, але
        ігнорується", як у forward() вище (той придатний лише для offline
        аналізу/eval, де обидва набори фічей вже є на диску).

        expensive_feats_fn: () -> (expensive_feats, coords_expensive)
        """
        cheap_pred, tier1_slide_preds, _ = self.cheap_model(cheap_feats, coords=coords_cheap)
        disagreement = self._disagreement(tier1_slide_preds).item()

        if disagreement < self.threshold:
            return cheap_pred, False, disagreement

        expensive_feats, coords_expensive = expensive_feats_fn()
        expensive_pred, _, _ = self.expensive_model(expensive_feats, coords=coords_expensive)
        return expensive_pred, True, disagreement
