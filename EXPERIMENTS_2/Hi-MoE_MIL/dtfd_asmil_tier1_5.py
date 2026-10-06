#!/usr/bin/env python
"""
СКЕЛЕТ нової архітектури: Tier1.5 -- bag-рівневий каскад з контекстом.

ВАЖЛИВО: це не composición двох готових моделей (як RoutedDTFDASMIL), а
СПРАВЖНІЙ НОВИЙ ТРЕНОВАНИЙ МОДУЛЬ. Перш ніж вкладати час у тренування,
запусти `validate_bag_level_routing.py` -- якщо там немає значущого p
(bag-рівневий сигнал vs random-вибір bags), тренувати Tier1.5 зарано.

Ідея (як ти її описав):
    1. Cheap Tier1 обробляє всі M=8 bags як завжди -> per-bag предикшени +
       "дистильовані" фічі (n_token*D_inner_cheap кожен).
    2. Обираємо top-K НАЙНЕВПЕВНЕНІШИХ bags (за власною ентропією їхнього
       cheap-передбачення -- те, що ми вже перевірили в
       validate_bag_level_routing.py).
    3. Для ЦИХ K bags: беремо EXPENSIVE (UNI2h) фічі ТИХ САМИХ тайлів і
       проганяємо через НОВИЙ модуль Tier1_5Module. Він не просто
       обробляє bag ізольовано -- отримує ще й "дистильований контекст"
       (агреговані фічі решти, ВПЕВНЕНИХ bags), щоб не судити наосліп.
    4. Замінюємо cheap-фічі/предикшени тих K bags на виходи Tier1.5.
    5. Уся послідовність з M елементів (частина від cheap Tier1, частина
       від Tier1.5) іде в ТОЙ САМИЙ, вже натренований Tier2 -- без змін.

ЗВІРЕНО З РЕАЛЬНИМ architecture/mytransformer.py (ACMIL_MYMHA), 2026-09-28:
    - feats_j (те, що ACMIL_MYMHA повертає як `return_feats=True` четвертий
      елемент): підтверджено -- [1, n_token, D_inner] -> reshape(1,-1) дає
      [1, n_token*D_inner]. Моя початкова здогадка була правильна.
    - Агрегація branches у bag-рівневий предикшен (pool_mode='moe', яким
      користується DTFD_ASMIL_v2): НЕ просте середнє, а НАВЧЕНИЙ MoE-гейт:
      bag_summary = feats.mean(dim=1) ("branch_mean"), gate =
      softmax(Linear(bag_summary)), slide_pred = (gate * sub_preds).sum().
      Моя початкова версія Tier1_5Module брала наївне mean() -- ВИПРАВЛЕНО
      нижче на такий самий gated-механізм, щоб bag_pred_out Tier1.5 був
      семантично сумісний з тим, як це рахує cheap Tier1.
    - Кожен з n_token branches у ACMIL_MYMHA має ОКРЕМІ ваги уваги
      (nn.ModuleList з n_token незалежних MutiHeadAttention-модулів, кожен
      зі своїм одним query-вектором q[:,i]) -- не один спільний
      multi-query attention. ВИПРАВЛЕНО нижче.
    - coords: приймається як параметр forward(), але в показаному коді
      НІДЕ не використовується всередині -- позиційне кодування (PPEG)
      працює без явних координат, через модуль PPEG (у
      modules/emb_position.py, якого я не бачив). Tier1_5Module НЕ включає
      PPEG -- свідомо, а не через недогляд; якщо PPEG виявиться важливим
      для якості, це окремий крок додати його за зразком
      `self.pos_layer = PPEG(dim=D_inner)`; `input = self.pos_layer(input)`.
    - n_token тут -- те саме, що M (кількість псевдо-bags)? НІ. У
      DTFD_ASMIL_v2.__init__ вони передаються ОКРЕМО: n_token -- к-сть
      "віток" ВСЕРЕДИНІ ОДНОГО bag'а (ACMIL branches), M -- к-сть bags.
      Tier1_5Module відтворює ту саму n_token-branch структуру для
      узгодженості розмірів з Tier2, не тому, що M і n_token рівні.

ТРЕНУВАННЯ (поки не реалізовано, лише архітектура):
    - Заморозити cheap_model.tier1 і tier2 (уже натреновані); тренувати
      лише Tier1_5Module (+ невеlikі проекційні шари) на тому самому
      slide-рівневому лоссі (CrossEntropy на final_slide_pred), можливо +
      малий допоміжний лосс на власний sub_pred Tier1.5, аналогічно
      tier1_weight=0.05-0.1 у поточній моделі.
    - Це вимагає інтеграції в Step3_WSI_classification_*.py -- окремий
      крок, не тут.
"""
import math

import torch
from torch import nn


class ContextAggregator(nn.Module):
    """Замість грубого `others.mean()` (2026-10-06).

    Простe усереднення робить контекст ОДНАКОВИМ для будь-якого flagged bag
    і дає однакову вагу впевненим та майже невпевненим сусідам. Тут:
      - query  = дешеві фічі САМОГО flagged bag'а (що саме мені треба
                 уточнити?),
      - keys/values = дешеві фічі ІНШИХ bags + їхня впевненість
                 (conf = 1 - entropy/log(n_class)) як додатковий сигнал,
      - cross-attention сам вирішує, які сусіди релевантні саме цьому bag'у.
    Вихід -- [1, d_ctx], йде у FiLM-кондиціонування Tier1_5Module.
    """

    def __init__(self, D_in, d_ctx=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Sequential(nn.LayerNorm(D_in), nn.Linear(D_in, d_ctx))
        self.kv_proj = nn.Sequential(nn.LayerNorm(D_in), nn.Linear(D_in, d_ctx))
        self.conf_proj = nn.Linear(1, d_ctx)
        self.attn = nn.MultiheadAttention(d_ctx, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_ctx)

    def forward(self, own_feat, others_feats, others_conf):
        """own_feat: [1, D_in]; others_feats: [1, K, D_in]; others_conf: [1, K]"""
        q = self.q_proj(own_feat).unsqueeze(1)  # [1, 1, d_ctx]
        kv = self.kv_proj(others_feats) + self.conf_proj(others_conf.unsqueeze(-1))  # [1, K, d_ctx]
        ctx, _ = self.attn(q, kv, kv, need_weights=False)  # [1, 1, d_ctx]
        return self.out_norm(ctx.squeeze(1))  # [1, d_ctx]


class Tier1_5Module(nn.Module):
    """
    Обробляє ОДИН "проблемний" bag дорогими (UNI2h) фічами його тайлів,
    з урахуванням дистильованого контексту решти bags слайду.

    Вхід:
        exp_tile_feats: [1, n_i, D_feat_exp] -- UNI2h-фічі тайлів ЦЬОГО bag'а
        context_vec:    [1, D_context] -- агрегований (напр. усереднений)
                         дистильований вектор ІНШИХ (впевнених) bags слайду,
                         уже спроєктований до D_context (див. DTFD_ASMIL_v2_5)

    Вихід:
        feats_out:  [1, n_token * D_inner_cheap] -- замінник feats_j для
                    цього bag'а, сумісний за розміром із тим, що очікує
                    вже натренований Tier2 (n_token*D_inner_cheap -- той
                    самий D_feat, з яким будувався tier2_D_feat у
                    DTFD_ASMIL_v2)
        sub_pred_out: [n_token, n_class] -- по-branch предикшени, для
                    допоміжного лоссу (аналог sub_preds_j у ACMIL_MYMHA)
        bag_pred_out: [1, n_class] -- предикшен bag'а в цілому (для
                    діагностики / порівняння з cheap-версією цього bag'а)
    """

    def __init__(self, D_feat_exp, D_inner_exp, D_context, n_token, D_inner_cheap,
                 n_class, n_heads=4, dropout=0.1, context_mode='attn', d_ctx=128):
        super().__init__()
        self.n_token = n_token
        self.D_inner_exp = D_inner_exp
        self.context_mode = context_mode

        # --- Проекція UNI2h-тайлів у робочий простір (спрощений аналог
        # DimReduction(D_feat, D_inner, numLayer_Res=0) з ACMIL_MYMHA;
        # точну внутрішню структуру DimReduction не бачив, тому Linear+GELU
        # як функціональний еквівалент "зменшення розмірності + нелінійність"). ---
        self.tile_proj = nn.Sequential(
            nn.Linear(D_feat_exp, D_inner_exp), nn.GELU(), nn.Dropout(dropout),
        )

        # --- FiLM-кондиціонування контекстом: масштаб+зсув застосовуються
        # ДО кожного тайла bag'а, ПЕРЕД branch-attention -- bag "бачить"
        # решту слайду ще до того, як формує власну думку. Це і є НОВА
        # частина, якої немає в оригінальному ACMIL_MYMHA (там кожен bag
        # оброблюється повністю ізольовано, без контексту з інших bags). ---
        # context_mode='mean' -- старий грубий варіант (середнє фіч решти
        # bags, для абляції); 'attn' -- ContextAggregator (cross-attention
        # від самого bag'а до інших bags + їхня впевненість).
        if context_mode == 'attn':
            self.ctx_agg = ContextAggregator(D_context, d_ctx=d_ctx, n_heads=n_heads,
                                             dropout=dropout)
            film_in = d_ctx
        elif context_mode == 'mean':
            self.ctx_agg = None
            film_in = D_context
        else:
            raise ValueError(f"context_mode має бути 'attn' або 'mean', отримано {context_mode!r}")
        self.context_to_film = nn.Sequential(
            nn.Linear(film_in, D_inner_exp * 2), nn.GELU(),
        )

        # --- n_token ОКРЕМИХ branch-attention модулів, кожен зі своїм
        # єдиним query-вектором -- як `self.sub_attention` (ModuleList) в
        # ACMIL_MYMHA, а не один спільний attention з n_token queries. ---
        self.branch_query = nn.Parameter(torch.zeros(1, n_token, D_inner_exp))
        nn.init.normal_(self.branch_query, std=1e-6)
        self.branch_attn = nn.ModuleList([
            nn.MultiheadAttention(D_inner_exp, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_token)
        ])
        self.branch_norm = nn.LayerNorm(D_inner_exp)

        # --- По-branch класифікатор (як self.classifier ModuleList) ---
        self.branch_classifier = nn.Linear(D_inner_exp, n_class)

        # --- MoE-гейт над branches -- ТОЙ САМИЙ механізм, що pool_mode='moe'
        # в ACMIL_MYMHA: gate = softmax(Linear(branch_mean)), bag_pred =
        # (gate * sub_preds).sum(). НЕ просте середнє. ---
        self.gate = nn.Linear(D_inner_exp, n_token)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)

        # --- Проекція в розмір, який очікує вже натренований Tier2 ---
        self.out_proj = nn.Linear(D_inner_exp, D_inner_cheap)
        # Нуль-ініціалізація: на старті out_proj видає РІВНО НУЛЬ, тому
        # feats_out (нижче, з residual) == cheap_feats_this_bag -- Tier1.5
        # стартує як "тотожність" і вчиться лише ДОДАВАТИ корисну поправку
        # поверх уже пораховних cheap-фіч, а не вигадувати весь вектор
        # наосліп зі свіжоініціалізованої підмережі. Див. ФІКС нижче.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, exp_tile_feats, others_feats, others_conf, cheap_feats_this_bag):
        """
        exp_tile_feats:       [1, n_i, D_feat_exp]
        others_feats:         [1, K, D_context] -- cheap-фічі ІНШИХ bags слайду
        others_conf:          [1, K] -- їхня впевненість, 1 - entropy/log(n_class)
        cheap_feats_this_bag: [1, n_token*D_inner_cheap] -- ВЖЕ пораховані
            cheap Tier1-фічі САМЕ ЦЬОГО bag'а (той самий feats_j, який би
            пішов у Tier2, якби bag НЕ був ескальований). Tier1.5 віддає
            residual-корекцію поверх нього (ФІКС 2026-09-29, див. коментар
            нижче у forward()), а не замінює вектор повністю.
        """
        B, n_i, _ = exp_tile_feats.shape
        assert B == 1, "Tier1_5Module очікує один bag за раз (B=1); батчуй зовні, якщо треба."

        x = self.tile_proj(exp_tile_feats)  # [1, n_i, D_inner_exp]

        if self.context_mode == 'attn':
            context_vec = self.ctx_agg(cheap_feats_this_bag, others_feats, others_conf)  # [1, d_ctx]
        else:
            context_vec = others_feats.mean(dim=1)  # [1, D_context]

        # FiLM: film_params -> (scale, shift), обидва [1, D_inner_exp]
        film_params = self.context_to_film(context_vec)  # [1, D_inner_exp*2]
        scale, shift = film_params.chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # [1, n_i, D_inner_exp]

        # n_token НЕЗАЛЕЖНИХ branch-attentions, кожен зі своїм query [1,1,D]
        branch_feats = []
        for i in range(self.n_token):
            q_i = self.branch_query[:, i:i + 1, :]  # [1, 1, D_inner_exp]
            feat_i, _ = self.branch_attn[i](q_i, x, x, need_weights=False)  # [1, 1, D_inner_exp]
            branch_feats.append(feat_i)
        feats = torch.cat(branch_feats, dim=1)  # [1, n_token, D_inner_exp]
        feats = self.branch_norm(feats)

        sub_pred_out = self.branch_classifier(feats).squeeze(0)  # [n_token, n_class]

        # MoE-гейт, branch_mean-стиль (як gate_input='branch_mean' за
        # замовчуванням в ACMIL_MYMHA)
        bag_summary = feats.mean(dim=1)  # [1, D_inner_exp]
        gate = torch.softmax(self.gate(bag_summary), dim=-1)  # [1, n_token]
        bag_pred_out = (gate.squeeze(0).unsqueeze(-1) * sub_pred_out).sum(dim=0, keepdim=True)  # [1, n_class]

        delta = self.out_proj(feats).reshape(1, -1)  # [1, n_token*D_inner_cheap]

        # ФІКС (2026-09-29): RESIDUAL, а не hard swap. Стара версія повертала
        # `feats_out = delta` напряму -- це означало, що заморожений Tier2
        # для flagged bags отримував вектор зі СВІЖОІНІЦІАЛІЗОВАНОЇ підмережі
        # (нормалізований через branch_norm=LayerNorm до mean~0/var~1), який
        # статистично НІЧОГО спільного не мав із feats_j, які реально
        # виробляє натренований cheap_model.tier1 -- Tier2 бачив вхід зі
        # "статистично чужого світу" з першого ж forward-проходу. Через
        # zeros_-ініціалізацію out_proj (вище) delta==0 на старті, тому
        # feats_out == cheap_feats_this_bag: Tier1.5 стартує як тотожність
        # (Tier2 на епосі 0 бачить РІВНО ТЕ САМЕ, що й без ескалації), і
        # вчиться лише ДОДАВАТИ корисний сигнал з дорогих UNI2h-фіч поверх
        # уже адекватного cheap-вектора, а не замінювати його наосліп.
        feats_out = cheap_feats_this_bag + delta

        return feats_out, sub_pred_out, bag_pred_out


class DTFD_ASMIL_v2_5(nn.Module):
    """
    Обгортка: заморожений натренований DTFD_ASMIL_v2 (cheap) + новий
    трен. Tier1_5Module для top-K найневпевненіших bags слайду.

    ПРИПУЩЕННЯ ПРО ВХІД (те саме, що вже використовують eval-скрипти):
    cheap і expensive фічі одного слайду вирівняні по тайлам (той самий
    порядок/кількість) -- forward() кидає AssertionError, якщо це не так;
    виклик (тренувальний цикл) має сам пропускати такі слайди, так само,
    як робить validate_bag_level_routing.py.

    forward(cheap_feats, expensive_feats, coords_cheap=None) очікує ПОВНІ
    фічі слайду для ОБОХ backbone (той самий .pt на диску, що вже
    використовується в eval-скриптах) -- НЕ підмножину по bags. Нарізка на
    M bags відбувається один раз всередині forward і застосовується
    однаково до cheap і expensive.
    """

    def __init__(self, cheap_model, D_feat_exp, D_inner_exp, n_token, D_inner_cheap,
                 n_class, top_k_bags=2, n_heads=4, dropout=0.1,
                 train_tier2=False, context_mode='attn', d_ctx=128):
        super().__init__()
        self.cheap_model = cheap_model
        # Tier1 ЗАВЖДИ заморожений (він лише постачає cheap-фічі/передбачення
        # і визначає, які bags ескалювати). Tier2 -- заморожений лише якщо
        # train_tier2=False; при train_tier2=True він донавчається разом із
        # Tier1.5 (2026-10-06), щоб навчитись читати вектори, які вже не
        # суто cheap-стилю (частина bags тепер "cheap + поправка з UNI2h").
        for p in self.cheap_model.parameters():
            p.requires_grad_(False)
        self.train_tier2 = train_tier2
        if train_tier2:
            for p in self.cheap_model.tier2.parameters():
                p.requires_grad_(True)
        self.cheap_model.eval()

        self.top_k_bags = top_k_bags
        self.n_token = n_token
        self.n_class = n_class

        # D_context: контекст = усереднений дистильований вектор ІНШИХ bags,
        # тобто розмірність n_token*D_inner_cheap (той самий формат, що й
        # feats_j від cheap Tier1).
        D_context = n_token * D_inner_cheap
        self.tier1_5 = Tier1_5Module(
            D_feat_exp=D_feat_exp, D_inner_exp=D_inner_exp, D_context=D_context,
            n_token=n_token, D_inner_cheap=D_inner_cheap, n_class=n_class,
            n_heads=n_heads, dropout=dropout, context_mode=context_mode, d_ctx=d_ctx,
        )

    @staticmethod
    def _entropy(probs, eps=1e-12):
        probs = probs.clamp_min(eps)
        return -(probs * probs.log()).sum(dim=-1)

    @staticmethod
    def _split_pseudo_bags_seeded(n_tiles, M, seed, device):
        """Детермінований аналог `cheap_model._split_pseudo_bags`, той самий
        принцип, що `split_bags_externally` у validate_bag_level_routing.py.
        Потрібен для ВІДТВОРЮВАНОЇ валідації/тесту: якщо не зафіксувати сід,
        кожен виклик forward() ділить тайли слайду на bags ПО-РІЗНОМУ (бо
        `_split_pseudo_bags` у базовій моделі викликає torch.randperm без
        сіда), і тоді порівняння "модель до/після епохи" порівнює не тільки
        різні ваги, а й РІЗНЕ bag-розбиття того самого слайду -- шум від
        цього маскує реальний прогрес навчання."""
        g = torch.Generator(device='cpu').manual_seed(seed)
        perm = torch.randperm(n_tiles, generator=g).to(device)
        m_eff = min(M, n_tiles)
        return list(torch.chunk(perm, m_eff))

    def forward(self, cheap_feats, expensive_feats, coords_cheap=None, bag_seed=None):
        """
        cheap_feats:      [1, N, D_feat_cheap] -- повний слайд, як у DTFD_ASMIL_v2.forward
        expensive_feats:  [1, N, D_feat_exp] -- UNI2h-фічі ЦІЛОГО ТОГО Ж слайду,
            ТІ САМІ N тайлів у ТОМУ САМОМУ порядку, що й cheap_feats (перевір
            це на своїх даних -- я додаю runtime assert нижче). Це "офлайн"-
            режим: ми вже маємо фічі всього слайду на диску (як і в
            validate_bag_level_routing.py), тому нарізка на bags відбувається
            ОДИН РАЗ всередині forward і застосовується однаково до cheap і
            expensive -- жодної окремої "лінивої" UNI2h-екстракції по
            flagged bags тут не потрібно.

            ІНШИЙ (майбутній, ще не реалізований) сценарій -- lazy inference,
            де UNI2h рахується лише для top-K bags "на льоту" після виявлення
            їх ентропії -- зажадає двопрохідної схеми і власного forward();
            цей forward() написаний для тренування, де фічі вже все одно
            лежать на диску для обох backbone.

        Повертає: final_slide_pred, flagged_bag_indices (діагностика),
                  tier1_5_sub_preds (для допоміжного лоссу, лише для
                  ескальованих bags).
        """
        device = cheap_feats.device
        n_tiles = cheap_feats.shape[1]
        assert expensive_feats.shape[1] == n_tiles, (
            f"cheap і expensive фічі мають різну к-сть тайлів "
            f"({n_tiles} vs {expensive_feats.shape[1]}) -- партиції на bags "
            f"будуть неспівставні. Пропусти цей слайд на рівні виклику "
            f"(так само, як робить validate_bag_level_routing.py).")
        if bag_seed is not None:
            # Детерміноване розбиття -- використовуй для валідації/тесту,
            # щоб порівнювати епохи на ОДНАКОВОМУ bag-розбитті кожного слайду.
            chunks = self._split_pseudo_bags_seeded(n_tiles, self.n_token, bag_seed, device)
        else:
            # Випадкове (як у базовій моделі) -- ОК для тренування, це навіть
            # корисна аугментація (той самий слайд бачить різні bag-розбиття
            # в різних епохах).
            chunks = self.cheap_model._split_pseudo_bags(n_tiles, device)

        with torch.no_grad():
            cheap_bag_feats_list = []
            cheap_bag_preds = []
            own_uncertainty = []
            for idx in chunks:
                pseudo_bag = cheap_feats[:, idx, :]
                bag_coords = coords_cheap[:, idx, :] if coords_cheap is not None else None
                _, slide_pred_j, _, feats_j = self.cheap_model.tier1(
                    pseudo_bag, coords=bag_coords, return_feats=True)
                cheap_bag_feats_list.append(feats_j.reshape(1, -1))  # [1, n_token*D_inner_cheap]
                cheap_bag_preds.append(slide_pred_j)
                probs = torch.softmax(slide_pred_j, dim=-1)
                own_uncertainty.append(self._entropy(probs).item())

        M = len(chunks)
        own_uncertainty = torch.tensor(own_uncertainty)
        k = min(self.top_k_bags, M)
        flagged = torch.topk(own_uncertainty, k=k).indices.tolist()

        # Контекст для кожного flagged bag = дистильовані фічі ІНШИХ bags
        # (КРІМ самого flagged, щоб не підглядав власну ще не оновлену
        # версію) + впевненість кожного з них. Як саме це агрегується --
        # вирішує Tier1_5Module (context_mode='attn': cross-attention від
        # самого bag'а; 'mean': старе грубе середнє для абляції).
        final_feats = list(cheap_bag_feats_list)  # копія, замінюватимемо flagged слоти
        tier1_5_sub_preds = {}
        conf_all = 1.0 - own_uncertainty / math.log(max(self.n_class, 2))  # [M], ~[0,1]

        for bag_idx in flagged:
            others_idx = [j for j in range(M) if j != bag_idx]
            if others_idx:
                others_feats = torch.stack(
                    [cheap_bag_feats_list[j] for j in others_idx], dim=1)  # [1, K, n_token*D_inner_cheap]
                others_conf = conf_all[others_idx].to(device).unsqueeze(0)  # [1, K]
            else:
                others_feats = torch.zeros_like(cheap_bag_feats_list[0]).unsqueeze(1)  # [1, 1, D]
                others_conf = torch.zeros(1, 1, device=device)

            idx = chunks[bag_idx]
            exp_bag_feats = expensive_feats[:, idx, :]  # [1, n_i, D_feat_exp] -- ТІ САМІ тайли, що cheap-bag bag_idx
            cheap_feats_this_bag = cheap_bag_feats_list[bag_idx]  # [1, n_token*D_inner_cheap], для residual (ФІКС)
            feats_out, sub_pred_out, bag_pred_out = self.tier1_5(
                exp_bag_feats, others_feats, others_conf, cheap_feats_this_bag)
            final_feats[bag_idx] = feats_out
            tier1_5_sub_preds[bag_idx] = (sub_pred_out, bag_pred_out)

        tier2_input = torch.cat(final_feats, dim=0).unsqueeze(0)  # [1, M, n_token*D_inner_cheap]
        # НЕ огортаємо цей виклик у torch.no_grad()! Якщо Tier2 заморожений
        # (train_tier2=False), requires_grad_(False) вже гарантує, що його
        # ваги не оновлюються; якщо розморожений -- він вчиться разом із
        # Tier1.5. В обох випадках градієнт головного (CE на
        # final_slide_pred) лоссу МАЄ проходити НАСКРІЗЬ через Tier2 назад до
        # Tier1_5Module, інакше Tier1.5 вчиться лише з допоміжного (aux)
        # лоссу, а не з головної цілі -- це саме та помилка, яку знайшли на
        # тренувальному прогоні 2026-09-28 (main loss не рухався, RuntimeError
        # "does not require grad" при aux_weight=0).
        final_slide_pred, _ = self.cheap_model.tier2(tier2_input)

        return final_slide_pred, flagged, tier1_5_sub_preds


if __name__ == "__main__":
    # Smoke-test на випадкових тензорах -- перевіряє лише узгодженість
    # РОЗМІРНОСТЕЙ Tier1_5Module (без cheap_model/checkpoints/GPU). Запусти
    # це ПЕРШИМ на своїй машині (де є torch), перш ніж інтегрувати в
    # реальний пайплайн -- дешевий спосіб зловити помилки у формах.
    torch.manual_seed(0)
    n_token, D_inner_cheap, D_feat_exp, D_inner_exp, n_class = 8, 128, 1536, 256, 4
    D_context = n_token * D_inner_cheap

    n_i = 37  # довільна к-сть тайлів у "проблемному" bag'у
    K = n_token - 1  # "інші" bags (M=8 -> 7)
    exp_tile_feats = torch.randn(1, n_i, D_feat_exp)
    others_feats = torch.randn(1, K, D_context)
    others_conf = torch.rand(1, K)
    cheap_feats_this_bag = torch.randn(1, n_token * D_inner_cheap)

    for mode in ('attn', 'mean'):
        mod = Tier1_5Module(D_feat_exp=D_feat_exp, D_inner_exp=D_inner_exp,
                             D_context=D_context, n_token=n_token,
                             D_inner_cheap=D_inner_cheap, n_class=n_class,
                             context_mode=mode)
        feats_out, sub_pred_out, bag_pred_out = mod(
            exp_tile_feats, others_feats, others_conf, cheap_feats_this_bag)
        assert feats_out.shape == (1, n_token * D_inner_cheap), feats_out.shape
        assert sub_pred_out.shape == (n_token, n_class), sub_pred_out.shape
        assert bag_pred_out.shape == (1, n_class), bag_pred_out.shape
        # Перевірка residual-фіксу: на старті (out_proj нуль-ініціалізований)
        # feats_out МАЄ бути рівно cheap_feats_this_bag (delta==0).
        assert torch.allclose(feats_out, cheap_feats_this_bag), \
            "residual-фікс зламано: feats_out != cheap_feats_this_bag при нульовій delta"
        n_par = sum(p.numel() for p in mod.parameters())
        print(f"Tier1_5Module smoke-test OK [context_mode={mode}]: "
              f"feats_out={tuple(feats_out.shape)}, params={n_par:,}")