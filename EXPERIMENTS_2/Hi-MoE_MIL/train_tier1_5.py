#!/usr/bin/env python
"""
Тренування Tier1.5: bag-рівневий каскад з контекстом (DTFD_ASMIL_v2_5).

Заморожує вже натреновану cheap DTFD_ASMIL_v2 (tier1 + tier2 -- ОБИДВА,
дивись forward() у dtfd_asmil_tier1_5.py, tier2 явно під torch.no_grad()),
тренує ЛИШЕ новий Tier1_5Module на:
  (а) головному slide-рівневому CrossEntropy лоссі (той самий таргет, що й
      база модель -- мітка слайду), і
  (б) необов'язковому допоміжному лоссі на власні sub_pred/bag_pred
      ЕСКАЛЬОВАНИХ bags (той самий принцип, що tier1_weight-дистиляція в
      базовій моделі: proxy ground truth bag'а = мітка слайду).

Дані: той самий формат, що й у validate_bag_level_routing.py --
PairedFeatureDataset (csv зі slide_id,label + окремі .pt файли з cheap і
expensive фічами ЦІЛОГО слайду). Ми вже маємо фічі всього слайду для обох
backbone на диску, тому НІЯКОЇ додаткової UNI2h-екстракції тут не треба --
DTFD_ASMIL_v2_5.forward() сам нарізає слайд на bags і бере відповідну
підмножину вже завантажених expensive-фіч для flagged bags ("офлайн"-режим,
на відміну від майбутнього lazy-inference режиму, де UNI2h рахувався б лише
для top-K bags на льоту -- це окрема, ще не реалізована оптимізація).

Запуск (з кореня Hi-MoE_MIL, той самий venv, що й для validate-скрипта):
    python train_tier1_5.py \
        --config_cheap config/bracs_vits_dino_config_100ep.yml \
        --config_expensive config/bracs_uni2h_config.yml \
        --cheap_ckpt ckpt/bracs_dtfd_v2_seed1/checkpoint-best.pth \
        --n_class 3 \
        --top_k_bags 2 \
        --epochs 30 \
        --lr 1e-4 \
        --aux_weight 0.1 \
        --out_dir ckpt/bracs_tier1_5_seed1

Для C17 (2 класи):
    python train_tier1_5.py \
        --config_cheap config/camelyon17_vitsdino_config_20ep.yml \
        --config_expensive config/camelyon17_uni2h_config.yml \
        --cheap_ckpt ckpt/c17_dtfd_v2_20ep_seed1/checkpoint-best.pth \
        --n_class 2 \
        --top_k_bags 2 \
        --epochs 30 \
        --lr 1e-4 \
        --aux_weight 0.1 \
        --out_dir ckpt/c17_tier1_5_seed1

--n_class постав правильно під датасет (BRACS: 3 класи -- benign/atypical/
malignant; C17: 2 класи). --top_k_bags -- скільки з M=8 bags ескалювати
(2 -- це 25%, орієнтуйся на найкращі % з validate_bag_level_routing.py:
BRACS ~20-30%, C17 навіть 10% давало дуже сильний ефект).

ВАЖЛИВО: цей скрипт тренує ОДИН seed (--cheap_ckpt на один конкретний
seed-чекпоінт cheap-моделі) -- для порівняння з таблицею (де ми маємо 10
сідів на cheap/expensive-only) треба буде прогнати це для кожного спільного
сіда окремо (простий bash-цикл з підстановкою seedN у --cheap_ckpt/--out_dir),
так само, як для validate-скрипта.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score

from utils.utils import Struct
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2
from dtfd_asmil_tier1_5 import DTFD_ASMIL_v2_5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PairedFeatureDataset(torch.utils.data.Dataset):
    """Той самий формат, що й у validate_bag_level_routing.py -- csv зі
    slide_id,label колонками + окремі .pt файли (dict з 'features'/'coords')
    для cheap і expensive фіч ЦІЛОГО слайду."""

    def __init__(self, split_csv, feature_dir_cheap, feature_dir_expensive):
        import pandas as pd
        self.df = pd.read_csv(split_csv)
        self.feature_dir_cheap = feature_dir_cheap
        self.feature_dir_expensive = feature_dir_expensive

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = int(row["label"])
        data_c = torch.load(os.path.join(self.feature_dir_cheap, f"{slide_id}.pt"),
                             map_location="cpu", weights_only=False)
        data_e = torch.load(os.path.join(self.feature_dir_expensive, f"{slide_id}.pt"),
                             map_location="cpu", weights_only=False)
        return {'feats_cheap': data_c['features'], 'coords_cheap': data_c['coords'].float(),
                'feats_expensive': data_e['features'], 'label': label, 'slide_id': slide_id}


def load_conf(config_path, pretrain, n_token=8):
    with open(config_path, "r") as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    c['pretrain'] = pretrain
    c['n_token'] = n_token
    conf = Struct(**c)
    if pretrain == 'medical_ssl':
        conf.D_feat, conf.D_inner = 384, 128
    elif pretrain == 'UNI2h':
        conf.D_feat, conf.D_inner = 1536, 768
    return conf


def load_cheap_model(conf, ckpt_path, M):
    model = DTFD_ASMIL_v2(conf, n_token=conf.n_token, M=M).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state['model'] if 'model' in state else state
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate(model, dataset):
    """ВАЖЛИВО: bag_seed=slide_i фіксує розбиття тайлів на bags ОДНАКОВИМ
    для того самого слайду на КОЖНОМУ виклику evaluate() (кожної епохи) --
    інакше `_split_pseudo_bags` (без сіда) ділить тайли по-різному щоразу,
    і зміни val_acc/val_f1 між епохами відображають більше шум від
    випадкового bag-розбиття, ніж реальний прогрес навчання Tier1.5."""
    model.eval()
    preds, labels = [], []
    n_skipped = 0
    for slide_i, item in enumerate(dataset):
        feats_c = item['feats_cheap'].unsqueeze(0).to(device)
        feats_e = item['feats_expensive'].unsqueeze(0).to(device)
        coords_c = item['coords_cheap'].unsqueeze(0).to(device)
        if feats_c.shape[1] != feats_e.shape[1]:
            n_skipped += 1
            continue
        slide_pred, flagged, _ = model(feats_c, feats_e, coords_cheap=coords_c,
                                        bag_seed=1000 + slide_i)
        preds.append(slide_pred.argmax(dim=-1).item())
        labels.append(item['label'])
    acc = accuracy_score(labels, preds) if preds else float('nan')
    f1 = f1_score(labels, preds, average='macro') if preds else float('nan')
    return acc, f1, len(preds), n_skipped


def train_one_epoch(model, dataset, optimizer, aux_weight, ce_loss):
    model.tier1_5.train()
    model.cheap_model.eval()  # cheap Tier1 завжди в eval (заморожений)
    if model.train_tier2:
        model.cheap_model.tier2.train()  # розморожений Tier2 вчиться -> dropout увімкнено

    order = np.random.permutation(len(dataset))
    total_loss, total_main_loss, total_aux_loss = 0.0, 0.0, 0.0
    n_seen, n_skipped, n_correct = 0, 0, 0
    grad_norm_sum, n_grad = 0.0, 0
    for i in order:
        item = dataset[int(i)]
        feats_c = item['feats_cheap'].unsqueeze(0).to(device)
        feats_e = item['feats_expensive'].unsqueeze(0).to(device)
        coords_c = item['coords_cheap'].unsqueeze(0).to(device)
        label = torch.tensor([item['label']], device=device)

        if feats_c.shape[1] != feats_e.shape[1]:
            n_skipped += 1
            continue

        slide_pred, flagged, tier1_5_sub_preds = model(feats_c, feats_e, coords_cheap=coords_c)
        main_loss = ce_loss(slide_pred, label)
        loss = main_loss
        aux_loss_val = 0.0

        if aux_weight > 0 and len(tier1_5_sub_preds) > 0:
            aux_loss = 0.0
            for bag_idx, (sub_pred_out, bag_pred_out) in tier1_5_sub_preds.items():
                # proxy ground truth = мітка слайду, той самий принцип, що
                # tier1_weight-дистиляція в базовій моделі.
                target = label.expand(sub_pred_out.shape[0])
                aux_loss = aux_loss + ce_loss(sub_pred_out, target)
                aux_loss = aux_loss + ce_loss(bag_pred_out, label)
            aux_loss = aux_loss / (2 * len(tier1_5_sub_preds))
            loss = loss + aux_weight * aux_loss
            aux_loss_val = aux_loss.item()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.tier1_5.parameters(), max_norm=1e9)
        grad_norm_sum += float(grad_norm)
        n_grad += 1
        optimizer.step()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss_val
        n_correct += int(slide_pred.argmax(dim=-1).item() == item['label'])
        n_seen += 1

    return {
        "loss": total_loss / max(n_seen, 1),
        "main_loss": total_main_loss / max(n_seen, 1),
        "aux_loss": total_aux_loss / max(n_seen, 1),
        "train_acc": n_correct / max(n_seen, 1),
        "grad_norm": grad_norm_sum / max(n_grad, 1),
        "n_skipped": n_skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--cheap_ckpt", required=True,
                     help="шлях до checkpoint-best.pth ВЖЕ натренованої cheap DTFD_ASMIL_v2 (один конкретний seed)")
    ap.add_argument("--n_class", type=int, required=True)
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--top_k_bags", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--d_inner_tier1_5", type=int, default=128,
                     help="Внутрішня робоча розмірність Tier1_5Module (НЕ те саме, "
                          "що conf_exp.D_inner=768 для UNI2h!). При 768 модуль має "
                          "n_token=8 ОКРЕМИХ nn.MultiheadAttention по ~2.4M параметрів "
                          "кожен (~19M сумарно) -- катастрофічно багато відносно "
                          "~390 тренувальних слайдів BRACS, overfit вже на epoch 1-2. "
                          "128 (розмір D_inner_cheap, як у реальному ACMIL) зменшує "
                          "це на порядок.")
    ap.add_argument("--context_mode", choices=["attn", "mean"], default="attn",
                     help="Як Tier1.5 агрегує контекст ІНШИХ bags: 'attn' -- cross-attention "
                          "від самого bag'а з урахуванням впевненості сусідів (нове, "
                          "2026-10-06); 'mean' -- старе грубе середнє (для абляції).")
    ap.add_argument("--d_ctx", type=int, default=128,
                     help="Розмірність виходу ContextAggregator (лише для --context_mode attn).")
    ap.add_argument("--train_tier2", action="store_true",
                     help="Розморозити Tier2 і донавчати його разом із Tier1.5. Tier1 лишається "
                          "замороженим. БЕЗ цього прапора Tier2 заморожений (як раніше).")
    ap.add_argument("--lr_tier2", type=float, default=1e-5,
                     help="lr для Tier2 (лише з --train_tier2). Менший за --lr, бо Tier2 уже "
                          "натренований -- це fine-tuning, а не навчання з нуля.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--aux_weight", type=float, default=0.05)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--patience", type=int, default=8,
                     help="Рання зупинка: скільки епох поспіль без покращення val_f1 "
                          "терпіти, перш ніж зупинити тренування достроково. На пілотних "
                          "прогонах (2026-09-28) пік val_f1 стабільно був на epoch 1-2, "
                          "далі лише перенавчання -- немає сенсу палити час на всі --epochs.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    conf_cheap = load_conf(args.config_cheap, "medical_ssl", args.n_token)
    conf_exp = load_conf(args.config_expensive, "UNI2h", args.n_token)
    M = args.n_token

    train_csv = os.path.join(conf_cheap.splits_dir, 'train.csv')
    val_csv = os.path.join(conf_cheap.splits_dir, 'val.csv')
    test_csv = os.path.join(conf_cheap.splits_dir, 'test.csv')

    # train_dir/test_dir -- як у validate_bag_level_routing.py: у цій
    # кодобазі і train, і val (і навіть частина test) фактично лежать у
    # тій самій "_train"-директорії (спліт визначається CSV, не назвою
    # папки) -- перевірено емпірично на сесії 2026-09-28. Якщо на твоєму
    # сервері val/test насправді в іншій директорії, підправ тут.
    train_ds = PairedFeatureDataset(train_csv, conf_cheap.train_dir, conf_exp.train_dir)
    val_ds = PairedFeatureDataset(val_csv, conf_cheap.train_dir, conf_exp.train_dir)
    test_ds = PairedFeatureDataset(test_csv, conf_cheap.test_dir, conf_exp.test_dir)

    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    cheap_model = load_cheap_model(conf_cheap, args.cheap_ckpt, M)

    model = DTFD_ASMIL_v2_5(
        cheap_model=cheap_model,
        D_feat_exp=conf_exp.D_feat,
        D_inner_exp=args.d_inner_tier1_5,  # НЕ conf_exp.D_inner=768 -- дивись help --d_inner_tier1_5
        n_token=conf_cheap.n_token,
        D_inner_cheap=conf_cheap.D_inner,
        n_class=args.n_class,
        top_k_bags=args.top_k_bags,
        n_heads=args.n_heads,
        dropout=args.dropout,
        train_tier2=args.train_tier2,
        context_mode=args.context_mode,
        d_ctx=args.d_ctx,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.tier1_5.parameters() if p.requires_grad)
    print(f"Tier1_5Module: {n_trainable:,} тренованих параметрів "
          f"(context_mode={args.context_mode}); Tier1 заморожений")
    param_groups = [{"params": list(model.tier1_5.parameters()), "lr": args.lr}]
    if args.train_tier2:
        tier2_params = [p for p in model.cheap_model.tier2.parameters() if p.requires_grad]
        n_t2 = sum(p.numel() for p in tier2_params)
        print(f"Tier2 РОЗМОРОЖЕНО: {n_t2:,} тренованих параметрів, lr={args.lr_tier2}")
        param_groups.append({"params": tier2_params, "lr": args.lr_tier2})
    else:
        print("Tier2 заморожений")

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(args.out_dir, 'checkpoint-best.pth')
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_ds, optimizer, args.aux_weight, ce_loss)
        val_acc, val_f1, val_n, val_skipped = evaluate(model, val_ds)
        print(f"epoch {epoch:3d}  loss={stats['loss']:.4f} "
              f"(main={stats['main_loss']:.4f} aux={stats['aux_loss']:.4f})  "
              f"train_acc={stats['train_acc']:.4f}  grad_norm={stats['grad_norm']:.4g}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  (n={val_n}, skip={val_skipped})")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            ckpt = {'tier1_5_state_dict': model.tier1_5.state_dict(),
                    'epoch': epoch, 'val_f1': val_f1, 'val_acc': val_acc,
                    'args': vars(args)}
            if args.train_tier2:
                ckpt['tier2_state_dict'] = model.cheap_model.tier2.state_dict()
            torch.save(ckpt, best_ckpt_path)
            print(f"  -> новий найкращий val_f1={val_f1:.4f}, збережено у {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  -> рання зупинка: {args.patience} епох без покращення val_f1")
                break

    # Фінальна оцінка на test з найкращим (за val_f1) чекпоінтом Tier1.5
    best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.tier1_5.load_state_dict(best['tier1_5_state_dict'])
    if args.train_tier2:
        model.cheap_model.tier2.load_state_dict(best['tier2_state_dict'])
    test_acc, test_f1, test_n, test_skipped = evaluate(model, test_ds)
    print(f"\n=== ФІНАЛЬНИЙ TEST (найкращий за val_f1={best['val_f1']:.4f}, epoch={best['epoch']}) ===")
    print(f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  (n={test_n}, skip={test_skipped})")
    print("\nПорівняй ці test_acc/test_f1 з рядками cheap-only / expensive-only "
          "у таблиці документа проєкту -- це і є число для рядка "
          "'наш DTFD-ASMIL v2.5 (Tier1.5)'.")


if __name__ == "__main__":
    main()