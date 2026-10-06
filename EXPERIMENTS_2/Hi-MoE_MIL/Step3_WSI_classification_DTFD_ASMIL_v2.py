#!/usr/bin/env python
"""
Step3_WSI_classification_DTFD_ASMIL_v2.py

DTFD-ASMIL v2: Tier1 (random split, ACMIL_MYMHA+MoE, concat distillation,
без змін від v1) + новий Tier2 (LSATransformerTier2 -- справжній 2-шаровий
Transformer з Q/K/V+FFN, і Location-Sensitive Attention між шарами).

--tier1_weight за замовчуванням 0.1 (не 0, не 1.0) -- компроміс, знайдений
експериментально: досить, щоб Tier1 не був повністю "поламаним" самостійним
класифікатором, недостатньо, щоб tier1_loss домінував і псував якість
дистильованих ознак для Tier 2.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed
from pt_dataset import PTClassificationDataset
from architecture.dtfd_asmil_v2 import DTFD_ASMIL_v2

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
import numpy as np
import wandb

DATASET_CLASS_NAMES = {
    'camelyon17': ['negative', 'itc', 'micro', 'macro'],
    'camelyon16': ['normal', 'tumor'],
    'bracs': ['benign', 'atypical', 'malignant'],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_arguments():
    parser = argparse.ArgumentParser('DTFD-ASMIL v2 training')
    parser.add_argument('--config', dest='config', default='config/camelyon17_vitsdino_config_30ep.yml')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'])
    parser.add_argument("--n_token", type=int, default=8, help="branches у Tier 1 (=M, кількість псевдо-bags)")
    parser.add_argument("--M", type=int, default=None, help="кількість псевдо-bags; за замовчуванням = n_token")
    parser.add_argument("--tier2_D_inner", type=int, default=None,
                        help="внутрішня розмірність Tier 2; за замовчуванням = conf.D_inner")
    parser.add_argument("--tier2_n_heads", type=int, default=4, help="кількість голів attention у Tier 2")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument('--pretrain', default='medical_ssl',
                        choices=['natural_supervised', 'medical_ssl', 'plip', 'UNI', 'GigaPath', 'UNI2h'])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--tier1_weight", type=float, default=0.1,
                        help="вага tier1_loss відносно tier2_loss (рекомендовано 0.05-0.1, не 0 і не 1.0)")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)
        set_seed(args.seed)

    if conf.pretrain == 'medical_ssl':
        conf.D_feat, conf.D_inner = 384, 128
    elif conf.pretrain == 'UNI2h':
        conf.D_feat, conf.D_inner = 1536, 768
    elif conf.pretrain == 'UNI':
        conf.D_feat, conf.D_inner = 1024, 512
    elif conf.pretrain == 'GigaPath':
        conf.D_feat, conf.D_inner = 1536, 768

    M = args.M if args.M is not None else args.n_token

    wandb.init(
        project="wsi_classification",
        config={'dataset': conf.dataset, 'pretrain': conf.pretrain, 'n_token': conf.n_token,
                'M': M, 'seed': conf.seed, 'method': 'DTFD_ASMIL_v2', 'tier1_weight': args.tier1_weight},
        mode=args.wandb_mode
    )
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(wandb.run.dir, 'saved_models')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Used config (DTFD-ASMIL v2):")
    pprint(vars(conf))
    print(f"M (псевдо-bags) = {M}, tier1_weight = {args.tier1_weight}")

    CLASS_NAMES = DATASET_CLASS_NAMES.get(conf.dataset, [f'class{i}' for i in range(conf.n_class)])

    train_data = PTClassificationDataset(
        split_csv=os.path.join(conf.splits_dir, 'train.csv'), feature_dir=conf.train_dir)
    val_data = PTClassificationDataset(
        split_csv=os.path.join(conf.splits_dir, 'val.csv'), feature_dir=conf.train_dir)
    test_data = PTClassificationDataset(
        split_csv=os.path.join(conf.splits_dir, 'test.csv'), feature_dir=conf.test_dir)

    class_counts = train_data.df['label'].value_counts().sort_index()
    counts_tensor = torch.tensor(
        [class_counts.get(c, 0) for c in range(conf.n_class)], dtype=torch.float32)
    class_weights = (counts_tensor.sum() / (conf.n_class * counts_tensor.clamp(min=1))).to(device)
    print("Class weights:", class_weights.tolist())

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    model = DTFD_ASMIL_v2(
        conf, n_token=args.n_token, M=M,
        tier2_n_heads=args.tier2_n_heads, tier2_D_inner=args.tier2_D_inner,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=conf.wd)

    best_state = {'epoch': -1, 'val_f1': 0, 'val_auc': 0}
    for epoch in range(conf.train_epoch):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, conf, args.tier1_weight)

        val_auc, val_acc, val_f1, val_loss, val_f1_pc, val_cm, val_t1_f1, val_t1_acc = evaluate(
            model, eval_criterion, val_loader, device, conf, 'Val', CLASS_NAMES)
        test_auc, test_acc, test_f1, test_loss, test_f1_pc, test_cm, test_t1_f1, test_t1_acc = evaluate(
            model, eval_criterion, test_loader, device, conf, 'Test', CLASS_NAMES)

        if args.wandb_mode != 'disabled':
            wandb.log({'perf/val_f1': val_f1, 'perf/val_auc': val_auc,
                      'perf/test_f1': test_f1, 'perf/test_auc': test_auc,
                      'diag/test_tier1_f1': test_t1_f1, 'diag/test_tier1_acc': test_t1_acc})

        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state.update(epoch=epoch, val_f1=val_f1, val_auc=val_auc,
                             test_f1=test_f1, test_auc=test_auc,
                             test_f1_pc=test_f1_pc, test_cm=test_cm)
            save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))
        print()

    print("=" * 60)
    print(f"FINAL EPOCH ({conf.train_epoch - 1}) RESULTS (DTFD-ASMIL v2):")
    print(f"  test_auc={test_auc:.4f} test_acc={test_acc:.2f} test_f1(macro)={test_f1:.4f}")
    print(f"  [DIAG] Tier1 сам по собі: acc={test_t1_acc:.3f} f1_macro={test_t1_f1:.3f}")
    print(f"  test per-class F1:", dict(zip(CLASS_NAMES[:conf.n_class], np.round(test_f1_pc, 3))))
    print("  test confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c:>10s}" for c in CLASS_NAMES[:conf.n_class]))
    for i, row in enumerate(test_cm):
        print(f"{CLASS_NAMES[i]:>8s} " + "  ".join(f"{v:>10d}" for v in row))
    np.save(os.path.join(ckpt_dir, 'test_confusion_matrix_final.npy'), test_cm)
    print("=" * 60)
    print("Best epoch:", {k: v for k, v in best_state.items() if k not in ('test_f1_pc', 'test_cm')})

    wandb.finish()


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, conf, tier1_weight):
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data_it, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
        image_patches = data[0].to(device, dtype=torch.float32)
        labels = data[1].to(device)
        coords = data[2].to(device, dtype=torch.float32)

        adjust_learning_rate(optimizer, epoch + data_it / len(data_loader), conf)

        final_slide_pred, tier1_slide_preds, _ = model(image_patches, coords=coords)

        tier2_loss = criterion(final_slide_pred, labels)

        labels_repeated = labels.repeat(tier1_slide_preds.shape[0])
        tier1_loss = criterion(tier1_slide_preds, labels_repeated)

        loss = tier2_loss + tier1_weight * tier1_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(tier1_loss=tier1_loss.item())
        metric_logger.update(tier2_loss=tier2_loss.item())

        if conf.wandb_mode != 'disabled':
            wandb.log({'tier1_loss': tier1_loss.item(), 'tier2_loss': tier2_loss.item()})


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf, header, class_names):
    model.eval()
    y_pred, y_true = [], []
    tier1_pred_all, tier1_true_all = [], []
    metric_logger = MetricLogger(delimiter=" ")
    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data[0].to(device, dtype=torch.float32)
        labels = data[1].to(device)
        coords = data[2].to(device, dtype=torch.float32)

        final_slide_pred, tier1_slide_preds, _ = model(image_patches, coords=coords)
        loss = criterion(final_slide_pred, labels)
        pred = torch.softmax(final_slide_pred, dim=-1)
        acc1 = accuracy(pred, labels, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

        y_pred.append(pred)
        y_true.append(labels)

        tier1_pred_softmax = torch.softmax(tier1_slide_preds, dim=-1)
        tier1_pred_all.append(tier1_pred_softmax)
        tier1_true_all.append(labels.repeat(tier1_slide_preds.shape[0]))

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    auroc = torchmetrics.AUROC(num_classes=conf.n_class, task='multiclass').to(device)(y_pred, y_true).item()
    f1_macro = torchmetrics.F1Score(num_classes=conf.n_class, task='multiclass', average='macro').to(device)(y_pred, y_true).item()
    f1_pc = torchmetrics.F1Score(num_classes=conf.n_class, task='multiclass', average=None).to(device)(y_pred, y_true).cpu().numpy()
    cm = torchmetrics.ConfusionMatrix(num_classes=conf.n_class, task='multiclass').to(device)(y_pred, y_true).cpu().numpy()

    tier1_pred_all = torch.cat(tier1_pred_all, dim=0)
    tier1_true_all = torch.cat(tier1_true_all, dim=0)
    tier1_f1_macro = torchmetrics.F1Score(num_classes=conf.n_class, task='multiclass', average='macro').to(device)(tier1_pred_all, tier1_true_all).item()
    tier1_acc1 = accuracy(tier1_pred_all, tier1_true_all, topk=(1,))[0].item()

    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f} '
          f'auroc {auroc:.3f} f1_score {f1_macro:.3f}')
    print(f'  per-class F1 ({header}):', dict(zip(class_names[:conf.n_class], np.round(f1_pc, 3))))
    print(f'  [DIAG] Tier1 сам по собі: acc={tier1_acc1:.3f} f1_macro={tier1_f1_macro:.3f}')

    return auroc, metric_logger.acc1.global_avg, f1_macro, metric_logger.loss.global_avg, f1_pc, cm, tier1_f1_macro, tier1_acc1


if __name__ == '__main__':
    main()
