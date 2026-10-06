#!/usr/bin/env python
"""
Дві дешеві діагностики (лише інференс, без тренування):

 (A) ШУМОВА СТЕЛЯ. Oracle між двома cheap-моделями з РІЗНИМИ сідами (і між
     двома expensive). Якщо вона ≈ oracle(cheap, expensive)=0.819, то "стеля"
     -- це здебільшого шум розбіжностей двох слабо корельованих моделей, а не
     системна комплементарність cheap/expensive.

 (B) ЧИ ЩО-НЕБУДЬ РОБИТЬ TIER1.5. Для кожного сіда на тесті, з ОДНАКОВИМ
     детермінованим bag-розбиттям (bag_seed=1000+i, як у тренуванні):
        cheap       -- оригінальна cheap-модель (Tier2 з чекпоінта cheap)
        v3          -- Tier1.5 v3, top_k як у тренуванні (з ескалацією)
        v3_noesc    -- ТА САМА v3-модель (той самий Tier2), але top_k=0
                       => різниця v3 vs v3_noesc = чистий ефект delta
        ctrl        -- control (top_k=0, Tier2 донавчений)
     Друкуємо acc, частку слайдів зі зміненим прогнозом між парами та
     відносну норму delta: ||feats_out - cheap_feats|| / ||cheap_feats||.

Запуск (з кореня Hi-MoE_MIL):
    python -u diagnose_tier1_5.py \
        --config_cheap config/bracs_vits_dino_config_100ep.yml \
        --config_expensive config/bracs_uni2h_config.yml \
        --pattern_cheap 'ckpt/bracs_dtfd_v2_seed{seed}' \
        --pattern_expensive 'ckpt/bracs_dtfd_v2_uni2h_seed{seed}' \
        --dir_v3 'ckpt/bracs_v3_seed{seed}' \
        --dir_ctrl 'ckpt/bracs_ctrl_seed{seed}' \
        --n_class 3 --seeds 1 2 3 4 5 6 7 8 9 10 --n_repeats 3
"""
import argparse
import itertools
import os

import numpy as np
import torch

from validate_bag_level_routing import PairedFeatureDataset, load_conf, load_model
from dtfd_asmil_tier1_5 import DTFD_ASMIL_v2_5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def predict(model, ds, which, n_repeats):
    preds = np.zeros((n_repeats, len(ds)), dtype=int)
    y = np.zeros(len(ds), dtype=int)
    for i in range(len(ds)):
        it = ds[i]
        f = it[f'feats_{which}'].unsqueeze(0).to(device)
        c = it[f'coords_{which}'].unsqueeze(0).to(device)
        y[i] = it['label']
        for r in range(n_repeats):
            out, _, _ = model(f, coords=c)
            preds[r, i] = int(out.argmax(-1).item())
    return preds, y


def part_A(args, ds, conf_c, conf_e, M):
    print("=" * 70)
    print("(A) ШУМОВА СТЕЛЯ: oracle між моделями з різними сідами")
    P = {'cheap': {}, 'exp': {}}
    y = None
    for s in args.seeds:
        for kind, conf, pat, which in (('cheap', conf_c, args.pattern_cheap, 'cheap'),
                                       ('exp', conf_e, args.pattern_expensive, 'expensive')):
            cp = os.path.join(pat.format(seed=s), args.ckpt_name)
            if not os.path.exists(cp):
                print(f"[skip] {cp}")
                continue
            m = load_model(conf, cp, M)
            P[kind][s], y = predict(m, ds, which, args.n_repeats)
            del m
    out = {}
    for kind in ('cheap', 'exp'):
        ss = sorted(P[kind])
        orc, resc = [], []
        for a, b in itertools.combinations(ss, 2):
            ca, cb = (P[kind][a] == y), (P[kind][b] == y)
            orc.append((ca | cb).mean())
            resc.append((~ca & cb).mean())
        acc = np.mean([(P[kind][s] == y).mean() for s in ss])
        out[kind] = (acc, np.mean(orc), np.mean(resc), len(orc))
        print(f"  {kind}-vs-{kind} (різні сіди, {len(orc)} пар): "
              f"acc={acc:.4f}  oracle={np.mean(orc):.4f}  rescue(одного іншим)={np.mean(resc):.4f}")
    # cheap-vs-exp той самий сід (як у headroom_and_paired.py)
    orc, resc = [], []
    for s in sorted(set(P['cheap']) & set(P['exp'])):
        cc, ce = (P['cheap'][s] == y), (P['exp'][s] == y)
        orc.append((cc | ce).mean())
        resc.append((~cc & ce).mean())
    print(f"  cheap-vs-exp (той самий сід): oracle={np.mean(orc):.4f}  rescue={np.mean(resc):.4f}")
    print("  Інтерпретація: якщо cheap-vs-cheap oracle майже дорівнює cheap-vs-exp, "
          "то 'стеля' -- це шум, а не специфічна комплементарність UNI2h.")


def build_v25(conf_c, conf_e, cheap_ckpt, tier_ckpt_path, n_class, M, force_topk=None):
    from train_tier1_5 import load_cheap_model
    cheap = load_cheap_model(conf_c, cheap_ckpt, M)
    ck = torch.load(tier_ckpt_path, map_location=device, weights_only=False) if tier_ckpt_path else None
    a = ck['args'] if ck else {}
    model = DTFD_ASMIL_v2_5(
        cheap_model=cheap, D_feat_exp=conf_e.D_feat,
        D_inner_exp=a.get('d_inner_tier1_5', 128), n_token=conf_c.n_token,
        D_inner_cheap=conf_c.D_inner, n_class=n_class,
        top_k_bags=(force_topk if force_topk is not None else a.get('top_k_bags', 2)),
        n_heads=a.get('n_heads', 4), dropout=a.get('dropout', 0.3),
        train_tier2=a.get('train_tier2', False),
        context_mode=a.get('context_mode', 'attn'), d_ctx=a.get('d_ctx', 128)).to(device)
    if ck:
        model.tier1_5.load_state_dict(ck['tier1_5_state_dict'])
        if 'tier2_state_dict' in ck:
            model.cheap_model.tier2.load_state_dict(ck['tier2_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def eval_v25(model, ds, track_delta=False):
    ratios = []
    if track_delta:
        orig = model.tier1_5.forward

        def wrapped(exp_feats, others_feats, others_conf, cheap_this):
            fo, sp, bp = orig(exp_feats, others_feats, others_conf, cheap_this)
            ratios.append(((fo - cheap_this).norm() / (cheap_this.norm() + 1e-9)).item())
            return fo, sp, bp
        model.tier1_5.forward = wrapped
    preds, y, keep = [], [], []
    for i in range(len(ds)):
        it = ds[i]
        fc = it['feats_cheap'].unsqueeze(0).to(device)
        fe = it['feats_expensive'].unsqueeze(0).to(device)
        if fc.shape[1] != fe.shape[1]:
            continue
        co = it['coords_cheap'].unsqueeze(0).to(device)
        out, _, _ = model(fc, fe, coords_cheap=co, bag_seed=1000 + i)
        preds.append(int(out.argmax(-1).item()))
        y.append(it['label'])
        keep.append(i)
    if track_delta:
        model.tier1_5.forward = orig
    return np.array(preds), np.array(y), keep, ratios


def part_B(args, ds, conf_c, conf_e, M):
    print("=" * 70)
    print("(B) Чи робить Tier1.5 щось: однакове bag-розбиття, тест")
    rows = []
    for s in args.seeds:
        cheap_ck = os.path.join(args.pattern_cheap.format(seed=s), args.ckpt_name)
        v3_ck = os.path.join(args.dir_v3.format(seed=s), 'checkpoint-best.pth')
        ct_ck = os.path.join(args.dir_ctrl.format(seed=s), 'checkpoint-best.pth')
        if not all(os.path.exists(p) for p in (cheap_ck, v3_ck, ct_ck)):
            print(f"[skip] seed{s}")
            continue
        m = build_v25(conf_c, conf_e, cheap_ck, None, args.n_class, M, force_topk=0)
        p_cheap, y, keep, _ = eval_v25(m, ds); del m
        m = build_v25(conf_c, conf_e, cheap_ck, v3_ck, args.n_class, M)
        p_v3, _, _, ratios = eval_v25(m, ds, track_delta=True)
        m.top_k_bags = 0
        p_v3n, _, _, _ = eval_v25(m, ds); del m
        m = build_v25(conf_c, conf_e, cheap_ck, ct_ck, args.n_class, M, force_topk=0)
        p_ct, _, _, _ = eval_v25(m, ds); del m
        r = dict(acc_cheap=(p_cheap == y).mean(), acc_v3=(p_v3 == y).mean(),
                 acc_v3n=(p_v3n == y).mean(), acc_ct=(p_ct == y).mean(),
                 flip_delta=(p_v3 != p_v3n).mean(),
                 flip_v3_cheap=(p_v3 != p_cheap).mean(),
                 flip_ct_cheap=(p_ct != p_cheap).mean(),
                 flip_v3_ct=(p_v3 != p_ct).mean(),
                 delta_ratio=float(np.mean(ratios)) if ratios else float('nan'))
        rows.append(r)
        print(f"seed{s:>2}: acc cheap={r['acc_cheap']:.3f} v3={r['acc_v3']:.3f} "
              f"v3_noesc={r['acc_v3n']:.3f} ctrl={r['acc_ct']:.3f} | "
              f"flip(v3 vs v3_noesc)={r['flip_delta']:.3f} flip(v3 vs ctrl)={r['flip_v3_ct']:.3f} "
              f"flip(ctrl vs cheap)={r['flip_ct_cheap']:.3f} | ||delta||/||feat||={r['delta_ratio']:.4f}")
    if rows:
        g = lambda k: np.mean([r[k] for r in rows])
        print("\n  СЕРЕДНЄ ПО СІДАХ:")
        print(f"  acc: cheap={g('acc_cheap'):.4f}  v3={g('acc_v3'):.4f}  "
              f"v3_noesc={g('acc_v3n'):.4f}  ctrl={g('acc_ct'):.4f}")
        print(f"  частка слайдів зі зміненим прогнозом через delta (v3 vs v3_noesc): {g('flip_delta'):.4f}")
        print(f"  v3 vs ctrl: {g('flip_v3_ct'):.4f}   ctrl vs cheap (ефект fine-tune Tier2): {g('flip_ct_cheap'):.4f}")
        print(f"  відносна норма delta: {g('delta_ratio'):.4f}")
        print("  Інтерпретація: flip(v3 vs v3_noesc)≈0 і delta_ratio≈0 => модуль лишився "
              "тотожністю (мертвий). Якщо flip помітний, але acc не росте -- delta змінює "
              "прогнози, але не в правильний бік.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_cheap", required=True)
    ap.add_argument("--config_expensive", required=True)
    ap.add_argument("--pattern_cheap", required=True, help="з {seed}")
    ap.add_argument("--pattern_expensive", required=True, help="з {seed}")
    ap.add_argument("--dir_v3", required=True, help="з {seed}")
    ap.add_argument("--dir_ctrl", required=True, help="з {seed}")
    ap.add_argument("--n_class", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument("--ckpt_name", default="checkpoint-best.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_token", type=int, default=8)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--only", choices=["A", "B", "AB"], default="AB")
    args = ap.parse_args()

    conf_c = load_conf(args.config_cheap, "medical_ssl", args.n_token)
    conf_e = load_conf(args.config_expensive, "UNI2h", args.n_token)
    M = args.n_token
    split_csv = os.path.join(conf_c.splits_dir, f'{args.split}.csv')
    fd_c = conf_c.test_dir if args.split == 'test' else conf_c.train_dir
    fd_e = conf_e.test_dir if args.split == 'test' else conf_e.train_dir
    ds = PairedFeatureDataset(split_csv, fd_c, fd_e)
    print(f"слайдів: {len(ds)}, сіди: {args.seeds}")
    if 'A' in args.only:
        part_A(args, ds, conf_c, conf_e, M)
    if 'B' in args.only:
        part_B(args, ds, conf_c, conf_e, M)


if __name__ == "__main__":
    main()
