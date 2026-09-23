"""Full evaluation + figure generation for results.md."""
import json, os, random
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from dataset import SurfaceDefectDetectionDataset, partitioning
from device import DEVICE
from unet import UNet_2D

OUT = 'results'
os.makedirs(OUT, exist_ok=True)

# ---- palette (validated reference instance, light mode) ----------------
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8880'
SURF = '#fcfcfb'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': '#d8d7d2', 'axes.linewidth': 1.0,
    'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'medium',
    'grid.color': '#eceae5', 'grid.linewidth': 1.0,
    'legend.frameon': False, 'figure.dpi': 130,
})

def style(ax, grid='y'):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    if grid: ax.grid(axis=grid)
    return ax

# ---- rebuild the exact test split -------------------------------------
random.seed(51); np.random.seed(51); torch.manual_seed(51)
partition = partitioning([0.70, 0.10, 0.20])
paths = partition['test']
ds = SurfaceDefectDetectionDataset(paths, 'test')

model = UNet_2D(1, 1, 32, 0.2).to(DEVICE)
model.load_state_dict(torch.load('model.pt', map_location=DEVICE))
model.eval()

probs, gts, classes = [], [], []
with torch.no_grad():
    for i, p in enumerate(paths):
        img, msk = ds[i]
        out = model(img.unsqueeze(0).to(DEVICE))
        probs.append(out.cpu().numpy().squeeze())
        gts.append(msk.cpu().numpy().squeeze())
        classes.append(p.split('/')[1].replace('MT_', ''))
probs = np.stack(probs); gts = np.stack(gts); classes = np.array(classes)
np.save(f'{OUT}/_probs.npy', probs.astype(np.float16))

SM, B2 = 1e-6, 0.3

def metrics_from(tp, fp, tn, fn, ae, n):
    spec = tn / (tn + fp + SM); sens = tp / (tp + fn + SM)
    prec = tp / (tp + fp + SM)
    f1 = (2 * prec * sens + SM) / (prec + sens + SM)
    f2 = (5 * tp + SM) / (5 * tp + 4 * fn + fp + SM)
    dsc = (2 * tp + SM) / (2 * tp + fn + fp + SM)
    fb = ((1 + B2) * prec * sens + SM) / (B2 * prec + sens + SM)
    return dict(specificity=spec, sensitivity=sens, precision=prec, F1_score=f1,
                F2_score=f2, DSC=dsc, F_beta=fb, MAE=ae / n, acc=(tp + tn) / n)

def counts(pred, gt):
    tp = float((pred * gt).sum()); tn = float(((1 - pred) * (1 - gt)).sum())
    fp = float((pred * (1 - gt)).sum()); fn = float(((1 - pred) * gt).sum())
    return tp, fp, tn, fn

# ---- per-image metrics @ 0.5 ------------------------------------------
TH = 0.5
rows = []
for i in range(len(paths)):
    pr = (probs[i] > TH).astype(np.float64); gt = gts[i].astype(np.float64)
    tp, fp, tn, fn = counts(pr, gt)
    n = gt.size
    m = metrics_from(tp, fp, tn, fn, np.abs(pr - gt).sum(), n)
    m.update(image=os.path.basename(paths[i]).replace('.jpg', ''), cls=classes[i],
             defect_px=int(gt.sum()), pred_px=int(pr.sum()),
             MAE_prob=float(np.abs(probs[i] - gt).mean()),
             defective=bool(gt.sum() > 0))
    rows.append(m)
df = pd.DataFrame(rows)
COLS = ['specificity', 'sensitivity', 'precision', 'F1_score', 'F2_score',
        'DSC', 'F_beta', 'MAE', 'acc']

# The smoothing term makes F_beta/F1/DSC return 1.0 for two degenerate cases:
# a total miss (gt>0, pred=0) and a pure false alarm (gt=0, pred>0), because
# numerator and denominator both collapse to `smooth`. Score those honestly.
def _corrected(r):
    if r.defect_px == 0 and r.pred_px == 0: return 1.0   # nothing there, none found
    if r.defect_px == 0 or r.pred_px == 0:  return 0.0   # total miss / false alarm
    return r.F_beta

df['F_beta_corr'] = df.apply(_corrected, axis=1)
df['degenerate'] = (df.F_beta.round(6) == 1.0) & ~((df.defect_px == 0) & (df.pred_px == 0))
df[['image', 'cls', 'defect_px', 'pred_px'] + COLS +
   ['MAE_prob', 'F_beta_corr', 'degenerate']].to_csv(
    f'{OUT}/per_image_metrics.csv', index=False)

# ---- pooled confusion matrix @ 0.5 ------------------------------------
pr_all = (probs > TH).astype(np.float64); gt_all = gts.astype(np.float64)
TP, FP, TN, FN = counts(pr_all, gt_all)
N = gt_all.size
pooled = metrics_from(TP, FP, TN, FN, np.abs(pr_all - gt_all).sum(), N)
pooled['MAE_prob'] = float(np.abs(probs - gts).mean())

# ---- threshold sweep --------------------------------------------------
ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
sweep = []
for t in ths:
    pr = (probs > t).astype(np.float64)
    tp, fp, tn, fn = counts(pr, gt_all)
    m = metrics_from(tp, fp, tn, fn, np.abs(pr - gt_all).sum(), N)
    per_img = []
    for i in range(len(paths)):
        p1 = (probs[i] > t).astype(np.float64); g1 = gts[i].astype(np.float64)
        a, b, c, d = counts(p1, g1)
        per_img.append(metrics_from(a, b, c, d, np.abs(p1 - g1).sum(), g1.size)['F_beta'])
    m.update(threshold=float(t), F_beta_img_mean=float(np.mean(per_img)),
             F_beta_img_mean_defective=float(np.mean(np.array(per_img)[df.defective.values])))
    sweep.append(m)
sw = pd.DataFrame(sweep)
sw.to_csv(f'{OUT}/threshold_sweep.csv', index=False)

best_pool = sw.loc[sw.F_beta.idxmax()]
best_img = sw.loc[sw.F_beta_img_mean.idxmax()]

# ---- PR curve (pooled pixels) -----------------------------------------
fl_p, fl_g = probs.ravel().astype(np.float32), gts.ravel().astype(np.float32)
order = np.argsort(-fl_p)
gs = fl_g[order]
tps = np.cumsum(gs); fps = np.cumsum(1 - gs)
prec_c = tps / np.maximum(tps + fps, 1e-9)
rec_c = tps / max(gs.sum(), 1e-9)
step = max(1, len(prec_c) // 4000)
prec_c, rec_c = prec_c[::step], rec_c[::step]
# recall is monotonically increasing along the sorted list -> integrate directly
ap = float(np.trapezoid(prec_c, rec_c))

# ---- per-class ---------------------------------------------------------
percls = df.groupby('cls')[COLS].mean().round(4)
percls.insert(0, 'n', df.groupby('cls').size())
percls['F_beta_corr'] = df.groupby('cls').F_beta_corr.mean().round(4)
percls.to_csv(f'{OUT}/per_class_metrics.csv')

# =======================  FIGURES  =====================================
loss = pd.read_csv('loss_epoch.csv')

# fig 1 — loss curves
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(loss.epoch, loss['Training Loss'], color=BLUE, lw=2, label='Training')
ax.plot(loss.epoch, loss['Validation Loss'], color=ORANGE, lw=2, label='Validation')
bi = int(loss['Validation Loss'].idxmin())
bx, by = loss.epoch[bi], loss['Validation Loss'][bi]
ax.scatter([bx], [by], s=42, color=ORANGE, zorder=5, ec=SURF, lw=2)
ax.annotate(f'best val {by:.4f}\nepoch {int(bx)}', (bx, by), textcoords='offset points',
            xytext=(-12, 26), ha='right', fontsize=9, color=INK2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Tversky loss')
ax.set_title('Training and validation loss')
ax.legend(loc='upper right'); style(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_loss_curves.png'); plt.close(fig)

# fig 2 — confusion matrix (row-normalised colour, raw counts annotated)
cm = np.array([[TN, FP], [FN, TP]])
cmn = cm / cm.sum(axis=1, keepdims=True)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seqblue', SEQ)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,.0f}\n{cmn[i,j]*100:.2f}%', ha='center', va='center',
                color='white' if cmn[i, j] > 0.5 else INK, fontsize=10)
ax.set_xticks([0, 1], ['flawless', 'defective'])
ax.set_yticks([0, 1], ['flawless', 'defective'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Ground truth')
ax.set_title(f'Pixel confusion matrix @ threshold {TH}\n{N:,} test pixels, row-normalised')
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.outline.set_visible(False); cb.set_label('share of true class', color=INK2)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_confusion_matrix.png'); plt.close(fig)

# fig 3 — threshold sweep
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(sw.threshold, sw.F_beta, color=BLUE, lw=2, label=r'$F_\beta$ (pooled pixels)')
ax.plot(sw.threshold, sw.precision, color=ORANGE, lw=2, label='Precision')
ax.plot(sw.threshold, sw.sensitivity, color=AQUA, lw=2, label='Sensitivity (recall)')
ax.scatter([best_pool.threshold], [best_pool.F_beta], s=42, color=BLUE, zorder=5, ec=SURF, lw=2)
ax.annotate(f'max $F_\\beta$ {best_pool.F_beta:.4f} @ {best_pool.threshold:.2f}',
            (best_pool.threshold, best_pool.F_beta), textcoords='offset points',
            xytext=(-8, -26), ha='right', fontsize=9, color=INK2)
ax.set_xlabel('Binarisation threshold'); ax.set_ylabel('Score')
ax.set_title(r'Metric response to threshold ($\beta^2=0.3$)')
ax.set_ylim(0, 1.02); ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.legend(loc='lower left'); style(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_threshold_sweep.png'); plt.close(fig)

# fig 4 — PR curve
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(rec_c, prec_c, color=BLUE, lw=2)
ax.fill_between(rec_c, prec_c, color=BLUE, alpha=0.10, lw=0)
base = gs.sum() / len(gs)
ax.axhline(base, color=MUTED, lw=1.2, ls='--')
ax.annotate(f'chance = {base:.4f}', (0.02, base), textcoords='offset points',
            xytext=(0, 6), fontsize=9, color=INK2)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'Pixel-level precision–recall  (AP = {ap:.3f})')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); style(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_pr_curve.png'); plt.close(fig)

# fig 5 — per-image F_beta by class
fig, ax = plt.subplots(figsize=(7.2, 3.6))
cls_order = ['Blowhole', 'Crack', 'Free']
cols = {'Blowhole': BLUE, 'Crack': ORANGE, 'Free': AQUA}
rng = np.random.default_rng(0)
for k, c in enumerate(cls_order):
    v = df[df.cls == c].F_beta.values
    ax.scatter(rng.normal(k, 0.07, len(v)), v, s=46, color=cols[c],
               alpha=0.85, ec=SURF, lw=1.5, zorder=3)
    ax.plot([k - 0.26, k + 0.26], [v.mean()] * 2, color=INK, lw=2, zorder=4)
    ax.annotate(f'mean {v.mean():.3f}', (k + 0.30, v.mean()), fontsize=9,
                color=INK2, va='center')
ax.set_xticks(range(3), [f'{c}\n(n={int((df.cls==c).sum())})' for c in cls_order])
ax.set_ylabel(r'$F_\beta$'); ax.set_xlim(-0.5, 3.25); ax.set_ylim(-0.03, 1.05)
ax.set_title(r'Per-image $F_\beta$ by defect class @ threshold 0.5')
style(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_per_class_fbeta.png'); plt.close(fig)

# fig 9 — metric degeneracy: as-implemented vs corrected
fig, ax = plt.subplots(figsize=(7.2, 3.6))
groups = ['All test\nimages', 'Defective\nonly', 'Blowhole', 'Crack', 'Free']
masks = [np.ones(len(df), bool), df.defective.values,
         (df.cls == 'Blowhole').values, (df.cls == 'Crack').values,
         (df.cls == 'Free').values]
asis = [df.F_beta[m].mean() for m in masks]
corr = [df.F_beta_corr[m].mean() for m in masks]
x = np.arange(len(groups)); w = 0.36
b1 = ax.bar(x - w/2 - 0.01, asis, w, color=BLUE, label='As implemented')
b2 = ax.bar(x + w/2 + 0.01, corr, w, color=ORANGE, label='Degenerate cases scored 0')
for bars, vals in ((b1, asis), (b2, corr)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
                ha='center', fontsize=8.5, color=INK2)
ax.set_xticks(x, groups); ax.set_ylabel(r'mean $F_\beta$'); ax.set_ylim(0, 1.15)
ax.set_title(r'Effect of the smoothing degeneracy on mean $F_\beta$')
ax.legend(loc='upper right', ncols=2); style(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig9_metric_degeneracy.png'); plt.close(fig)

# fig 6/7/8 — qualitative grids
def _img(i):
    im, _ = ds[i]
    return im.cpu().numpy().squeeze()

def qual(cls, fname, title, n=4, want_defect=True):
    idx = [i for i in range(len(paths)) if classes[i] == cls
           and ((gts[i].sum() > 0) == want_defect)][:n]
    fig, axes = plt.subplots(len(idx), 3, figsize=(7.4, 2.55 * len(idx)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        pr = (probs[i] > TH).astype(float)
        for c, arr in enumerate([_img(i), gts[i], pr]):
            ax = axes[r, c]; ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color('#d8d7d2')
        axes[r, 0].set_ylabel(f'{df.iloc[i]["image"][:14]}\n$F_\\beta$={df.iloc[i].F_beta:.3f}',
                              fontsize=8, color=INK2)
    for c, t in enumerate(['input image', 'ground truth', f'prediction @ {TH}']):
        axes[0, c].set_title(t, fontsize=10)
    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}'); plt.close(fig)
    return idx

qual('Blowhole', 'fig6_blowhole_predictions.png', 'Blowhole — test predictions')
qual('Crack', 'fig7_crack_predictions.png', 'Crack — test predictions')
qual('Free', 'fig8_free_predictions.png', 'Free (defect-free) — test predictions',
     n=4, want_defect=False)

# ---- dump numbers for the writeup -------------------------------------
summary = dict(
    device=str(DEVICE), threshold=TH, n_test=len(paths),
    n_defective=int(df.defective.sum()), n_free=int((~df.defective).sum()),
    total_pixels=int(N), defect_pixels=int(gt_all.sum()),
    defect_fraction=float(gt_all.sum() / N),
    pooled={k: float(v) for k, v in pooled.items()},
    cm=dict(TN=TN, FP=FP, FN=FN, TP=TP),
    img_mean={k: float(df[k].mean()) for k in COLS},
    img_mean_defective={k: float(df[df.defective][k].mean()) for k in COLS},
    img_mean_MAE_prob=float(df.MAE_prob.mean()),
    F_beta_corr_all=float(df.F_beta_corr.mean()),
    F_beta_corr_defective=float(df[df.defective].F_beta_corr.mean()),
    n_degenerate=int(df.degenerate.sum()),
    degenerate_images=df[df.degenerate]['image'].tolist(),
    best_pooled=dict(threshold=float(best_pool.threshold), F_beta=float(best_pool.F_beta),
                     MAE=float(best_pool.MAE), precision=float(best_pool.precision),
                     sensitivity=float(best_pool.sensitivity)),
    best_img=dict(threshold=float(best_img.threshold),
                  F_beta_img_mean=float(best_img.F_beta_img_mean),
                  MAE=float(best_img.MAE)),
    ap=ap,
    best_epoch=int(loss.epoch[bi]), best_val=float(by),
    final_train=float(loss['Training Loss'].iloc[-1]),
    final_val=float(loss['Validation Loss'].iloc[-1]),
    first_train=float(loss['Training Loss'].iloc[0]),
    first_val=float(loss['Validation Loss'].iloc[0]),
)
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print('\nper-class:\n', percls.to_string())
print('\nsweep:\n', sw[['threshold', 'precision', 'sensitivity', 'F_beta',
                        'F_beta_img_mean', 'MAE']].round(4).to_string(index=False))
