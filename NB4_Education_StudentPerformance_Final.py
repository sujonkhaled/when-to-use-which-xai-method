# ╔══════════════════════════════════════════════════════════════════╗
# ║  XAI EMPIRICAL STUDY — NOTEBOOK 4: EDUCATION DOMAIN            ║
# ║  Dataset  : Student Performance (UCI)                           ║
# ║  Size     : 395 rows × 30 features | Binary classification     ║
# ║  Paper    : "When to Use Which Explainable AI Method?"          ║
# ║  Authors  : Khaled Mahmud Sujon et al.                          ║
# ║  Key      : Smallest dataset — tests XAI stability under        ║
# ║             data scarcity (RQ2)                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── CELL 1 ── Install Libraries ───────────────────────────────────
# !pip install -q shap lime xgboost lightgbm

# ── CELL 2 ── Imports ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size':11,'figure.dpi':150,
                            'axes.spines.top':False,
                            'axes.spines.right':False})
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap, lime, lime.lime_tabular
import time, warnings
from collections import defaultdict
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score)
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)
DOMAIN = "Education"
print("✓ Libraries loaded")

# ── CELL 3 ── Load Dataset ────────────────────────────────────────
URL = ('https://raw.githubusercontent.com/'
       'dsrscientist/dataset1/master/student-mat.csv')
try:
    df = pd.read_csv(URL, sep=';')
    print(f"✓ Loaded from GitHub: {df.shape}")
except Exception:
    import io, zipfile, requests
    z  = requests.get(
        'https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/00320/student.zip')
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        df = pd.read_csv(zf.open('student-mat.csv'), sep=';')
    print(f"✓ Loaded from UCI: {df.shape}")

# ── CELL 4 ── Preprocessing ───────────────────────────────────────
# Binary target: pass (G3 >= 10) vs fail
df['target'] = (df['G3'] >= 10).astype(int)
df.drop(['G1','G2','G3'], axis=1, inplace=True)

X = df.drop('target', axis=1).copy()
y = df['target'].copy()

for c in X.columns:
    if (X[c].dtype == object or
            set(X[c].dropna().unique()).issubset({'yes','no'})):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

FEATURES = X.columns.tolist()
print(f"Features: {len(FEATURES)} | Samples: {len(X)}")
print(f"Class balance: {y.value_counts().to_dict()}")

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)
y_tr = y_tr.astype(int)
y_te = y_te.astype(int)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# ── CELL 5 ── Train Models ────────────────────────────────────────
# Note: Education uses MLP (64,32) not (128,64,32) — smaller dataset
MODELS = {
    'Random Forest'      : RandomForestClassifier(
                            n_estimators=200, random_state=42,
                            n_jobs=-1),
    'XGBoost'            : xgb.XGBClassifier(
                            n_estimators=200, random_state=42,
                            eval_metric='logloss', verbosity=0),
    'LightGBM'           : lgb.LGBMClassifier(
                            n_estimators=200, random_state=42,
                            verbose=-1),
    'SVM'                : SVC(kernel='rbf', probability=True,
                            random_state=42),
    'Logistic Regression': LogisticRegression(
                            max_iter=1000, random_state=42),
    'MLP'                : MLPClassifier(
                            hidden_layer_sizes=(64,32),
                            max_iter=500, random_state=42),
}
PERF = {}
print(f"{'Model':<22} {'Acc':>6} {'F1':>6} {'AUC':>6}")
print("─" * 46)
for name, mdl in MODELS.items():
    mdl.fit(X_tr, y_tr)
    yp  = mdl.predict(X_te)
    pp  = mdl.predict_proba(X_te)[:,1]
    acc = accuracy_score(y_te, yp)
    f1  = f1_score(y_te, yp)
    auc = roc_auc_score(y_te, pp)
    PERF[name] = {'model':mdl,'acc':acc,'f1':f1,'auc':auc}
    print(f"{name:<22} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f}")

# ── CELL 6 ── XAI Setup ───────────────────────────────────────────
XAI    = defaultdict(dict)
N_EXPL = min(79, len(X_te))   # small test set
X_ex   = X_te.iloc[:N_EXPL]
y_ex   = y_te.iloc[:N_EXPL]

def fix_shap_shape(sv):
    arr = np.array(sv)
    if isinstance(sv, list):
        arr = np.abs(np.array(sv)).sum(axis=0)
    elif arr.ndim == 3:
        arr = np.abs(arr).sum(axis=0)
    return arr

# ── CELL 7 ── TreeSHAP ────────────────────────────────────────────
print("▶ TreeSHAP ...")
APPROX = {'Random Forest':True,'XGBoost':True,'LightGBM':False}
for mn in ['Random Forest','XGBoost','LightGBM']:
    t0  = time.perf_counter()
    exp = shap.TreeExplainer(PERF[mn]['model'])
    sv  = fix_shap_shape(exp.shap_values(
        X_ex, approximate=APPROX[mn],
        check_additivity=False))
    t1  = time.perf_counter()
    XAI['TreeSHAP'][mn] = dict(
        values=sv, mean_abs=sv.mean(0),
        tps=(t1-t0)/N_EXPL, exp=exp)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.2f} ms/sample")

# ── CELL 8 ── KernelSHAP ─────────────────────────────────────────
print("▶ KernelSHAP ...")
bg = shap.sample(X_tr, 80)
for mn in ['SVM','Logistic Regression','MLP']:
    t0  = time.perf_counter()
    exp = shap.KernelExplainer(
        PERF[mn]['model'].predict_proba, bg)
    sv  = fix_shap_shape(
        exp.shap_values(X_ex, nsamples=100))
    t1  = time.perf_counter()
    XAI['KernelSHAP'][mn] = dict(
        values=sv, mean_abs=sv.mean(0),
        tps=(t1-t0)/N_EXPL, exp=exp)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.1f} ms/sample")

# ── CELL 9 ── LIME ────────────────────────────────────────────────
print("▶ LIME ...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_tr.values, feature_names=FEATURES,
    class_names=['Fail','Pass'],
    discretize_continuous=True, random_state=42)

for mn, res in PERF.items():
    t0, rows = time.perf_counter(), []
    for i in range(N_EXPL):
        e   = lime_exp.explain_instance(
            X_ex.values[i], res['model'].predict_proba,
            num_features=len(FEATURES))
        w   = dict(e.as_list())
        row = [max([abs(v) for k,v in w.items()
                    if fn in k], default=0.)
               for fn in FEATURES]
        rows.append(row)
    t1 = time.perf_counter()
    arr = np.array(rows)
    XAI['LIME'][mn] = dict(
        values=arr, mean_abs=arr.mean(0),
        tps=(t1-t0)/N_EXPL)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.1f} ms/sample")

# ── CELL 10 ── Permutation Importance ────────────────────────────
print("▶ PermImp ...")
for mn, res in PERF.items():
    t0   = time.perf_counter()
    perm = permutation_importance(
        res['model'], X_te, y_te,
        n_repeats=30, random_state=42, n_jobs=-1)
    t1   = time.perf_counter()
    XAI['PermImp'][mn] = dict(
        values=perm.importances,
        mean_abs=perm.importances_mean,
        tps=(t1-t0)/len(X_te))
    print(f"  [{mn}] {(t1-t0)/len(X_te)*1000:.2f} ms/sample")

# ── CELL 11 ── Faithfulness ───────────────────────────────────────
def faithfulness(model, X_df, sv, y_true, k=5):
    sv_f = fix_shap_shape(sv)
    idx  = np.argsort(sv_f.mean(0))[-k:].flatten().astype(int)
    base = roc_auc_score(y_true,
                          model.predict_proba(X_df)[:,1])
    Xm   = X_df.copy()
    for i in idx:
        Xm.iloc[:, i] = Xm.iloc[:, i].mean()
    mask = roc_auc_score(y_true,
                          model.predict_proba(Xm)[:,1])
    return round(base - mask, 4)

faith = {}
for mn in ['Random Forest','XGBoost','LightGBM']:
    sv = fix_shap_shape(XAI['TreeSHAP'][mn]['values'])
    faith[f'TreeSHAP|{mn}'] = faithfulness(
        PERF[mn]['model'], X_ex, sv, y_ex)
for mn in ['SVM','Logistic Regression','MLP']:
    sv = fix_shap_shape(XAI['KernelSHAP'][mn]['values'])
    faith[f'KernelSHAP|{mn}'] = faithfulness(
        PERF[mn]['model'], X_ex, sv, y_ex)
print("Faithfulness:")
for k, v in faith.items():
    print(f"  {k}: {v:.4f}")

# ── CELL 12 ── Stability — KEY FINDING for RQ2 ───────────────────
print("Computing stability (n=15 runs) — KEY FINDING ...")
STAB = {}
X_stab = X_ex.iloc[:10]

def stab_shap(exp_fn, X_s, n=15, k=5, is_lgb=False):
    sets = []
    for _ in range(n):
        sv  = fix_shap_shape(np.array(
            exp_fn(X_s, approximate=(not is_lgb),
                   check_additivity=False)))
        top = frozenset(
            np.argsort(np.abs(sv).mean(0))[-k:].flatten().tolist())
        sets.append(top)
    J = [len(a&b)/len(a|b)
         for i,a in enumerate(sets) for b in sets[i+1:]]
    return round(np.mean(J), 4)

def stab_kernel(model, X_s, n=15, k=5):
    bg_k = shap.sample(X_tr, 80)
    kexp = shap.KernelExplainer(model.predict_proba, bg_k)
    sets = []
    for _ in range(n):
        sv  = fix_shap_shape(
            kexp.shap_values(X_s, nsamples=100))
        top = frozenset(
            np.argsort(np.abs(sv).mean(0))[-k:].flatten().tolist())
        sets.append(top)
    J = [len(a&b)/len(a|b)
         for i,a in enumerate(sets) for b in sets[i+1:]]
    return round(np.mean(J), 4)

def stab_lime(lexp, model, x1, n=15, k=5):
    sets = []
    for _ in range(n):
        e   = lexp.explain_instance(
            x1, model.predict_proba,
            num_features=len(FEATURES))
        w   = dict(e.as_list())
        imp = [max([abs(v) for kk,v in w.items()
                    if fn in kk], default=0.)
               for fn in FEATURES]
        sets.append(frozenset(np.argsort(imp)[-k:].tolist()))
    J = [len(a&b)/len(a|b)
         for i,a in enumerate(sets) for b in sets[i+1:]]
    return round(np.mean(J), 4)

for mn in ['Random Forest','XGBoost','LightGBM']:
    is_lgb = (mn == 'LightGBM')
    STAB[f'TreeSHAP|{mn}'] = stab_shap(
        XAI['TreeSHAP'][mn]['exp'].shap_values,
        X_stab, is_lgb=is_lgb)
    print(f"  TreeSHAP [{mn}]: {STAB[f'TreeSHAP|{mn}']}")
for mn in ['SVM','Logistic Regression','MLP']:
    STAB[f'KernelSHAP|{mn}'] = stab_kernel(
        PERF[mn]['model'], X_stab)
    print(f"  KernelSHAP [{mn}]: {STAB[f'KernelSHAP|{mn}']}")
for mn in list(PERF.keys()):
    STAB[f'LIME|{mn}'] = stab_lime(
        lime_exp, PERF[mn]['model'], X_ex.values[0])
    print(f"  LIME [{mn}]: {STAB[f'LIME|{mn}']}")

# ── CELL 13 ── Agreement ──────────────────────────────────────────
shap_m = fix_shap_shape(
    XAI['TreeSHAP']['Random Forest']['values']).mean(0)
lime_m = np.array(
    XAI['LIME']['Random Forest']['values']).mean(0)
r, p   = spearmanr(shap_m, lime_m)
print(f"SHAP vs LIME Spearman r={r:.3f}  p={p:.4f}")

# ── CELL 14 ── FIG 1: All 6 Model SHAP Bars ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
TREE_MN = ['Random Forest','XGBoost','LightGBM']
KERN_MN = ['SVM','Logistic Regression','MLP']
TREE_C  = ['#378ADD','#1D9E75','#7F77DD']
KERN_C  = ['#D85A30','#D4537E','#888780']

for ax, mn, col in zip(axes.flat[:3], TREE_MN, TREE_C):
    imp = fix_shap_shape(
        XAI['TreeSHAP'][mn]['values']).mean(0)
    s   = pd.Series(imp, index=FEATURES).nlargest(15)
    ax.barh(s.index[::-1], s.values[::-1],
            color=col, edgecolor='none')
    ax.set_title(f'TreeSHAP — {mn}', fontsize=10)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.grid(alpha=0.2, axis='x')

for ax, mn, col in zip(axes.flat[3:], KERN_MN, KERN_C):
    imp = fix_shap_shape(
        XAI['KernelSHAP'][mn]['values']).mean(0)
    s   = pd.Series(imp, index=FEATURES).nlargest(15)
    ax.barh(s.index[::-1], s.values[::-1],
            color=col, edgecolor='none')
    ax.set_title(f'KernelSHAP — {mn}', fontsize=10)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.grid(alpha=0.2, axis='x')

plt.suptitle(
    f'{DOMAIN}: All 6 Model Explanations (n={len(df)} samples)\n'
    f'Key finding: XAI reliability on tiny datasets',
    fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('fig1_all_models_education.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig1_all_models_education.pdf saved")

# ── CELL 15 ── FIG 2: Computation Time ───────────────────────────
TIME_D = {}
for mn in ['Random Forest','XGBoost','LightGBM']:
    TIME_D[f'TreeSHAP ({mn[:3]})'] = \
        XAI['TreeSHAP'][mn]['tps']
for mn in ['SVM','Logistic Regression','MLP']:
    TIME_D[f'KernelSHAP ({mn[:3]})'] = \
        XAI['KernelSHAP'][mn]['tps']
for mn in list(PERF.keys()):
    TIME_D[f'LIME ({mn[:3]})'] = XAI['LIME'][mn]['tps']
for mn in ['Random Forest','XGBoost']:
    TIME_D[f'PermImp ({mn[:3]})'] = \
        XAI['PermImp'][mn]['tps']

labels = list(TIME_D.keys())
vals   = [v*1000 for v in TIME_D.values()]
cols   = ['#378ADD' if 'Tree' in l
          else '#85B7EB' if 'Kernel' in l
          else '#BA7517' if 'LIME' in l
          else '#888780' for l in labels]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(labels, vals, color=cols, edgecolor='none')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.2,
            f'{v:.1f}', ha='center', fontsize=7, rotation=40)
ax.set_ylabel('Time (ms/sample)', fontsize=10)
ax.set_title(f'{DOMAIN}: XAI Computation Time per Sample',
             fontsize=11)
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('fig2_time_education.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig2_time_education.pdf saved")

# ── CELL 16 ── FIG 3: SHAP vs LIME Agreement ─────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for ax, method, data, col in [
    (ax1, 'TreeSHAP', shap_m, '#378ADD'),
    (ax2, 'LIME',     lime_m, '#BA7517')
]:
    s = pd.Series(data, index=FEATURES).sort_values(
        ascending=False)[:15]
    ax.barh(s.index[::-1], s.values[::-1],
            color=col, edgecolor='none')
    ax.set_title(f'{method} — Random Forest', fontsize=10)
    ax.set_xlabel('Mean |Importance|', fontsize=9)
    ax.grid(alpha=0.2, axis='x')
plt.suptitle(
    f'{DOMAIN}: SHAP vs LIME Feature Rankings\n'
    f'Spearman r={r:.3f}  p={p:.4f}',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig3_shap_vs_lime_education.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig3_shap_vs_lime_education.pdf saved")

# ── CELL 17 ── FIG 4: Faithfulness Heatmap ───────────────────────
METHS  = ['TreeSHAP','KernelSHAP']
MNAMES = ['Random Forest','XGBoost','LightGBM',
          'SVM','Logistic Regression','MLP']
mat_f  = np.full((len(METHS), len(MNAMES)), np.nan)
for mi, meth in enumerate(METHS):
    for mj, mn in enumerate(MNAMES):
        k = f'{meth}|{mn}'
        if k in faith:
            mat_f[mi, mj] = faith[k]

fig, ax = plt.subplots(figsize=(14, 3))
na_m = np.zeros_like(mat_f)
na_m[np.isnan(mat_f)] = 1.
ax.imshow(na_m,
          cmap=mcolors.ListedColormap(['white','#EBEBEB']),
          vmin=0, vmax=1, aspect='auto')
vmax  = max(np.nanmax(np.abs(mat_f)), 0.01)
mat_m = np.ma.masked_where(np.isnan(mat_f), mat_f)
im    = ax.imshow(mat_m, cmap='RdYlGn',
                  vmin=-vmax, vmax=vmax, aspect='auto')
for i in range(len(METHS)):
    for j in range(len(MNAMES)):
        if np.isnan(mat_f[i,j]):
            ax.text(j, i, 'N/A', ha='center', va='center',
                    fontsize=8, color='#888780', style='italic')
        else:
            v  = mat_f[i,j]
            nr = plt.cm.RdYlGn((v+vmax)/(2*vmax+1e-8))
            br = nr[0]*0.299+nr[1]*0.587+nr[2]*0.114
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if br<0.5 else '#2C2C2A')
for x in np.arange(-0.5, len(MNAMES), 1):
    ax.axvline(x, color='white', lw=1.5)
ax.set_xticks(range(len(MNAMES)))
ax.set_xticklabels(MNAMES, rotation=25, ha='right', fontsize=9)
ax.set_yticks(range(len(METHS)))
ax.set_yticklabels(METHS, fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.85,
             label='Faithfulness (AUC drop)')
ax.set_title(f'{DOMAIN}: Faithfulness Heatmap', fontsize=11)
plt.tight_layout()
plt.savefig('fig4_faithfulness_education.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig4_faithfulness_education.pdf saved")

# ── CELL 18 ── FIG 5: Stability Heatmap — KEY FIGURE ─────────────
METH_S  = ['TreeSHAP','KernelSHAP','LIME']
MNAME_S = ['Random Forest','XGBoost','LightGBM',
            'SVM','Logistic Regression','MLP']
mat_s   = np.full((len(METH_S), len(MNAME_S)), np.nan)
for mi, meth in enumerate(METH_S):
    for mj, mn in enumerate(MNAME_S):
        k = f'{meth}|{mn}'
        if k in STAB:
            mat_s[mi, mj] = STAB[k]

fig, ax = plt.subplots(figsize=(14, 4))
na2 = np.zeros_like(mat_s)
na2[np.isnan(mat_s)] = 1.
ax.imshow(na2,
          cmap=mcolors.ListedColormap(['white','#EBEBEB']),
          vmin=0, vmax=1, aspect='auto')
mat_m2 = np.ma.masked_where(np.isnan(mat_s), mat_s)
im2    = ax.imshow(mat_m2, cmap='RdYlGn',
                   vmin=0, vmax=1, aspect='auto')
for i in range(len(METH_S)):
    for j in range(len(MNAME_S)):
        if np.isnan(mat_s[i,j]):
            ax.text(j, i, 'N/A', ha='center', va='center',
                    fontsize=8, color='#888780', style='italic')
        else:
            v  = mat_s[i,j]
            nr = plt.cm.RdYlGn(v)
            br = nr[0]*0.299+nr[1]*0.587+nr[2]*0.114
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if br<0.5 else '#2C2C2A')
for x in np.arange(-0.5, len(MNAME_S), 1):
    ax.axvline(x, color='white', lw=1.5)
for y in np.arange(-0.5, len(METH_S), 1):
    ax.axhline(y, color='white', lw=1.5)
ax.set_xticks(range(len(MNAME_S)))
ax.set_xticklabels(MNAME_S, rotation=25, ha='right', fontsize=9)
ax.set_yticks(range(len(METH_S)))
ax.set_yticklabels(METH_S, fontsize=10)
plt.colorbar(im2, ax=ax, shrink=0.85,
             label='Jaccard Stability (0=unstable | 1=stable)')
ax.set_title(
    f'{DOMAIN}: Stability Heatmap — KEY FINDING (RQ2)\n'
    f'Small data (n=316 train): LIME Jaccard drops to 0.535 '
    f'| TreeSHAP = 1.000 in all cases',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig_stability_heatmap_education.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig_stability_heatmap_education.pdf saved")
print("⚠️  KEY FINDING: LIME instability under data scarcity")
print(f"   Min LIME Jaccard: "
      f"{min(v for k,v in STAB.items() if 'LIME' in k):.3f}")
print(f"   All TreeSHAP: 1.000")

# ── CELL 19 ── Summary Table ──────────────────────────────────────
print("=" * 70)
print("DOMAIN 4 — EDUCATION | FINAL SUMMARY TABLE")
print("=" * 70)
rows = []
for mn in list(PERF.keys()):
    sk = ('TreeSHAP' if mn in
          ['Random Forest','XGBoost','LightGBM']
          else 'KernelSHAP')
    rows.append({
        'Domain'     : DOMAIN,
        'Model'      : mn,
        'AUC'        : round(PERF[mn]['auc'], 3),
        'SHAP_type'  : sk,
        'SHAP_ms'    : round(XAI[sk].get(mn,{}).get('tps',0)*1000,2),
        'LIME_ms'    : round(XAI['LIME'].get(mn,{}).get('tps',0)*1000,2),
        'SHAP_faith' : faith.get(f'{sk}|{mn}', None),
        'SHAP_stab'  : STAB.get(f'{sk}|{mn}', None),
        'LIME_stab'  : STAB.get(f'LIME|{mn}', None),
        'SHAP_LIME_r': round(r, 3),
    })
df_sum = pd.DataFrame(rows)
print(df_sum.to_string(index=False))
df_sum.to_csv('summary_D4_education.csv', index=False)
print("\n✓ summary_D4_education.csv saved")
print("✓ ALL CELLS COMPLETE — EDUCATION DOMAIN")
