# ╔══════════════════════════════════════════════════════════════════╗
# ║  XAI EMPIRICAL STUDY — NOTEBOOK 2: CYBERSECURITY DOMAIN        ║
# ║  Dataset  : UNSW-NB15 Network Intrusion Detection               ║
# ║  Size     : 257,673 rows × 49 features | Multi-class           ║
# ║  Paper    : "When to Use Which Explainable AI Method?"          ║
# ║  Authors  : Khaled Mahmud Sujon et al.                          ║
# ║                                                                  ║
# ║  DATA SETUP (choose one option before running):                  ║
# ║  Option A — Kaggle API:                                          ║
# ║    kaggle datasets download -d mrwellsdavid/unsw-nb15           ║
# ║  Option B — Direct download from UNSW:                          ║
# ║    https://research.unsw.edu.au/projects/unsw-nb15-dataset      ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── CELL 1 ── Install Libraries ───────────────────────────────────
# !pip install -q shap lime alibi xgboost lightgbm

# ── CELL 2 ── Imports ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 150,
                            'axes.spines.top': False,
                            'axes.spines.right': False})
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
import lime, lime.lime_tabular
import time, warnings, glob
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
DOMAIN = "Cybersecurity"
print("✓ Libraries loaded")

# ── CELL 3 ── Load Dataset ────────────────────────────────────────
# Adjust filename to match your downloaded file
csv_files = glob.glob('UNSW*.csv') or glob.glob('*.csv')
print("Found CSV files:", csv_files)
df = pd.read_csv(csv_files[0])
print(f"Shape: {df.shape}")

# ── CELL 4 ── Preprocessing ───────────────────────────────────────
# Identify label column
label_col = [c for c in df.columns
             if c.lower() in ['label','attack_cat']][0]
print(f"Label column: {label_col}")
print(df[label_col].value_counts())

# Drop non-feature columns
drop_cols = [c for c in df.columns
             if c.lower() in ['id','srcip','dstip',
                               'sport','dsport','stime','ltime',
                               'attack_cat','label']
             or c == label_col]
X = df.drop(columns=drop_cols, errors='ignore')

# Encode label
le = LabelEncoder()
y  = pd.Series(le.fit_transform(df[label_col].astype(str)))

# Handle non-numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(0)

FEATURES = X.columns.tolist()
print(f"Features: {len(FEATURES)} | Classes: {len(np.unique(y))}")

# ── CELL 5 ── Subsample + Split ───────────────────────────────────
SAMPLE_N = 50000
if len(X) > SAMPLE_N:
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=SAMPLE_N,
        random_state=42, stratify=y)
    print(f"Stratified sample: {SAMPLE_N} rows")
else:
    X_s, y_s = X, y

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X_s),
                       columns=FEATURES)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y_s, test_size=0.2,
    random_state=42, stratify=y_s)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# Binary label for AUC
is_binary  = len(np.unique(y)) == 2
auc_kwargs = {'multi_class':'ovr','average':'macro'} \
             if not is_binary else {}

# ── CELL 6 ── Train Models ────────────────────────────────────────
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
                            hidden_layer_sizes=(128,64,32),
                            max_iter=500, random_state=42),
}
PERF = {}
print(f"{'Model':<22} {'Acc':>6} {'F1':>6} {'AUC':>6}")
print("─" * 46)
for name, mdl in MODELS.items():
    mdl.fit(X_tr, y_tr)
    yp  = mdl.predict(X_te)
    pp  = mdl.predict_proba(X_te)
    acc = accuracy_score(y_te, yp)
    f1  = f1_score(y_te, yp, average='macro')
    auc = roc_auc_score(y_te, pp, **auc_kwargs)
    PERF[name] = {'model':mdl,'acc':acc,'f1':f1,'auc':auc}
    print(f"{name:<22} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f}")

# ── CELL 7 ── XAI Setup ───────────────────────────────────────────
XAI    = defaultdict(dict)
N_EXPL = 100
X_ex   = X_te.iloc[:N_EXPL]
y_ex   = y_te.iloc[:N_EXPL]

def fix_shap_shape(sv):
    arr = np.array(sv)
    if isinstance(sv, list):
        arr = np.abs(np.array(sv)).sum(axis=0)
    elif arr.ndim == 3:
        arr = np.abs(arr).sum(axis=0)
    return arr

# ── CELL 8 ── TreeSHAP ────────────────────────────────────────────
print("▶ TreeSHAP ...")
APPROX = {'Random Forest':True, 'XGBoost':True, 'LightGBM':False}
for mn in ['Random Forest','XGBoost','LightGBM']:
    mdl = PERF[mn]['model']
    t0  = time.perf_counter()
    exp = shap.TreeExplainer(mdl)
    sv  = fix_shap_shape(exp.shap_values(
        X_ex, approximate=APPROX[mn],
        check_additivity=False))
    t1  = time.perf_counter()
    XAI['TreeSHAP'][mn] = dict(
        values=sv, mean_abs=np.abs(sv).mean(0),
        tps=(t1-t0)/N_EXPL, exp=exp)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.2f} ms/sample")

# ── CELL 9 ── KernelSHAP (slow) ──────────────────────────────────
print("▶ KernelSHAP (slow ~10-20 min) ...")
bg = shap.sample(X_tr, 100)
for mn in ['SVM','Logistic Regression','MLP']:
    mdl = PERF[mn]['model']
    t0  = time.perf_counter()
    exp = shap.KernelExplainer(mdl.predict_proba, bg)
    sv  = fix_shap_shape(
        exp.shap_values(X_ex.iloc[:50], nsamples=128))
    t1  = time.perf_counter()
    XAI['KernelSHAP'][mn] = dict(
        values=sv, mean_abs=np.abs(sv).mean(0),
        tps=(t1-t0)/50)
    print(f"  [{mn}] {(t1-t0)/50*1000:.1f} ms/sample")

# ── CELL 10 ── LIME (RF only — large dataset) ─────────────────────
print("▶ LIME (RF only) ...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_tr.values, feature_names=FEATURES,
    class_names=[str(c) for c in sorted(y.unique())],
    discretize_continuous=True, random_state=42)

def lime_imp(model, Xsample, n):
    t0, rows = time.perf_counter(), []
    for i in range(n):
        e   = lime_exp.explain_instance(
            Xsample.values[i], model.predict_proba,
            num_features=len(FEATURES))
        w   = dict(e.as_list())
        row = [max([abs(v) for k,v in w.items()
                    if fn in k], default=0.)
               for fn in FEATURES]
        rows.append(row)
    t1 = time.perf_counter()
    return np.array(rows), (t1-t0)/n

for mn in ['Random Forest','Logistic Regression']:
    arr, tps = lime_imp(PERF[mn]['model'], X_ex.iloc[:50], 50)
    XAI['LIME'][mn] = dict(
        values=arr, mean_abs=arr.mean(0), tps=tps)
    print(f"  [{mn}] {tps*1000:.1f} ms/sample")

# ── CELL 11 ── Permutation Importance ────────────────────────────
print("▶ Permutation Importance ...")
for mn, res in PERF.items():
    t0   = time.perf_counter()
    perm = permutation_importance(
        res['model'], X_te, y_te,
        n_repeats=10, random_state=42, n_jobs=-1)
    t1   = time.perf_counter()
    XAI['PermImp'][mn] = dict(
        values=perm.importances,
        mean_abs=perm.importances_mean,
        tps=(t1-t0)/len(X_te))
    print(f"  [{mn}] {(t1-t0)/len(X_te)*1000:.2f} ms/sample")

# ── CELL 12 ── Anchors (5 samples — intractability test) ──────────
print("▶ Anchors (5 samples — testing intractability) ...")
try:
    from alibi.explainers import AnchorTabular
    for mn in ['Random Forest','XGBoost']:
        ae = AnchorTabular(
            PERF[mn]['model'].predict,
            feature_names=FEATURES)
        ae.fit(X_tr.values, disc_perc=[25,50,75])
        times, prec, cov, rlen = [], [], [], []
        for i in range(5):
            print(f"    Sample {i+1}/5 ...", end=' ')
            t0 = time.perf_counter()
            ex = ae.explain(X_ex.values[i])
            t1 = time.perf_counter()
            times.append(t1-t0)
            prec.append(ex.data['precision'])
            cov.append(ex.data['coverage'])
            rlen.append(len(ex.data['anchor']))
            print(f"{(t1-t0)*1000:.0f} ms")
        XAI['Anchors'][mn] = dict(
            tps=np.mean(times),
            precision=np.mean(prec),
            coverage=np.mean(cov),
            rule_len=np.mean(rlen))
        print(f"  [{mn}] avg={np.mean(times)*1000:.0f} ms/sample")
        print(f"  ⚠️  INTRACTABLE at scale: "
              f"{np.mean(times)*1000:.0f} ms/sample")
except Exception as e:
    print(f"  Anchors skipped: {e}")

# ── CELL 13 ── Faithfulness ───────────────────────────────────────
def faithfulness(model, X_df, shap_vals, y_true,
                 top_k=5, is_bin=True):
    sv   = fix_shap_shape(shap_vals)
    idx  = np.argsort(sv.mean(0))[-top_k:].flatten().tolist()
    pp_fn = (lambda Xd: model.predict_proba(Xd)[:,1]
             if is_bin
             else model.predict_proba(Xd))
    base  = roc_auc_score(y_true, pp_fn(X_df),
                          **({} if is_bin else
                             {'multi_class':'ovr',
                              'average':'macro'}))
    Xm = X_df.copy()
    for i in idx:
        Xm.iloc[:, int(i)] = Xm.iloc[:, int(i)].mean()
    mask = roc_auc_score(y_true, pp_fn(Xm),
                         **({} if is_bin else
                            {'multi_class':'ovr',
                             'average':'macro'}))
    return round(base - mask, 4)

faith = {}
for mn in ['Random Forest','XGBoost','LightGBM']:
    sv = fix_shap_shape(XAI['TreeSHAP'][mn]['values'])
    faith[f'TreeSHAP|{mn}'] = faithfulness(
        PERF[mn]['model'], X_ex, sv, y_ex, is_bin=is_binary)
for mn in ['SVM','Logistic Regression','MLP']:
    sv = fix_shap_shape(XAI['KernelSHAP'][mn]['values'])
    faith[f'KernelSHAP|{mn}'] = faithfulness(
        PERF[mn]['model'], X_ex.iloc[:50],
        sv, y_ex.iloc[:50], is_bin=is_binary)
print("Faithfulness:")
for k,v in faith.items():
    print(f"  {k}: {v:.4f}")

# ── CELL 14 ── Stability (RF only) ───────────────────────────────
def stab_shap(exp_fn, X_s, n=15, k=5, approx=True):
    sets = []
    for _ in range(n):
        sv  = fix_shap_shape(np.array(
            exp_fn(X_s, approximate=approx,
                   check_additivity=False)))
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

X_stab    = X_ex.iloc[:10]
stab_shap_rf = stab_shap(
    XAI['TreeSHAP']['Random Forest']['exp'].shap_values,
    X_stab, approx=True)
stab_lime_rf = stab_lime(
    lime_exp, PERF['Random Forest']['model'],
    X_ex.values[0])
print(f"Stability TreeSHAP RF: {stab_shap_rf}")
print(f"Stability LIME RF:     {stab_lime_rf}")

# ── CELL 15 ── Agreement ──────────────────────────────────────────
shap_m = fix_shap_shape(
    XAI['TreeSHAP']['Random Forest']['values']).mean(0)
lime_m = np.array(
    XAI['LIME']['Random Forest']['values']).mean(0).flatten()
r, p   = spearmanr(shap_m, lime_m)
print(f"SHAP vs LIME Spearman r={r:.3f}  p={p:.4f}")

# ── CELL 16 ── FIG 1: SHAP Top-20 Feature Importance ─────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
colors = ['#378ADD','#1D9E75','#7F77DD']
for ax, (mn, col) in zip(
        axes,
        zip(['Random Forest','XGBoost','LightGBM'], colors)):
    imp = fix_shap_shape(
        XAI['TreeSHAP'][mn]['values']).mean(0)
    top = np.argsort(imp)[::-1][:20]
    ax.barh([FEATURES[i] for i in top[::-1]],
            imp[top[::-1]], color=col, edgecolor='none')
    ax.set_title(f'TreeSHAP — {mn}', fontsize=10)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.grid(alpha=0.2, axis='x')
plt.suptitle(f'{DOMAIN}: Top-20 Feature Importance (TreeSHAP)',
             fontsize=12)
plt.tight_layout()
plt.savefig('fig1_shap_security.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig1_shap_security.pdf saved")

# ── CELL 17 ── FIG 2: Computation Time ───────────────────────────
TIME_D = {}
for mn in ['Random Forest','XGBoost','LightGBM']:
    TIME_D[f'TreeSHAP ({mn[:3]})'] = \
        XAI['TreeSHAP'][mn]['tps']
for mn in ['SVM','Logistic Regression','MLP']:
    if mn in XAI['KernelSHAP']:
        TIME_D[f'KernelSHAP ({mn[:3]})'] = \
            XAI['KernelSHAP'][mn]['tps']
for mn in ['Random Forest','Logistic Regression']:
    if mn in XAI['LIME']:
        TIME_D[f'LIME ({mn[:3]})'] = \
            XAI['LIME'][mn]['tps']
for mn in ['Random Forest','XGBoost']:
    TIME_D[f'PermImp ({mn[:3]})'] = \
        XAI['PermImp'][mn]['tps']
if XAI.get('Anchors'):
    for mn in XAI['Anchors']:
        TIME_D[f'Anchors ({mn[:3]}) ⚠️'] = \
            XAI['Anchors'][mn]['tps']

labels = list(TIME_D.keys())
vals   = [v*1000 for v in TIME_D.values()]
cols   = ['#378ADD' if 'Tree' in l
          else '#85B7EB' if 'Kernel' in l
          else '#BA7517' if 'LIME' in l
          else '#888780' if 'Perm' in l
          else '#D4537E' for l in labels]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(labels, vals, color=cols, edgecolor='none')
ax.set_yscale('log')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()*1.1,
            f'{v:.1f}', ha='center',
            fontsize=7.5, rotation=40)
ax.set_ylabel('Time (ms/sample, log scale)', fontsize=10)
ax.set_title(
    f'{DOMAIN}: XAI Computation Time\n'
    f'⚠️ Anchors intractable at scale',
    fontsize=11)
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('fig2_time_security.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig2_time_security.pdf saved")

# ── CELL 18 ── FIG 3: SHAP vs LIME Agreement ─────────────────────
n_top  = 15
shap_s = pd.Series(shap_m, index=FEATURES).nlargest(n_top)
lime_s = pd.Series(lime_m, index=FEATURES).nlargest(n_top)
all_f  = list(dict.fromkeys(
    list(shap_s.index) + list(lime_s.index)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
for ax, vals, col, title in [
    (ax1, pd.Series(shap_m, index=FEATURES)[all_f],
     '#378ADD', 'TreeSHAP (RF)'),
    (ax2, pd.Series(lime_m, index=FEATURES)[all_f],
     '#BA7517', 'LIME (RF)')
]:
    srt = vals.sort_values(ascending=True)
    ax.barh(srt.index, srt.values, color=col, edgecolor='none')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Mean Importance', fontsize=9)
    ax.grid(alpha=0.2, axis='x')
plt.suptitle(
    f'{DOMAIN}: SHAP vs LIME Top-{n_top} Agreement\n'
    f'Spearman r={r:.3f}  p={p:.4f}',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig3_agreement_security.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig3_agreement_security.pdf saved")

# ── CELL 19 ── FIG 4: Faithfulness Heatmap ───────────────────────
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
na_mask = np.zeros_like(mat_f)
na_mask[np.isnan(mat_f)] = 1.
ax.imshow(na_mask,
          cmap=mcolors.ListedColormap(['white','#EBEBEB']),
          vmin=0, vmax=1, aspect='auto')
mat_m = np.ma.masked_where(np.isnan(mat_f), mat_f)
vmax  = np.nanmax(np.abs(mat_f))
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
plt.savefig('fig4_faithfulness_security.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig4_faithfulness_security.pdf saved")

# ── CELL 20 ── Summary Table ──────────────────────────────────────
print("=" * 70)
print("DOMAIN 2 — CYBERSECURITY | FINAL SUMMARY TABLE")
print("=" * 70)
rows = []
for mn, res in PERF.items():
    sk = ('TreeSHAP' if mn in
          ['Random Forest','XGBoost','LightGBM']
          else 'KernelSHAP')
    rows.append({
        'Domain'     : DOMAIN,
        'Model'      : mn,
        'AUC'        : round(res['auc'], 3),
        'SHAP_type'  : sk,
        'SHAP_ms'    : round(XAI[sk].get(mn,{}).get('tps',0)*1000,2),
        'LIME_ms'    : round(XAI['LIME'].get(mn,{}).get('tps',0)*1000,2),
        'PermImp_ms' : round(XAI['PermImp'].get(mn,{}).get('tps',0)*1000,2),
        'SHAP_faith' : faith.get(f'{sk}|{mn}', '—'),
        'SHAP_stab'  : stab_shap_rf if mn=='Random Forest' else '—',
        'LIME_stab'  : stab_lime_rf if mn=='Random Forest' else '—',
        'SHAP_LIME_r': round(r, 3),
    })
df_sum = pd.DataFrame(rows)
print(df_sum.to_string(index=False))
df_sum.to_csv('summary_D2_security.csv', index=False)
print("\n✓ summary_D2_security.csv saved")
print("✓ ALL CELLS COMPLETE — CYBERSECURITY DOMAIN")
