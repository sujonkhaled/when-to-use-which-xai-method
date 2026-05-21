# ╔══════════════════════════════════════════════════════════════════╗
# ║  XAI EMPIRICAL STUDY — NOTEBOOK 3: FINANCE DOMAIN              ║
# ║  Dataset  : German Credit (UCI Statlog)                         ║
# ║  Size     : 1,000 rows × 20 features | Binary classification   ║
# ║  Paper    : "When to Use Which Explainable AI Method?"          ║
# ║  Authors  : Khaled Mahmud Sujon et al.                          ║
# ║  Key      : GDPR-relevant — Anchors and DiCE evaluated          ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── CELL 1 ── Install Libraries ───────────────────────────────────
# !pip install -q shap lime dice-ml alibi xgboost lightgbm

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
DOMAIN = "Finance"
print("✓ Libraries loaded")

# ── CELL 3 ── Load Dataset ────────────────────────────────────────
URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases'
       '/statlog/german/german.data')
COL_NAMES = [
    'checking_acct','duration','credit_history','purpose',
    'credit_amount','savings_acct','employment',
    'installment_rate','personal_status','other_debtors',
    'residence','property','age','other_plans','housing',
    'existing_credits','job','dependents','telephone',
    'foreign_worker','target'
]
df = pd.read_csv(URL, sep=' ', names=COL_NAMES)
df['target'] = (df['target'] - 1).astype(int)
print(f"Shape: {df.shape}")
print(f"Class balance: {df['target'].value_counts().to_dict()}")

# Encode categoricals
for c in df.select_dtypes('object').columns:
    df[c] = LabelEncoder().fit_transform(df[c])
X = df[COL_NAMES[:-1]].copy()
y = df['target'].copy()
FEATURES = X.columns.tolist()

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# ── CELL 4 ── Train Models ────────────────────────────────────────
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
    pp  = mdl.predict_proba(X_te)[:,1]
    acc = accuracy_score(y_te, yp)
    f1  = f1_score(y_te, yp)
    auc = roc_auc_score(y_te, pp)
    PERF[name] = {'model':mdl,'acc':acc,'f1':f1,'auc':auc}
    print(f"{name:<22} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f}")

# ── CELL 5 ── XAI Setup ───────────────────────────────────────────
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

# ── CELL 6 ── TreeSHAP ────────────────────────────────────────────
print("▶ TreeSHAP ...")
for mn in ['Random Forest','XGBoost','LightGBM']:
    is_lgb = (mn == 'LightGBM')
    t0  = time.perf_counter()
    exp = shap.TreeExplainer(PERF[mn]['model'])
    sv  = fix_shap_shape(exp.shap_values(
        X_ex, approximate=(not is_lgb),
        check_additivity=False))
    t1  = time.perf_counter()
    XAI['TreeSHAP'][mn] = dict(
        values=sv, mean_abs=np.abs(sv).mean(0),
        tps=(t1-t0)/N_EXPL, exp=exp)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.2f} ms/sample")

# ── CELL 7 ── KernelSHAP ─────────────────────────────────────────
print("▶ KernelSHAP ...")
bg = shap.sample(X_tr, 100)
for mn in ['SVM','Logistic Regression','MLP']:
    t0  = time.perf_counter()
    exp = shap.KernelExplainer(PERF[mn]['model'].predict_proba, bg)
    sv  = fix_shap_shape(exp.shap_values(X_ex, nsamples=100))
    t1  = time.perf_counter()
    XAI['KernelSHAP'][mn] = dict(
        values=sv, mean_abs=np.abs(sv).mean(0),
        tps=(t1-t0)/N_EXPL, exp=exp)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.1f} ms/sample")

# ── CELL 8 ── LIME ────────────────────────────────────────────────
print("▶ LIME ...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_tr.values, feature_names=FEATURES,
    class_names=['Good','Bad'],
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

# ── CELL 9 ── Permutation Importance ─────────────────────────────
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

# ── CELL 10 ── Anchors (GDPR rules) ──────────────────────────────
print("▶ Anchors (GDPR rule-based) ...")
try:
    from alibi.explainers import AnchorTabular
    for mn in ['Random Forest','XGBoost','Logistic Regression']:
        ae = AnchorTabular(
            PERF[mn]['model'].predict,
            feature_names=FEATURES)
        ae.fit(X_tr.values, disc_perc=[25,50,75])
        times, prec, cov, rlen = [], [], [], []
        n_samples = 5 if mn == 'Random Forest' else 20
        for i in range(n_samples):
            t0 = time.perf_counter()
            ex = ae.explain(X_ex.values[i])
            t1 = time.perf_counter()
            times.append(t1-t0)
            prec.append(ex.data['precision'])
            cov.append(ex.data['coverage'])
            rlen.append(len(ex.data['anchor']))
        XAI['Anchors'][mn] = dict(
            tps=np.mean(times),
            precision=np.mean(prec),
            coverage=np.mean(cov),
            rule_len=np.mean(rlen))
        print(f"  [{mn}] {np.mean(times)*1000:.0f} ms | "
              f"prec={np.mean(prec):.3f} | "
              f"rule_len={np.mean(rlen):.1f}")
except Exception as e:
    print(f"  Anchors skipped: {e}")

# ── CELL 11 ── DiCE Counterfactuals ──────────────────────────────
print("▶ DiCE counterfactuals ...")
try:
    import dice_ml
    df_dice = pd.concat([
        X_tr.reset_index(drop=True),
        y_tr.reset_index(drop=True).rename('target')
    ], axis=1)
    d_obj = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=FEATURES,
        outcome_name='target')
    for mn in ['Random Forest','Logistic Regression']:
        m_obj  = dice_ml.Model(
            model=PERF[mn]['model'], backend='sklearn')
        dice_e = dice_ml.Dice(d_obj, m_obj, method='random')
        times, valids = [], []
        n_q = 5 if mn == 'Random Forest' else 20
        for i in range(n_q):
            t0 = time.perf_counter()
            e  = dice_e.generate_counterfactuals(
                X_ex.iloc[[i]], total_CFs=3,
                desired_class='opposite', verbose=False)
            t1 = time.perf_counter()
            times.append(t1-t0)
            cfs = e.cf_examples_list[0].final_cfs_df
            valids.append(len(cfs) if cfs is not None else 0)
        XAI['DiCE'][mn] = dict(
            tps=np.mean(times),
            valid_cfs=np.mean(valids))
        print(f"  [{mn}] {np.mean(times)*1000:.0f} ms | "
              f"CFs={np.mean(valids):.1f}")
except Exception as e:
    print(f"  DiCE skipped: {e}")

# ── CELL 12 ── Faithfulness ───────────────────────────────────────
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

# ── CELL 13 ── Stability ──────────────────────────────────────────
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
    bg_k = shap.sample(X_tr, 100)
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
for mn in ['SVM','Logistic Regression','MLP']:
    STAB[f'KernelSHAP|{mn}'] = stab_kernel(
        PERF[mn]['model'], X_stab)
for mn in list(PERF.keys()):
    STAB[f'LIME|{mn}'] = stab_lime(
        lime_exp, PERF[mn]['model'], X_ex.values[0])
print("Stability:")
for k, v in STAB.items():
    print(f"  {k}: {v:.4f}")

# ── CELL 14 ── Agreement ──────────────────────────────────────────
shap_m = fix_shap_shape(
    XAI['TreeSHAP']['Random Forest']['values']).mean(0)
lime_m = np.array(
    XAI['LIME']['Random Forest']['values']).mean(0)
r, p   = spearmanr(shap_m, lime_m)
print(f"SHAP vs LIME Spearman r={r:.3f}  p={p:.4f}")

# ── CELL 15 ── FIG 1: SHAP Bars ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#2471A3','#1E8449','#7D3C98']
for ax, (mn, col) in zip(
        axes,
        zip(['Random Forest','XGBoost','LightGBM'], colors)):
    sv  = fix_shap_shape(XAI['TreeSHAP'][mn]['values'])
    imp = np.abs(sv).mean(0)
    idx = np.argsort(imp)
    ax.barh([FEATURES[i] for i in idx], imp[idx],
            color=col, edgecolor='none')
    ax.set_title(f'TreeSHAP — {mn}', fontsize=10)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.grid(alpha=0.2, axis='x')
plt.suptitle(f'{DOMAIN}: Feature Importance via TreeSHAP',
             fontsize=12)
plt.tight_layout()
plt.savefig('fig1_shap_finance.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig1_shap_finance.pdf saved")

# ── CELL 16 ── FIG 2: Anchors GDPR Analysis ──────────────────────
if XAI.get('Anchors'):
    mns   = list(XAI['Anchors'].keys())
    precs = [XAI['Anchors'][m]['precision'] for m in mns]
    covs  = [XAI['Anchors'][m]['coverage']  for m in mns]
    rlens = [XAI['Anchors'][m]['rule_len']   for m in mns]
    tps   = [XAI['Anchors'][m]['tps']*1000   for m in mns]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, vals, ylabel, title in zip(
            axes,
            [precs, covs, rlens, tps],
            ['Precision','Coverage','Rule Length','Time (ms)'],
            ['Precision','Coverage','Rule Length','Compute Time']):
        ax.bar(mns, vals, color='#1D9E75', edgecolor='none')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(mns, rotation=20, ha='right', fontsize=8)
        ax.grid(alpha=0.2, axis='y')
        for bar, v in zip(ax.patches, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.01,
                    f'{v:.2f}', ha='center', fontsize=8)
    plt.suptitle(
        f'{DOMAIN}: Anchors Rule Quality (GDPR Compliance)',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('fig2_anchors_finance.pdf',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ fig2_anchors_finance.pdf saved")

# ── CELL 17 ── FIG 3: DiCE Counterfactual Analysis ───────────────
if XAI.get('DiCE'):
    mns_d  = list(XAI['DiCE'].keys())
    tps_d  = [XAI['DiCE'][m]['tps']*1000 for m in mns_d]
    cfs_d  = [XAI['DiCE'][m]['valid_cfs'] for m in mns_d]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(mns_d, tps_d, color=['#2471A3','#1D9E75'],
            edgecolor='none')
    for bar, v in zip(ax1.patches, tps_d):
        ax1.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+500,
                 f'{v:,.0f} ms', ha='center', fontsize=9)
    ax1.set_ylabel('Time (ms/sample)', fontsize=9)
    ax1.set_title('DiCE Computation Time', fontsize=10)
    ax1.grid(alpha=0.2, axis='y')

    ax2.bar(mns_d, cfs_d, color=['#2471A3','#1D9E75'],
            edgecolor='none')
    for bar, v in zip(ax2.patches, cfs_d):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.05,
                 f'{v:.1f}', ha='center', fontsize=9)
    ax2.set_ylabel('Valid Counterfactuals', fontsize=9)
    ax2.set_title('DiCE Valid CFs per Query', fontsize=10)
    ax2.grid(alpha=0.2, axis='y')
    plt.suptitle(
        f'{DOMAIN}: DiCE Counterfactual Results',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('fig3_dice_finance.pdf',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ fig3_dice_finance.pdf saved")

# ── CELL 18 ── FIG 4: Faithfulness Heatmap ───────────────────────
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
plt.savefig('fig4_faithfulness_finance.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig4_faithfulness_finance.pdf saved")

# ── CELL 19 ── FIG 5: Stability Heatmap ──────────────────────────
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
             label='Jaccard Stability')
ax.set_title(
    f'{DOMAIN}: Stability Heatmap (15 runs, top-5)\n'
    f'green=stable | red=unstable | grey=N/A',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig_stability_heatmap_finance.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig_stability_heatmap_finance.pdf saved")

# ── CELL 20 ── Summary Table ──────────────────────────────────────
print("=" * 70)
print("DOMAIN 3 — FINANCE | FINAL SUMMARY TABLE")
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
        'Anchors_ms' : round(XAI['Anchors'].get(mn,{}).get('tps',0)*1000,0)
                       if XAI.get('Anchors') and mn in XAI['Anchors']
                       else None,
        'DiCE_ms'    : round(XAI['DiCE'].get(mn,{}).get('tps',0)*1000,0)
                       if XAI.get('DiCE') and mn in XAI['DiCE']
                       else None,
    })
df_sum = pd.DataFrame(rows)
print(df_sum.to_string(index=False))
df_sum.to_csv('summary_D3_finance.csv', index=False)
print("\n✓ summary_D3_finance.csv saved")
print("✓ ALL CELLS COMPLETE — FINANCE DOMAIN")
