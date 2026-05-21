# ╔══════════════════════════════════════════════════════════════════╗
# ║  XAI EMPIRICAL STUDY — NOTEBOOK 1: HEALTHCARE DOMAIN           ║
# ║  Dataset  : PIMA Indians Diabetes (UCI)                         ║
# ║  Size     : 768 rows × 8 features | Binary classification       ║
# ║  Paper    : "When to Use Which Explainable AI Method?"          ║
# ║  Authors  : Khaled Mahmud Sujon et al.                          ║
# ║  Run on   : Google Colab (GPU not required for this notebook)   ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── CELL 1 ── Install Libraries ───────────────────────────────────
# Run this cell first, then Runtime > Restart, then run all below
# !pip install -q shap lime dice-ml alibi xgboost lightgbm

# ── CELL 2 ── Imports ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 150,
                            'axes.spines.top': False,
                            'axes.spines.right': False})
import matplotlib.pyplot as plt
import shap
import lime, lime.lime_tabular
import time, warnings
from collections import defaultdict
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score)
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)
DOMAIN = "Healthcare"
print("✓ Libraries loaded")

# ── CELL 3 ── Load & Preprocess Dataset ───────────────────────────
URL  = ('https://raw.githubusercontent.com/jbrownlee/'
        'Datasets/master/pima-indians-diabetes.data.csv')
COLS = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

df = pd.read_csv(URL, names=COLS)
print(f"Shape: {df.shape} | "
      f"Class balance: {df['Outcome'].value_counts().to_dict()}")

X = df.drop('Outcome', axis=1).copy()
y = df['Outcome'].copy()
FEATURES = X.columns.tolist()

# Replace biologically impossible zeros with median
for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    X[c] = X[c].replace(0, np.nan)
X = X.fillna(X.median())

scaler = StandardScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# ── CELL 4 ── Train 6 Scikit-learn Models ─────────────────────────
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

# ── CELL 5 ── Train TF-MLP ────────────────────────────────────────
X_tr_np = np.array(X_tr, dtype=np.float32)
X_te_np = np.array(X_te, dtype=np.float32)
y_tr_np = np.array(y_tr, dtype=np.float32)
y_te_np = np.array(y_te, dtype=np.float32)

tf_mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_tr_np.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
tf_mlp.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
tf_mlp.fit(X_tr_np, y_tr_np, epochs=80, batch_size=32,
           validation_split=0.1, verbose=0)
tf_auc = roc_auc_score(
    y_te_np, tf_mlp.predict(X_te_np, verbose=0).ravel())
print(f"{'TF-MLP':<22} {'—':>6} {'—':>6} {tf_auc:>6.3f}")

# ── CELL 6 ── XAI Setup ───────────────────────────────────────────
XAI    = defaultdict(dict)
N_EXPL = 100
X_ex   = X_te.iloc[:N_EXPL]
y_ex   = y_te.iloc[:N_EXPL]

def fix_shap_shape(sv):
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv.sum(axis=0)
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv

# ── CELL 7 ── TreeSHAP (RF, XGB, LGB) ────────────────────────────
print("▶ TreeSHAP ...")
for mn in ['Random Forest','XGBoost','LightGBM']:
    mdl = PERF[mn]['model']
    t0  = time.perf_counter()
    exp = shap.TreeExplainer(mdl)
    sv  = fix_shap_shape(exp.shap_values(X_ex))
    t1  = time.perf_counter()
    XAI['TreeSHAP'][mn] = dict(
        values=sv, mean_abs=np.abs(sv).mean(0),
        tps=(t1-t0)/N_EXPL, exp=exp)
    print(f"  [{mn}] {(t1-t0)/N_EXPL*1000:.1f} ms/sample")

# ── CELL 8 ── KernelSHAP (SVM, LR, MLP) ──────────────────────────
print("▶ KernelSHAP (slow ~2-5 min) ...")
bg = shap.sample(X_tr, 100)
for mn in ['SVM','Logistic Regression','MLP']:
    mdl = PERF[mn]['model']
    t0  = time.perf_counter()
    exp = shap.KernelExplainer(mdl.predict_proba, bg)
    sv  = fix_shap_shape(exp.shap_values(X_ex.iloc[:50],
                                          nsamples=128))
    t1  = time.perf_counter()
    XAI['KernelSHAP'][mn] = dict(
        values=sv, mean_abs=np.abs(sv).mean(0),
        tps=(t1-t0)/50, exp=exp)
    print(f"  [{mn}] {(t1-t0)/50*1000:.1f} ms/sample")

# ── CELL 9 ── DeepSHAP (TF-MLP) ──────────────────────────────────
print("▶ DeepSHAP (TF-MLP) ...")
bg_tf = X_tr.values[:200].astype(np.float32)
dexp  = shap.DeepExplainer(tf_mlp, bg_tf)
t0    = time.perf_counter()
sv_tf = np.array(dexp.shap_values(
    X_ex.values.astype(np.float32)))
sv_tf = fix_shap_shape(sv_tf)
t1    = time.perf_counter()
XAI['DeepSHAP']['TF-MLP'] = dict(
    values=sv_tf, mean_abs=np.abs(sv_tf).mean(0),
    tps=(t1-t0)/N_EXPL)
print(f"  [TF-MLP] {(t1-t0)/N_EXPL*1000:.1f} ms/sample")

# ── CELL 10 ── Integrated Gradients (TF-MLP) ─────────────────────
print("▶ Integrated Gradients (TF-MLP) ...")

@tf.function
def _grad(inp, model):
    with tf.GradientTape() as tape:
        tape.watch(inp)
        pred = model(inp)
    return tape.gradient(pred, inp)

def intgrad(model, x, steps=50):
    x  = tf.cast(x, tf.float32)
    bl = tf.zeros_like(x)
    al = tf.linspace(0., 1., steps+1)
    pts = tf.stack([bl + a*(x-bl) for a in al])
    gs  = _grad(pts, model)
    return ((x-bl)*tf.reduce_mean(
        (gs[:-1]+gs[1:])/2., 0)).numpy()

t0   = time.perf_counter()
ig_v = np.array([intgrad(tf_mlp, X_ex.values[i])
                  for i in range(N_EXPL)])
t1   = time.perf_counter()
XAI['IntGrad']['TF-MLP'] = dict(
    values=ig_v, mean_abs=np.abs(ig_v).mean(0),
    tps=(t1-t0)/N_EXPL)
print(f"  [TF-MLP] {(t1-t0)/N_EXPL*1000:.1f} ms/sample")

# ── CELL 11 ── LIME (all models) ──────────────────────────────────
print("▶ LIME (~5 min) ...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_tr.values, feature_names=FEATURES,
    class_names=['Neg','Pos'],
    discretize_continuous=True, random_state=42)

def lime_importances(model, X_sample, n=N_EXPL):
    t0, rows = time.perf_counter(), []
    for i in range(n):
        e   = lime_exp.explain_instance(
            X_sample.values[i], model.predict_proba,
            num_features=len(FEATURES))
        w   = dict(e.as_list())
        row = [max([abs(v) for k,v in w.items()
                    if fn in k], default=0.)
               for fn in FEATURES]
        rows.append(row)
    t1  = time.perf_counter()
    arr = np.array(rows)
    return arr, (t1-t0)/n

for mn, res in PERF.items():
    arr, tps = lime_importances(res['model'], X_ex)
    XAI['LIME'][mn] = dict(
        values=arr, mean_abs=arr.mean(0), tps=tps)
    print(f"  [{mn}] {tps*1000:.1f} ms/sample")

# ── CELL 12 ── Permutation Importance ────────────────────────────
print("▶ Permutation Importance ...")
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

# ── CELL 13 ── Anchors (RF, XGB) ──────────────────────────────────
print("▶ Anchors (~5-10 min) ...")
try:
    from alibi.explainers import AnchorTabular
    for mn in ['Random Forest','XGBoost']:
        ae = AnchorTabular(
            PERF[mn]['model'].predict,
            feature_names=FEATURES)
        ae.fit(X_tr.values, disc_perc=[25,50,75])
        times, prec, cov, rlen = [], [], [], []
        for i in range(20):
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

# ── CELL 14 ── DiCE Counterfactuals (RF, XGB) ─────────────────────
print("▶ DiCE (~2-4 min) ...")
try:
    import dice_ml
    df_dice = pd.concat([
        X_tr.reset_index(drop=True),
        y_tr.reset_index(drop=True).rename('Outcome')
    ], axis=1)
    d_obj = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=FEATURES,
        outcome_name='Outcome')
    for mn in ['Random Forest','XGBoost']:
        m_obj  = dice_ml.Model(
            model=PERF[mn]['model'], backend='sklearn')
        dice_e = dice_ml.Dice(d_obj, m_obj, method='random')
        times, valids = [], []
        for i in range(20):
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
              f"avg CFs={np.mean(valids):.1f}")
except Exception as e:
    print(f"  DiCE skipped: {e}")

# ── CELL 15 ── Faithfulness ───────────────────────────────────────
def faithfulness(model, X_df, shap_vals, y_true, top_k=5):
    sv   = fix_shap_shape(np.array(shap_vals))
    base = roc_auc_score(y_true,
                          model.predict_proba(X_df)[:,1])
    idx  = np.argsort(np.abs(sv).mean(0))[-top_k:].flatten()
    Xm   = X_df.copy()
    for i in idx:
        Xm.iloc[:, int(i)] = Xm.iloc[:, int(i)].mean()
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
        PERF[mn]['model'], X_ex.iloc[:50],
        sv, y_ex.iloc[:50])

print("Faithfulness (AUC drop, top-5 masked):")
for k, v in faith.items():
    print(f"  {k}: {v:.4f}")

# ── CELL 16 ── Stability (Jaccard, 15 runs) ───────────────────────
def stab_shap(exp_fn, X_s, n=15, k=5):
    sets = []
    for _ in range(n):
        sv  = fix_shap_shape(np.array(exp_fn(X_s)))
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

STAB = {}
X_stab = X_ex.iloc[:10]
for mn in ['Random Forest','XGBoost','LightGBM']:
    STAB[f'TreeSHAP|{mn}'] = stab_shap(
        XAI['TreeSHAP'][mn]['exp'].shap_values, X_stab)
bg_k = shap.sample(X_tr, 100)
for mn in ['SVM','Logistic Regression','MLP']:
    kexp = shap.KernelExplainer(
        PERF[mn]['model'].predict_proba, bg_k)
    STAB[f'KernelSHAP|{mn}'] = stab_shap(
        kexp.shap_values, X_stab)
for mn in list(PERF.keys()):
    STAB[f'LIME|{mn}'] = stab_lime(
        lime_exp, PERF[mn]['model'], X_ex.values[0])

print("Stability results:")
for k, v in STAB.items():
    print(f"  {k}: {v:.4f}")

# ── CELL 17 ── Inter-Method Agreement ────────────────────────────
shap_mean = np.abs(fix_shap_shape(
    XAI['TreeSHAP']['Random Forest']['values'])).mean(0)
lime_mean = np.array(
    XAI['LIME']['Random Forest']['values']).mean(0)
r, p = spearmanr(shap_mean, lime_mean)
print(f"SHAP vs LIME — Spearman r={r:.3f}  p={p:.4f}")

# ── CELL 18 ── FIG 1: SHAP Bar Charts ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#2471A3','#1E8449','#7D3C98']
for ax, (mn, col) in zip(
        axes,
        zip(['Random Forest','XGBoost','LightGBM'], colors)):
    sv   = fix_shap_shape(XAI['TreeSHAP'][mn]['values'])
    imp  = np.abs(sv).mean(0)
    idx  = np.argsort(imp)
    ax.barh([FEATURES[i] for i in idx], imp[idx],
            color=col, edgecolor='none')
    ax.set_title(f'TreeSHAP — {mn}', fontsize=10)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.grid(alpha=0.2, axis='x')
plt.suptitle(
    f'{DOMAIN}: Feature Importance via TreeSHAP',
    fontsize=12)
plt.tight_layout()
plt.savefig('fig1_shap_bar_healthcare.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig1_shap_bar_healthcare.pdf saved")

# ── CELL 19 ── FIG 2: SHAP vs LIME Agreement ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(shap_mean, lime_mean,
           color='#2471A3', s=80,
           edgecolors='white', lw=1.2)
for i, f in enumerate(FEATURES):
    ax.annotate(f, (shap_mean[i], lime_mean[i]),
                fontsize=7.5,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Mean |SHAP|', fontsize=10)
ax.set_ylabel('Mean |LIME|', fontsize=10)
ax.set_title(
    f'{DOMAIN}: SHAP vs LIME Agreement\n'
    f'Spearman r={r:.3f}  p={p:.4f}',
    fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('fig2_shap_vs_lime_healthcare.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig2_shap_vs_lime_healthcare.pdf saved")

# ── CELL 20 ── FIG 3: Faithfulness Heatmap ───────────────────────
import matplotlib.colors as mcolors

METHS  = ['TreeSHAP','KernelSHAP','LIME','IntGrad','PermImp']
MNAMES = ['Random Forest','XGBoost','LightGBM',
          'SVM','Logistic Regression','MLP']
mat_f  = np.full((len(METHS), len(MNAMES)), np.nan)
for mi, meth in enumerate(METHS):
    for mj, mn in enumerate(MNAMES):
        k = f'{meth}|{mn}'
        if k in faith:
            mat_f[mi, mj] = faith[k]

fig, ax = plt.subplots(figsize=(14, 4))
na_mask = np.zeros_like(mat_f)
na_mask[np.isnan(mat_f)] = 1.
ax.imshow(na_mask,
          cmap=mcolors.ListedColormap(['white','#EBEBEB']),
          vmin=0, vmax=1, aspect='auto')
vmax = np.nanmax(np.abs(mat_f))
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
            nr = plt.cm.RdYlGn((v+vmax)/(2*vmax))
            br = nr[0]*0.299+nr[1]*0.587+nr[2]*0.114
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if br<0.5 else '#2C2C2A')
for x in np.arange(-0.5, len(MNAMES), 1):
    ax.axvline(x, color='white', lw=1.5)
for y in np.arange(-0.5, len(METHS), 1):
    ax.axhline(y, color='white', lw=1.5)
ax.set_xticks(range(len(MNAMES)))
ax.set_xticklabels(MNAMES, rotation=25, ha='right', fontsize=9)
ax.set_yticks(range(len(METHS)))
ax.set_yticklabels(METHS, fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.85,
             label='Faithfulness (AUC drop)')
ax.set_title(
    f'{DOMAIN}: Faithfulness Heatmap\n'
    f'(green=faithful | red=misleading | grey=N/A)',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig3_faithfulness_heatmap_healthcare.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig3_faithfulness_heatmap_healthcare.pdf saved")

# ── CELL 21 ── FIG 4: Computation Time Bar Chart ─────────────────
TIME_D = {
    'TreeSHAP (RF)'   : XAI['TreeSHAP']['Random Forest']['tps'],
    'TreeSHAP (XGB)'  : XAI['TreeSHAP']['XGBoost']['tps'],
    'TreeSHAP (LGB)'  : XAI['TreeSHAP']['LightGBM']['tps'],
    'KernelSHAP (SVM)': XAI['KernelSHAP']['SVM']['tps'],
    'KernelSHAP (LR)' : XAI['KernelSHAP']['Logistic Regression']['tps'],
    'KernelSHAP (MLP)': XAI['KernelSHAP']['MLP']['tps'],
    'LIME (RF)'       : XAI['LIME']['Random Forest']['tps'],
    'LIME (LR)'       : XAI['LIME']['Logistic Regression']['tps'],
    'PermImp (RF)'    : XAI['PermImp']['Random Forest']['tps'],
    'DeepSHAP (TF-MLP)': XAI['DeepSHAP']['TF-MLP']['tps'],
    'IntGrad (TF-MLP)': XAI['IntGrad']['TF-MLP']['tps'],
}
if XAI.get('Anchors'):
    for mn in XAI['Anchors']:
        TIME_D[f'Anchors ({mn[:2]})'] = XAI['Anchors'][mn]['tps']
if XAI.get('DiCE'):
    for mn in XAI['DiCE']:
        TIME_D[f'DiCE ({mn[:2]})'] = XAI['DiCE'][mn]['tps']

labels = list(TIME_D.keys())
vals   = [v*1000 for v in TIME_D.values()]
cols   = ['#2471A3' if 'Tree' in l
          else '#85B7EB' if 'Kernel' in l
          else '#BA7517' if 'LIME' in l
          else '#888780' if 'Perm' in l
          else '#7F77DD' if 'IntGrad' in l or 'Deep' in l
          else '#1D9E75' if 'Anchors' in l
          else '#D4537E' for l in labels]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(labels, vals, color=cols, edgecolor='none')
ax.set_yscale('log')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()*1.1,
            f'{v:.1f}', ha='center', fontsize=7,
            rotation=45)
ax.set_ylabel('Time (ms/sample, log scale)', fontsize=10)
ax.set_title(f'{DOMAIN}: XAI Computation Time per Sample',
             fontsize=11)
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
ax.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('fig4_time_healthcare.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig4_time_healthcare.pdf saved")

# ── CELL 22 ── FIG 5: Stability Heatmap ──────────────────────────
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
na_mask2 = np.zeros_like(mat_s)
na_mask2[np.isnan(mat_s)] = 1.
ax.imshow(na_mask2,
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
    f'{DOMAIN}: Explanation Stability (15 runs, top-5)\n'
    f'green=stable | red=unstable | grey=N/A',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig5_stability_heatmap_healthcare.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig5_stability_heatmap_healthcare.pdf saved")

# ── CELL 23 ── Summary Table ──────────────────────────────────────
print("=" * 70)
print(f"DOMAIN 1 — HEALTHCARE | FINAL SUMMARY TABLE")
print("=" * 70)
rows = []
for mn in list(PERF.keys()):
    sk = ('TreeSHAP' if mn in
          ['Random Forest','XGBoost','LightGBM']
          else 'KernelSHAP')
    rows.append({
        'Domain'   : DOMAIN,
        'Model'    : mn,
        'AUC'      : round(PERF[mn]['auc'], 3),
        'SHAP_type': sk,
        'SHAP_ms'  : round(XAI[sk].get(mn,{}).get('tps',0)*1000, 2),
        'LIME_ms'  : round(XAI['LIME'].get(mn,{}).get('tps',0)*1000, 2),
        'SHAP_faith': faith.get(f'{sk}|{mn}', None),
        'SHAP_stab' : STAB.get(f'{sk}|{mn}', None),
        'LIME_stab' : STAB.get(f'LIME|{mn}', None),
        'SHAP_LIME_r': round(r, 3),
    })
df_sum = pd.DataFrame(rows)
print(df_sum.to_string(index=False))
df_sum.to_csv('summary_D1_healthcare.csv', index=False)
print("\n✓ summary_D1_healthcare.csv saved")
print("✓ ALL CELLS COMPLETE — HEALTHCARE DOMAIN")
