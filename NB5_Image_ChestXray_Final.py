# ╔══════════════════════════════════════════════════════════════════╗
# ║  XAI EMPIRICAL STUDY — NOTEBOOK 5: MEDICAL IMAGING DOMAIN      ║
# ║  Dataset  : Chest X-Ray Pneumonia (Kaggle / Kermany 2018)       ║
# ║  Size     : 5,863 images | Binary: NORMAL vs PNEUMONIA          ║
# ║  Paper    : "When to Use Which Explainable AI Method?"          ║
# ║  Authors  : Khaled Mahmud Sujon et al.                          ║
# ║  GPU      : Required — use Google Colab T4 GPU runtime           ║
# ║                                                                  ║
# ║  DATA SETUP:                                                     ║
# ║    Upload kaggle.json when prompted (Cell 3)                     ║
# ║    OR manually upload the chest_xray folder to Colab             ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── CELL 1 ── Install Libraries ───────────────────────────────────
# !pip install -q shap lime opencv-python-headless scikit-image

# ── CELL 2 ── Imports ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size':11,'figure.dpi':150})
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cv2
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import time, warnings, os, gc
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, confusion_matrix,
                              classification_report)

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

DOMAIN     = "Medical Imaging"
IMG_SIZE   = 128
BATCH      = 32
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
print("✓ Libraries loaded")

# ── CELL 3 ── Kaggle Dataset Download ────────────────────────────
from google.colab import files
print("Upload your kaggle.json:")
uploaded = files.upload()
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'wb') as f:
    f.write(list(uploaded.values())[0])
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
os.system('kaggle datasets download -d paultimothymooney/'
          'chest-xray-pneumonia --unzip -p /content/')
print("✓ Dataset downloaded")

# ── CELL 4 ── Data Generators ────────────────────────────────────
TRAIN_DIR = '/content/chest_xray/train'
TEST_DIR  = '/content/chest_xray/test'
VAL_DIR   = '/content/chest_xray/val'

def make_generators(train_dir, test_dir, val_dir,
                    img_size=IMG_SIZE, batch=BATCH):
    train_gen = ImageDataGenerator(
        rescale=1./255, rotation_range=15,
        width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1,
        horizontal_flip=True, fill_mode='nearest')
    test_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(
        train_dir, target_size=(img_size, img_size),
        batch_size=batch, class_mode='binary',
        color_mode='rgb', shuffle=True, seed=42)
    val_flow = test_gen.flow_from_directory(
        val_dir, target_size=(img_size, img_size),
        batch_size=batch, class_mode='binary',
        color_mode='rgb', shuffle=False)
    test_flow = test_gen.flow_from_directory(
        test_dir, target_size=(img_size, img_size),
        batch_size=batch, class_mode='binary',
        color_mode='rgb', shuffle=False)
    return train_flow, val_flow, test_flow

train_flow, val_flow, test_flow = make_generators(
    TRAIN_DIR, TEST_DIR, VAL_DIR)
print(f"Train: {train_flow.samples} | "
      f"Val: {val_flow.samples} | "
      f"Test: {test_flow.samples}")

# ── CELL 5 ── Build Custom CNN ───────────────────────────────────
def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inp = keras.Input(shape=input_shape, name='input')
    x   = layers.Conv2D(32, 3, padding='same',
                         activation='relu',
                         name='conv2d')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(64, 3, padding='same',
                         activation='relu',
                         name='conv2d_1')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(128, 3, padding='same',
                         activation='relu',
                         name='conv2d_2')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(256, 3, padding='same',
                         activation='relu',
                         name='last_conv')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m   = keras.Model(inp, out, name='CustomCNN')
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return m

cnn = build_custom_cnn()
cnn.summary()

# ── CELL 6 ── Train Custom CNN ───────────────────────────────────
cb = [
    keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True,
        monitor='val_auc', mode='max'),
    keras.callbacks.ReduceLROnPlateau(
        patience=3, factor=0.5, verbose=1,
        monitor='val_auc', mode='max')
]
history_cnn = cnn.fit(
    train_flow, epochs=30,
    validation_data=val_flow,
    callbacks=cb, verbose=1)

# Evaluate CNN
test_flow.reset()
cnn_probs = cnn.predict(test_flow, verbose=0).ravel()
cnn_preds = (cnn_probs > 0.5).astype(int)
cnn_true  = test_flow.classes
cnn_auc   = roc_auc_score(cnn_true, cnn_probs)
cnn_acc   = accuracy_score(cnn_true, cnn_preds)
cnn_f1    = f1_score(cnn_true, cnn_preds)
cm_cnn    = confusion_matrix(cnn_true, cnn_preds)
cnn_tn, cnn_fp, cnn_fn, cnn_tp = cm_cnn.ravel()
cnn_sens  = cnn_tp / (cnn_tp + cnn_fn)
cnn_spec  = cnn_tn / (cnn_tn + cnn_fp)
print(f"Custom CNN — Acc:{cnn_acc:.3f} | F1:{cnn_f1:.3f} | "
      f"AUC:{cnn_auc:.3f} | Sens:{cnn_sens:.3f} | "
      f"Spec:{cnn_spec:.3f} | FP:{cnn_fp} | FN:{cnn_fn}")
cnn.save('cnn_model.h5')
conv_layer = 'last_conv'
print(f"✓ CNN saved | conv layer: {conv_layer}")

# ── CELL 7 ── Build & Train ResNet-50V2 ──────────────────────────
def build_resnet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base = ResNet50V2(include_top=False, weights='imagenet',
                      input_shape=input_shape)
    base.trainable = False
    inp = keras.Input(shape=input_shape)
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m   = keras.Model(inp, out, name='ResNet50V2_TL')
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return m

resnet = build_resnet()

# Phase 1: frozen base
cb_res = [keras.callbacks.EarlyStopping(
    patience=3, restore_best_weights=True,
    monitor='val_auc', mode='max')]
resnet.fit(train_flow, epochs=15,
           validation_data=val_flow,
           callbacks=cb_res, verbose=1)

# Phase 2: fine-tune top-10 layers
print("Phase 2: Fine-tuning top-10 layers ...")
resnet.layers[1].trainable = True
for layer in resnet.layers[1].layers[:-10]:
    layer.trainable = False
resnet.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')])
resnet.fit(train_flow, epochs=5,
           validation_data=val_flow,
           callbacks=cb_res, verbose=1)

# Evaluate ResNet
test_flow.reset()
res_probs = resnet.predict(test_flow, verbose=0).ravel()
res_preds = (res_probs > 0.5).astype(int)
res_auc   = roc_auc_score(cnn_true, res_probs)
res_acc   = accuracy_score(cnn_true, res_preds)
res_f1    = f1_score(cnn_true, res_preds)
cm_res    = confusion_matrix(cnn_true, res_preds)
res_tn, res_fp, res_fn, res_tp = cm_res.ravel()
res_sens  = res_tp / (res_tp + res_fn)
res_spec  = res_tn / (res_tn + res_fp)
print(f"ResNet-50V2 TL — Acc:{res_acc:.3f} | F1:{res_f1:.3f} | "
      f"AUC:{res_auc:.3f} | Sens:{res_sens:.3f} | "
      f"Spec:{res_spec:.3f} | FP:{res_fp} | FN:{res_fn}")
resnet.save('resnet_model.h5')
print("✓ ResNet saved")

# ── CELL 8 ── Load Balanced XAI Sample ───────────────────────────
y_all_full = test_flow.classes
norm_idx   = np.where(y_all_full == 0)[0]
pneu_idx   = np.where(y_all_full == 1)[0]

BALANCED_4  = list(norm_idx[:2]) + list(pneu_idx[:2])
BALANCED_10 = list(norm_idx[:5]) + list(pneu_idx[:5])
BALANCED_20 = list(norm_idx[:10]) + list(pneu_idx[:10])

def load_single_img(flow, target_idx, img_size=IMG_SIZE):
    fpath = flow.filepaths[target_idx]
    img   = Image.open(fpath).convert('RGB')
    img   = img.resize((img_size, img_size))
    return np.array(img, dtype=np.float32) / 255.0

print("Loading 20 balanced images ...")
X_imgs = np.array([load_single_img(test_flow, i)
                   for i in BALANCED_20])
y_imgs = np.array([y_all_full[i] for i in BALANCED_20])
print(f"X_imgs: {X_imgs.shape} | "
      f"NORMAL: {(y_imgs==0).sum()} | "
      f"PNEUMONIA: {(y_imgs==1).sum()}")

# ── CELL 9 ── GRAD-CAM Helpers ───────────────────────────────────
def get_gradcam(model, img, conv_layer_name):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output,
                 model.output])
    img_t = tf.cast(np.expand_dims(img, 0), tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_t)
        loss = preds[0]
    grads        = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out     = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    return heatmap

def get_gradcam_pp(model, img, conv_layer_name):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output,
                 model.output])
    img_t = tf.cast(np.expand_dims(img, 0), tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        conv_out, preds = grad_model(img_t)
        loss = preds[0]
    grads    = tape.gradient(loss, conv_out)
    grads_2  = tape.gradient(grads, conv_out)
    grads_3  = tape.gradient(grads_2, conv_out)
    alpha    = grads_2 / (2*grads_2 + conv_out*grads_3 + 1e-8)
    weights  = tf.reduce_mean(
        alpha * tf.nn.relu(grads), axis=(0,1,2))
    heatmap  = tf.reduce_sum(
        conv_out[0] * weights, axis=-1).numpy()
    heatmap  = np.maximum(heatmap, 0)
    heatmap  = heatmap / (heatmap.max() + 1e-8)
    heatmap  = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    del tape
    return heatmap

@tf.function
def _grad_fn(inp, model):
    with tf.GradientTape() as tape:
        tape.watch(inp)
        pred = model(inp)
    return tape.gradient(pred, inp)

def intgrad_image(model, img, steps=50):
    img_t = tf.cast(img, tf.float32)
    bl    = tf.zeros_like(img_t)
    al    = tf.linspace(0., 1., steps+1)
    pts   = tf.stack([bl + a*(img_t-bl) for a in al])
    gs    = _grad_fn(pts, model)
    return ((img_t-bl) * tf.reduce_mean(
        (gs[:-1]+gs[1:])/2., 0)).numpy()

# ── CELL 10 ── Generate All XAI Methods ──────────────────────────
XAI_IMG  = defaultdict(dict)
TIME_IMG = {}

# GRAD-CAM
print("▶ GRAD-CAM ...")
t0    = time.perf_counter()
gcams = [get_gradcam(cnn, X_imgs[i], conv_layer)
          for i in range(20)]
t1    = time.perf_counter()
XAI_IMG['GRADCAM']['CNN'] = {
    'heatmaps': gcams, 'tps': (t1-t0)/20}
TIME_IMG['GRAD-CAM'] = (t1-t0)/20
print(f"  {(t1-t0)/20*1000:.1f} ms/sample")

# GRAD-CAM++
print("▶ GRAD-CAM++ ...")
t0     = time.perf_counter()
gcams2 = [get_gradcam_pp(cnn, X_imgs[i], conv_layer)
           for i in range(20)]
t1     = time.perf_counter()
XAI_IMG['GRADCAMpp']['CNN'] = {
    'heatmaps': gcams2, 'tps': (t1-t0)/20}
TIME_IMG['GRAD-CAM++'] = (t1-t0)/20
print(f"  {(t1-t0)/20*1000:.1f} ms/sample")

# Integrated Gradients
print("▶ Integrated Gradients ...")
t0      = time.perf_counter()
ig_maps = [intgrad_image(cnn, X_imgs[i])
            for i in range(20)]
t1      = time.perf_counter()
XAI_IMG['IntGrad']['CNN'] = {
    'values': ig_maps, 'tps': (t1-t0)/20}
TIME_IMG['IntGrad'] = (t1-t0)/20
print(f"  {(t1-t0)/20*1000:.1f} ms/sample")

# DeepSHAP
print("▶ DeepSHAP ...")
bg_shap = X_imgs[:20].astype(np.float32)
t0      = time.perf_counter()
deep_e  = shap.DeepExplainer(cnn, bg_shap)
sv_imgs = deep_e.shap_values(X_imgs.astype(np.float32))
sv_imgs = sv_imgs[0] if isinstance(sv_imgs, list) else sv_imgs
sv_imgs = np.array(sv_imgs)
t1      = time.perf_counter()
XAI_IMG['DeepSHAP']['CNN'] = {
    'values': sv_imgs, 'tps': (t1-t0)/20}
TIME_IMG['DeepSHAP'] = (t1-t0)/20
print(f"  shape={sv_imgs.shape} | "
      f"{(t1-t0)/20*1000:.1f} ms/sample")
if np.abs(sv_imgs).max() < 0.001:
    print("  ⚠️  DeepSHAP near-zero attributions "
          "— known limitation for deep CNNs")

# LIME Image
print("▶ LIME Image (~5-10 min) ...")
lime_img_exp = lime_image.LimeImageExplainer(random_state=42)

def predict_fn_cnn(imgs):
    probs = cnn.predict(
        imgs.astype(np.float32), verbose=0).ravel()
    return np.column_stack([1-probs, probs])

lime_maps, lime_masks = [], []
t0 = time.perf_counter()
for i in range(20):
    exp = lime_img_exp.explain_instance(
        X_imgs[i], predict_fn_cnn,
        top_labels=1, hide_color=0,
        num_samples=1000)
    _, mask = exp.get_image_and_mask(
        exp.top_labels[0], positive_only=True,
        num_features=5, hide_rest=False)
    lime_maps.append(mask.astype(float))
t1 = time.perf_counter()
XAI_IMG['LIMEImage']['CNN'] = {
    'maps': lime_maps, 'tps': (t1-t0)/20}
TIME_IMG['LIME Image'] = (t1-t0)/20
print(f"  {(t1-t0)/20*1000:.1f} ms/sample")
lime_ratio = TIME_IMG['LIME Image'] / TIME_IMG['GRAD-CAM']
print(f"  LIME is {lime_ratio:.0f}× slower than GRAD-CAM")

# ── CELL 11 ── FIG 1: Main Saliency Grid ─────────────────────────
def pct_norm(m, lo=2, hi=98):
    p_lo = np.percentile(m, lo)
    p_hi = np.percentile(m, hi)
    return np.clip((m-p_lo)/(p_hi-p_lo+1e-8), 0, 1)

def smooth_saliency(m, ksize=11):
    m_u8    = (pct_norm(m)*255).astype(np.uint8)
    blurred = cv2.GaussianBlur(m_u8, (ksize,ksize), 0)
    return blurred.astype(float) / 255.

def ig_to_saliency(ig_map, ksize=11):
    ig_abs  = np.abs(ig_map).sum(axis=-1)
    return smooth_saliency(ig_abs, ksize)

disp_idx  = BALANCED_4
n_disp    = len(disp_idx)
n_methods = 5
cmaps     = ['jet','plasma','viridis','hot',None]
m_titles  = ['Original','GRAD-CAM','GRAD-CAM++',
             'GRAD-CAM\nThreshold','IntGrad','LIME']

fig = plt.figure(figsize=(18, n_disp*3.5))
gs  = gridspec.GridSpec(n_disp, n_methods+1, hspace=0.3, wspace=0.05)

for row, xi in enumerate(disp_idx):
    img  = X_imgs[xi]
    lbl  = CLASS_NAMES[int(y_imgs[xi])]
    pred = float(cnn.predict(
        np.expand_dims(img,0), verbose=0).ravel()[0])
    conf = pred if y_imgs[xi]==1 else 1-pred

    # Col 0: Original
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(img)
    ax.set_title(f'{lbl}\npred={pred:.2f}',
                 fontsize=8, pad=2)
    ax.axis('off')
    if row == 0:
        ax.set_title(f'Original\n{lbl} pred={pred:.2f}',
                     fontsize=8)

    saliency_maps = [
        gcams[xi], gcams2[xi],
        (gcams[xi] > np.percentile(gcams[xi], 60)).astype(float),
        ig_to_saliency(ig_maps[xi]),
        lime_maps[xi].astype(float)
    ]

    for col, (sal, cmap) in enumerate(zip(saliency_maps, cmaps)):
        ax = fig.add_subplot(gs[row, col+1])
        ax.imshow(img)
        if cmap:
            ax.imshow(sal, alpha=0.55, cmap=cmap,
                      vmin=0, vmax=1)
        else:
            ax.imshow(
                mark_boundaries(img, sal.astype(int)),
                alpha=0.9)
        ax.axis('off')
        if row == 0:
            ax.set_title(m_titles[col+1], fontsize=8)

plt.suptitle(
    f'{DOMAIN}: XAI Saliency Comparison\n'
    f'Row 1-2: NORMAL | Row 3-4: PNEUMONIA',
    fontsize=11, y=1.01)
plt.savefig('fig1_saliency_grid_image.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ fig1_saliency_grid_image.pdf saved")

# ── CELL 12 ── FIG 2: Computation Time ───────────────────────────
TCOLS = {
    'GRAD-CAM' :'#D85A30', 'GRAD-CAM++':'#D85A30',
    'IntGrad'  :'#7F77DD', 'DeepSHAP'  :'#AFA9EC',
    'LIME Image':'#BA7517'
}
ts   = pd.Series(TIME_IMG)
cols = [TCOLS.get(k,'#888780') for k in ts.index]

fig, ax = plt.subplots(figsize=(10,4))
bars = ax.barh(ts.index, ts.values*1000,
               color=cols, edgecolor='none')
for bar, v in zip(bars, ts.values*1000):
    ax.text(bar.get_width()+10,
            bar.get_y()+bar.get_height()/2,
            f'{v:.1f} ms', va='center', fontsize=9)
ax.set_xlabel('Time (ms/sample)', fontsize=10)
ax.set_title(
    f'{DOMAIN}: XAI Computation Time\n'
    f'LIME Image = {lime_ratio:.0f}× slower than GRAD-CAM',
    fontsize=11)
ax.grid(alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('fig2_time_image.pdf',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ fig2_time_image.pdf saved")

# ── CELL 13 ── FIG 3: GRAD-CAM Across Layers ─────────────────────
conv_layers_cnn = [l.name for l in cnn.layers
                   if 'conv2d' in l.name.lower()]
print(f"CNN conv layers: {conv_layers_cnn}")

disp2  = BALANCED_4[:2]   # 1 NORMAL + 1 PNEUMONIA
n_layers = len(conv_layers_cnn)

fig, axes = plt.subplots(2, n_layers, figsize=(n_layers*4, 8))
for row, xi in enumerate(disp2):
    img = X_imgs[xi]
    lbl = CLASS_NAMES[int(y_imgs[xi])]
    for col, layer_name in enumerate(conv_layers_cnn):
        hm = get_gradcam(cnn, img, layer_name)
        axes[row,col].imshow(img)
        axes[row,col].imshow(hm, alpha=0.55,
                              cmap='jet', vmin=0, vmax=1)
        axes[row,col].axis('off')
        if row == 0:
            axes[row,col].set_title(
                layer_name, fontsize=9)
        if col == 0:
            axes[row,col].set_ylabel(lbl, fontsize=9)
plt.suptitle(
    f'{DOMAIN}: GRAD-CAM Across Convolutional Layers\n'
    f'Row 1: NORMAL | Row 2: PNEUMONIA',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig3_gradcam_layers.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ fig3_gradcam_layers.pdf saved")

# ── CELL 14 ── FIG 4: Training History ───────────────────────────
def smooth(values, weight=0.6):
    smoothed, last = [], values[0]
    for v in values:
        last = last*weight + v*(1-weight)
        smoothed.append(last)
    return smoothed

fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, metric, c_tr, c_val, label in [
    (axes[0],'accuracy','#378ADD','#85B7EB','Accuracy'),
    (axes[1],'auc',     '#1D9E75','#9FE1CB','AUC'),
]:
    tr_v  = history_cnn.history.get(metric, [])
    val_v = history_cnn.history.get(f'val_{metric}', [])
    ep    = range(1, len(tr_v)+1)
    ax.plot(ep, tr_v, color=c_tr, alpha=0.3, lw=1)
    ax.plot(ep, smooth(tr_v), color=c_tr, lw=2,
            label=f'Train {label}')
    if val_v:
        ax.plot(ep, val_v, color=c_val, alpha=0.3, lw=1)
        ax.plot(ep, smooth(val_v), color=c_val, lw=2,
                ls='--', label=f'Val {label}')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Custom CNN: {label}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
plt.suptitle(f'{DOMAIN}: Custom CNN Training History',
             fontsize=11)
plt.tight_layout()
plt.savefig('fig4_training_history_image.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ fig4_training_history_image.pdf saved")

# ── CELL 15 ── FIG 5: Integrated Gradients ───────────────────────
B10_LOCAL = list(range(20))[:10]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for plot_i, xi in enumerate(B10_LOCAL):
    ax   = axes[plot_i//5, plot_i%5]
    img  = X_imgs[xi]
    lbl  = CLASS_NAMES[int(y_imgs[xi])]
    pred = float(cnn.predict(
        np.expand_dims(img,0), verbose=0).ravel()[0])
    ig_clean = ig_to_saliency(ig_maps[xi])
    ax.imshow(img)
    ax.imshow(ig_clean, alpha=0.70, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'{lbl}\np={pred:.2f}', fontsize=8)
    ax.axis('off')
plt.suptitle(
    f'{DOMAIN}: Integrated Gradients Attribution\n'
    f'Top row: NORMAL | Bottom row: PNEUMONIA',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig5_intgrad_pixels_image.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ fig5_intgrad_pixels_image.pdf saved")

# ── CELL 16 ── FIG 6: CNN vs ResNet GRAD-CAM ─────────────────────
resnet_base     = resnet.get_layer('resnet50v2')
res_conv_layers = [l.name for l in resnet_base.layers
                   if 'conv' in l.name.lower()]
res_layer       = res_conv_layers[-1]
print(f"ResNet last conv: {res_layer}")

def get_gradcam_resnet(outer_model, inner_base, img):
    img_t = tf.cast(np.expand_dims(img,0), tf.float32)
    with tf.GradientTape() as tape:
        features = inner_base(img_t, training=False)
        tape.watch(features)
        x = outer_model.layers[-3](features)
        x = outer_model.layers[-2](x)
        pred = outer_model.layers[-1](x)
        loss = pred[0]
    grads = tape.gradient(loss, features)
    if grads is None:
        return np.zeros((IMG_SIZE, IMG_SIZE))
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    hm = features[0] @ pooled[..., tf.newaxis]
    hm = tf.squeeze(hm).numpy()
    hm = np.maximum(hm, 0)
    hm = hm / (hm.max() + 1e-8)
    return cv2.resize(hm, (IMG_SIZE, IMG_SIZE))

disp3 = BALANCED_4
fig, axes = plt.subplots(2, len(disp3), figsize=(16, 8))
for col_i, xi in enumerate(disp3):
    img = X_imgs[xi]
    lbl = CLASS_NAMES[int(y_imgs[xi])]
    # CNN
    hm_cnn = gcams[xi]
    axes[0,col_i].imshow(img)
    axes[0,col_i].imshow(hm_cnn, alpha=0.55,
                          cmap='jet', vmin=0, vmax=1)
    axes[0,col_i].set_title(f'{lbl}', fontsize=9)
    axes[0,col_i].axis('off')
    # ResNet
    hm_res = get_gradcam_resnet(resnet, resnet_base, img)
    axes[1,col_i].imshow(img)
    axes[1,col_i].imshow(hm_res, alpha=0.55,
                          cmap='hot', vmin=0, vmax=1)
    axes[1,col_i].axis('off')

axes[0,0].set_ylabel('Custom CNN\n(JET)', fontsize=9)
axes[1,0].set_ylabel('ResNet-50V2\n(HOT)', fontsize=9)
plt.suptitle(
    f'{DOMAIN}: GRAD-CAM CNN vs ResNet-50V2\n'
    f'CNN: focused | ResNet: diffuse whole-image',
    fontsize=11)
plt.tight_layout()
plt.savefig('fig6_cnn_vs_resnet_gradcam.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ fig6_cnn_vs_resnet_gradcam.pdf saved")

# ── CELL 17 ── FIG A: Misclassification Analysis ─────────────────
test_flow.reset()
all_probs = cnn.predict(test_flow, verbose=0).ravel()
all_preds = (all_probs > 0.5).astype(int)
all_true  = y_all_full

wrong_idx  = np.where(all_preds != all_true)[0]
wrong_conf = np.abs(all_probs[wrong_idx] - 0.5)
top4_wrong = wrong_idx[np.argsort(wrong_conf)[::-1][:4]]

fp_idx = [i for i in top4_wrong if all_true[i]==0][:2]
fn_idx = [i for i in top4_wrong if all_true[i]==1][:2]
disp_err = fp_idx + fn_idx

print(f"Total misclassified: {len(wrong_idx)}/624 "
      f"({len(wrong_idx)/624*100:.1f}%)")
print(f"FP (NORMAL→PNEUMONIA): {(all_preds[all_true==0]==1).sum()}")
print(f"FN (PNEUMONIA→NORMAL): {(all_preds[all_true==1]==0).sum()}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col_i, img_idx in enumerate(disp_err):
    img  = load_single_img(test_flow, img_idx)
    lbl  = CLASS_NAMES[all_true[img_idx]]
    pred = CLASS_NAMES[all_preds[img_idx]]
    conf = float(all_probs[img_idx])
    hm   = get_gradcam(cnn, img, conv_layer)
    axes[0,col_i].imshow(img)
    axes[0,col_i].set_title(
        f'True:{lbl}\nPred:{pred} ({conf:.2f})',
        fontsize=8, color='#C0392B')
    axes[0,col_i].axis('off')
    axes[1,col_i].imshow(img)
    axes[1,col_i].imshow(hm, alpha=0.55,
                          cmap='jet', vmin=0, vmax=1)
    axes[1,col_i].axis('off')
    if col_i == 0:
        axes[0,col_i].set_ylabel('Image', fontsize=9)
        axes[1,col_i].set_ylabel('GRAD-CAM', fontsize=9)

plt.suptitle(
    f'{DOMAIN}: Most Confident Misclassifications\n'
    f'Col 1-2: FP (NORMAL→PNEUMONIA) | '
    f'Col 3-4: FN (PNEUMONIA→NORMAL)',
    fontsize=11)
plt.tight_layout()
plt.savefig('figA_misclassification_analysis.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ figA_misclassification_analysis.pdf saved")

# ── CELL 18 ── FIG B: Confidence + Clinical Metrics ──────────────
test_flow.reset()
res_probs_all = resnet.predict(test_flow, verbose=0).ravel()
res_preds_all = (res_probs_all > 0.5).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# Panel 1: CNN confidence
for lbl_idx, (lbl, col) in enumerate(
        [('NORMAL','#1D9E75'),('PNEUMONIA','#D85A30')]):
    mask = all_true == lbl_idx
    axes[0].hist(all_probs[mask], bins=25, alpha=0.7,
                  color=col, label=lbl)
axes[0].set_xlabel('Predicted Probability', fontsize=10)
axes[0].set_ylabel('Count', fontsize=10)
axes[0].set_title('CNN Confidence Distribution', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].axvline(0.5, color='black', ls='--', lw=1.5)
axes[0].grid(alpha=0.2)

# Panel 2: ResNet confidence
for lbl_idx, (lbl, col) in enumerate(
        [('NORMAL','#1D9E75'),('PNEUMONIA','#D85A30')]):
    mask = all_true == lbl_idx
    axes[1].hist(res_probs_all[mask], bins=25, alpha=0.7,
                  color=col, label=lbl)
axes[1].set_xlabel('Predicted Probability', fontsize=10)
axes[1].set_title('ResNet-50V2 Confidence Distribution',
                   fontsize=10)
axes[1].legend(fontsize=9)
axes[1].axvline(0.5, color='black', ls='--', lw=1.5)
axes[1].grid(alpha=0.2)

# Panel 3: Clinical metrics comparison
metrics   = ['Sensitivity','Specificity','Precision','F1','AUC']
cnn_vals  = [cnn_sens, cnn_spec,
              cnn_tp/(cnn_tp+cnn_fp+1e-8),
              cnn_f1, cnn_auc]
res_vals  = [res_sens, res_spec,
              res_tp/(res_tp+res_fp+1e-8),
              res_f1, res_auc]
x_pos = np.arange(len(metrics))
axes[2].bar(x_pos-0.2, cnn_vals, 0.35,
             label='Custom CNN', color='#2471A3',
             edgecolor='none')
axes[2].bar(x_pos+0.2, res_vals, 0.35,
             label='ResNet-50V2', color='#1E8449',
             edgecolor='none')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(metrics, rotation=20, ha='right',
                         fontsize=9)
axes[2].set_ylim(0, 1.1)
axes[2].set_title('Clinical Metrics Comparison', fontsize=10)
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.2, axis='y')

plt.suptitle(f'{DOMAIN}: Confidence Distributions & '
             f'Clinical Metrics', fontsize=11)
plt.tight_layout()
plt.savefig('figB_confidence_clinical_metrics.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ figB_confidence_clinical_metrics.pdf saved")

# ── CELL 19 ── FIG C: XAI Quantitative Evaluation ────────────────
def saliency_iou(m1, m2, threshold=0.5):
    b1 = (m1>threshold).astype(int)
    b2 = (m2>threshold).astype(int)
    inter = (b1&b2).sum()
    union = (b1|b2).sum()
    return inter/(union+1e-8) if union>0 else 1.0

def lung_coverage(hm, threshold=0.4):
    return float((hm>threshold).mean())

# Threshold robustness
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
gcam_norm_iou, gcam_pneu_iou = [], []
gcampp_norm_iou, gcampp_pneu_iou = [], []

for thr in thresholds:
    norm_iou  = np.mean([saliency_iou(gcams[i], gcams[i], thr)
                          for i in range(10) if y_imgs[i]==0])
    pneu_iou  = np.mean([saliency_iou(gcams[i], gcams[i], thr)
                          for i in range(10) if y_imgs[i]==1])
    norm_iou2 = np.mean([
        saliency_iou(gcams2[i], gcams2[i], thr)
        for i in range(10) if y_imgs[i]==0])
    pneu_iou2 = np.mean([
        saliency_iou(gcams2[i], gcams2[i], thr)
        for i in range(10) if y_imgs[i]==1])
    gcam_norm_iou.append(norm_iou)
    gcam_pneu_iou.append(pneu_iou)
    gcampp_norm_iou.append(norm_iou2)
    gcampp_pneu_iou.append(pneu_iou2)

# Coverage
cov_norm = np.mean([lung_coverage(gcams[i])
                     for i in range(20) if y_imgs[i]==0])
cov_pneu = np.mean([lung_coverage(gcams[i])
                     for i in range(20) if y_imgs[i]==1])

# Confidence vs coverage
confs = []
covs  = []
for i in range(20):
    p = float(cnn.predict(
        np.expand_dims(X_imgs[i],0), verbose=0).ravel()[0])
    confs.append(p if y_imgs[i]==1 else 1-p)
    covs.append(lung_coverage(gcams[i]))
r_cc, p_cc = pearsonr(confs, covs)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Threshold robustness
axes[0].plot(thresholds, gcam_norm_iou,
             'o-', color='#1D9E75', label='GRAD-CAM NORMAL')
axes[0].plot(thresholds, gcam_pneu_iou,
             's-', color='#D85A30', label='GRAD-CAM PNEUMONIA')
axes[0].plot(thresholds, gcampp_norm_iou,
             'o--', color='#2471A3', label='GRAD-CAM++ NORMAL')
axes[0].plot(thresholds, gcampp_pneu_iou,
             's--', color='#7D3C98', label='GRAD-CAM++ PNEUMONIA')
axes[0].set_xlabel('Activation Threshold', fontsize=10)
axes[0].set_ylabel('IoU', fontsize=10)
axes[0].set_title('Threshold Robustness', fontsize=10)
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.2)

# Panel 2: Coverage
axes[1].bar(['NORMAL','PNEUMONIA'], [cov_norm, cov_pneu],
             color=['#1D9E75','#D85A30'], edgecolor='none')
axes[1].text(0, cov_norm+0.003, f'{cov_norm:.3f}',
              ha='center', fontsize=10)
axes[1].text(1, cov_pneu+0.003, f'{cov_pneu:.3f}',
              ha='center', fontsize=10)
axes[1].set_ylabel('Activation Coverage', fontsize=10)
axes[1].set_title('Mean GRAD-CAM Coverage', fontsize=10)
axes[1].grid(alpha=0.2, axis='y')

# Panel 3: Confidence vs coverage
for i in range(20):
    c = '#D85A30' if y_imgs[i]==1 else '#1D9E75'
    axes[2].scatter(confs[i], covs[i], color=c, s=60,
                     alpha=0.8, edgecolors='white', lw=0.5)
axes[2].set_xlabel('Prediction Confidence', fontsize=10)
axes[2].set_ylabel('Saliency Coverage', fontsize=10)
axes[2].set_title(
    f'Confidence vs Coverage\nr={r_cc:.3f}  p={p_cc:.3f}',
    fontsize=10)
axes[2].grid(alpha=0.2)

plt.suptitle(f'{DOMAIN}: XAI Quantitative Evaluation',
             fontsize=11)
plt.tight_layout()
plt.savefig('figC_xai_quantitative_evaluation.pdf',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ figC_xai_quantitative_evaluation.pdf saved")

# ── CELL 20 ── Summary Tables ─────────────────────────────────────
print("=" * 70)
print(f"DOMAIN 5 — {DOMAIN} | SUMMARY")
print("=" * 70)

t1_img = pd.DataFrame([
    {'Model':'Custom CNN','Accuracy':round(cnn_acc,3),
     'F1':round(cnn_f1,3),'AUC':round(cnn_auc,3),
     'Sensitivity':round(cnn_sens,3),
     'Specificity':round(cnn_spec,3),
     'FP':int(cnn_fp),'FN':int(cnn_fn)},
    {'Model':'ResNet-50V2 TL','Accuracy':round(res_acc,3),
     'F1':round(res_f1,3),'AUC':round(res_auc,3),
     'Sensitivity':round(res_sens,3),
     'Specificity':round(res_spec,3),
     'FP':int(res_fp),'FN':int(res_fn)},
])
print("\nTABLE 1 — MODEL PERFORMANCE:")
print(t1_img.to_string(index=False))

t2_img = pd.DataFrame([
    {'Method':k, 'Time_ms':round(v*1000,2),
     'Relative_to_GRADCAM':round(v/TIME_IMG['GRAD-CAM'],1),
     'Deployment':'Real-time' if v<0.1 else 'Batch/Offline'}
    for k,v in TIME_IMG.items()
])
print("\nTABLE 2 — COMPUTATION TIME:")
print(t2_img.to_string(index=False))

t1_img.to_csv('table1_performance_image.csv', index=False)
t2_img.to_csv('table2_time_image.csv', index=False)
print("\n✓ summary tables saved")
print("✓ ALL CELLS COMPLETE — MEDICAL IMAGING DOMAIN")
