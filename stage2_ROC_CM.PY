#!/usr/bin/env python3
# ---------------------------------------------------------------
#  Stage-2  Çok-Sınıflı  Random-Forest  Modeli  İçin  Metri̇k  Çizimi
#  • Confusion Matrix (FIGURE A)
#  • Mikro Ortalama ROC Eğrisi (FIGURE B)
# ---------------------------------------------------------------

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------
# 1) Dosya Yolları — Burayı Güncelleyin
# ---------------------------------------------------------------
MODEL_FILE = "models/stage2_pipe.joblib"               # Eğitilmiş Stage-2 artefaktı
TEST_FILE  = "UNSW_NB15_testing-set.csv"         # Resmî test CSV

# ---------------------------------------------------------------
# 2) Model ve Veri Yükleme
# ---------------------------------------------------------------
print("› Model yükleniyor…")
pipe = joblib.load(MODEL_FILE)                   # Pipeline: preprocess + RF

print("› Test verisi okunuyor…")
df = pd.read_csv(TEST_FILE)
df.columns = df.columns.str.strip().str.lower()

# Stage-2 yalnızca ‘Attack’ etiketli kayıtları işler
df_attack = df[df["label"] == 1].copy()

# Kategori adlarını eğitimde kullanılan altı sınıfa indirgeme
mapping = {
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "dos": "DoS",
    "reconnaissance": "Recon",
    "generic": "Generic"
}
df_attack["attack_cat"] = (
    df_attack["attack_cat"]
    .str.strip().str.lower()
    .map(mapping)
    .fillna("Other")
)

y_true = df_attack["attack_cat"].values
X_test = df_attack.drop(columns=["label", "attack_cat"], errors="ignore")

print("› Tahmin yapılıyor…")
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)             # (n_samples, n_classes)

# ---------------------------------------------------------------
# 3) Karışıklık Matrisi  – FIGURE A
# ---------------------------------------------------------------
labels = pipe.classes_                           # LabelEncoder sırası
cm = confusion_matrix(y_true, y_pred, labels=labels)

fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
disp.plot(ax=ax_cm, colorbar=True, xticks_rotation=45)
ax_cm.set_title("FIGURE 4. Confusion matrix of the Stage-2 multiclass classifier")
plt.tight_layout()
fig_cm.savefig("stage2_confusion.png", dpi=300)
plt.close(fig_cm)
print("✓ stage2_confusion.png kaydedildi")

# ---------------------------------------------------------------
# 4) Mikro-Ortalama ROC Eğrisi  – FIGURE B
# ---------------------------------------------------------------
# Gerçek etiketleri binarize et (One-vs-Rest)
y_true_bin = label_binarize(y_true, classes=labels)

# Mikro ortalama ROC
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
roc_auc     = roc_auc_score(y_true_bin, y_proba, average="micro")

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr, tpr, linewidth=2,
            label=f"Micro-average AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("FIGURE 5. Micro-average ROC curve for Stage-2 classifier")
ax_roc.legend(loc="lower right")
plt.tight_layout()
fig_roc.savefig("stage2_roc.png", dpi=300)
plt.close(fig_roc)
print("✓ stage2_roc.png kaydedildi")
