#!/usr/bin/env python3
"""
plot_stage1_metrics.py
Stage-1 Random-Forest (binary) modeli için:
  – Karışıklık matrisi (Confusion Matrix)
  – ROC eğrisi (Receiver-Operating Characteristic)
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# -------------------------------------------------------------------
# 1) MODELLER VE TEST VERİSİ
# -------------------------------------------------------------------
MODEL_FILE = "stage1_binary_pipe.joblib"           # artefakt
TEST_FILE  = "UNSW_NB15_testing-set.csv"           # CSV yolu

print("› Model yükleniyor…")
model = joblib.load(MODEL_FILE)

print("› Test seti yükleniyor…")
df_test = pd.read_csv(TEST_FILE)
df_test.columns = df_test.columns.str.strip().str.lower()

# Hedef ve özellik ayır
y_true = (df_test["label"] != 0).astype(int)           # 0=Normal, 1=Attack
X_test = df_test.drop(columns=["label", "attack_cat"], errors="ignore")

# -------------------------------------------------------------------
# 2) TAHMİN
# -------------------------------------------------------------------
print("› Tahmin yapılıyor…")
y_proba = model.predict_proba(X_test)[:, 1]            # Attack olasılığı
y_pred  = (y_proba >= 0.60).astype(int)                # Aynı eşik

# -------------------------------------------------------------------
# 3) KARŞIKLIK MATRİSİ – FIGURE X
# -------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax_cm, colorbar=True)
ax_cm.set_title("FIGURE 3. Confusion matrix of the Stage-1 binary classifier")
plt.tight_layout()
fig_cm.savefig("stage1_confusion.png", dpi=300)
plt.close(fig_cm)
print("✓ stage1_confusion.png kaydedildi")

# -------------------------------------------------------------------
# 4) ROC EĞRİSİ – FIGURE Y
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc     = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("FIGURE 2. ROC curve of the Stage-1 binary classifier")
ax_roc.legend(loc="lower right")
plt.tight_layout()
fig_roc.savefig("stage1_roc.png", dpi=300)
plt.close(fig_roc)
print("✓ stage1_roc.png kaydedildi")
