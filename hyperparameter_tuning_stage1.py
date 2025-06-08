#!/usr/bin/env python3
"""
hyperparameter_tuning_stage1.py

Stage-1: Normal vs Attack (binary) classifier için hiperparametre araması.
• UNSW-NB15 CSV’lerinden tüm sütunlar strip+lower.
• label & attack_cat sütunları X’ten çıkarılır (sızıntı yok).
• Dinamik sayısal/kategorik ayrımıyla ColumnTransformer oluşturulur.
• Tek Pipeline (preproc + RF) üzerinden RandomizedSearchCV.
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble     import RandomForestClassifier
from sklearn.pipeline     import Pipeline
from sklearn.compose      import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics      import classification_report, confusion_matrix

def main():
    # 1) CSV’leri yükle ve sütun adlarını normalize et
    train_df = pd.read_csv("UNSW_NB15_training-set.csv")
    test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")
    for df in (train_df, test_df):
        df.columns = df.columns.str.strip().str.lower()

    # 2) Hedef değişken: label != 0 → Attack(1), Normal(0)
    y_train = (train_df["label"] != 0).astype(int)
    y_test  = (test_df ["label"] != 0).astype(int)

    # 3) Özellik matrisini al ve sızıntıyı önle
    X_train = train_df.drop(columns=["label","attack_cat"], errors="ignore")
    X_test  = test_df .drop(columns=["label","attack_cat"], errors="ignore")

    # 4) Eğitim/validasyon bölmesi
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # 5) Dinamik sayısal/kategorik sütun listesi
    num_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=["object","category","bool"]).columns.tolist()
    print(f" Sayısal sütun: {num_cols}")
    print(f" Kategorik sütun: {cat_cols}")

    # 6) ColumnTransformer + Pipeline
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    pipe = Pipeline([
        ("pre", preproc),
        ("clf", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # 7) Hiperparametre araması (Attack recall maksimize)
    param_dist = {
        "clf__n_estimators":      [50, 100, 200],
        "clf__max_depth":         [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
    }
    rs = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        scoring="recall",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score="raise"
    )
    print(" Hiperparametre araması başlatılıyor (Attack recall)…")
    rs.fit(X_tr, y_tr)
    best = rs.best_estimator_
    print("✓ En iyi parametreler:", rs.best_params_)

    # 8) Validation & Test değerlendirme
    print("\n Validation Set:")
    yv_pred = best.predict(X_val)
    print(classification_report(y_val, yv_pred, target_names=["Normal","Attack"], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, yv_pred))

    print("\n Test Set:")
    yt_pred = best.predict(X_test)
    print(classification_report(y_test, yt_pred, target_names=["Normal","Attack"], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, yt_pred))

    # 9) Modeli kaydet
    joblib.dump(best, "stage1_binary_pipe.joblib")
    print("\n Kaydedildi: stage1_binary_pipe.joblib")

if __name__ == "__main__":
    main()
