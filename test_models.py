#!/usr/bin/env python3
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def load_and_prep(path):
    df = pd.read_csv(path)
    # 1) Normalize headers
    df.columns = df.columns.str.strip().str.lower()
    # 2) Binary label
    y_bin = (df["label"] != 0).astype(int)
    # 3) Multiclass cat (only for evaluation)
    y_cat = df["attack_cat"]
    # 4) Drop leakage cols
    X = df.drop(columns=["label","attack_cat"], errors="ignore")
    return X, y_bin, y_cat

def main():
    # 1) Model yükle
    stage1 = joblib.load("stage1_binary_pipe.joblib")
    stage2 = joblib.load("models/stage2_pipe.joblib")

    # 2) Test verisi
    X_test, y_bin_test, y_cat_test = load_and_prep("UNSW_NB15_testing-set.csv")

    # 3) Stage-1 tahminleri
    y_bin_pred = stage1.predict(X_test)
    print("=== Stage-1 (Binary) ===")
    print(classification_report(y_bin_test, y_bin_pred, target_names=["Normal","Attack"], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_bin_test, y_bin_pred))

    # 4) Stage-2 tahminleri (sadece attack örnekleri)
    mask_att = y_bin_pred == 1
    X_attack = X_test[mask_att]
    true_cat = y_cat_test[mask_att]

    if len(X_attack):
        y_cat_pred = stage2.predict(X_attack)
        print("\n=== Stage-2 (Multiclass) on Predicted-Attacks ===")
        print(classification_report(true_cat, y_cat_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(true_cat, y_cat_pred))
    else:
        print("\n(!) Stage-1 hiç saldırı tespit etmedi, Stage-2 çalıştırılmadı.")

if __name__ == "__main__":
    main()

