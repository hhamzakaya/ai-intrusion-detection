
import pandas as pd
import joblib
from tcp_listener import prepare_dataframe  # JSON → DataFrame helper

# ────────────────────────────────────────────────────────────────
# 1) Modeller ve özellik listeleri
# ────────────────────────────────────────────────────────────────
stage1 = joblib.load("stage1_binary_rf.joblib")
stage2 = joblib.load("models/stage2_collapsed_rf.joblib")

STAGE1_COLS = list(stage1.feature_names_in_)
STAGE2_COLS = list(stage2.feature_names_in_)
THRESHOLD   = 0.7   # 0-1 arası; fp oranını ayarlamak için gerekirse değiştirin
NEEDED = ["srcip","dstip","proto","service","state",
          "sport","dport","dur","sbytes","dbytes"]
# ────────────────────────────────────────────────────────────────
# 2) Test setinden 1 Normal + 1 Attack kayıt seç
# ────────────────────────────────────────────────────────────────
TEST_CSV = "UNSW_NB15_testing-set.csv"
test_df = pd.read_csv(TEST_CSV)

rec_normal = (test_df.loc[test_df["label"]==0, NEEDED]
                         .iloc[0].to_dict())
rec_attack = (test_df.loc[test_df["label"]!=0, NEEDED]
                         .iloc[0].to_dict())
true_cat    = rec_attack.get("attack_cat", "Unknown")

# ────────────────────────────────────────────────────────────────
# 3) Manuel kayıt (tamamen normal)
# ────────────────────────────────────────────────────────────────
rec_manual = {
    "srcip": "10.0.0.1", "dstip": "10.0.0.2",
    "proto": "TCP",      "service": "http",  "state": "SF",
    "sport": 12345,       "dport": 80,
    "dur": 0.5,           "sbytes": 150.0,     "dbytes": 100.0
}

records = [
    ("Gerçek NORMAL",   rec_normal,  "Normal"),
    ("Gerçek ATTACK",   rec_attack,  true_cat),
    ("Manuel kayıt #1", rec_manual,  None)
]

# ────────────────────────────────────────────────────────────────
# 4) Yardımcı fonksiyon: tek kaydı tahmin et
# ────────────────────────────────────────────────────────────────

def predict_flow(json_rec: dict):
    """ Tek bir JSON kaydı için Stage-1 olasılığı, karar ve (varsa) Stage-2 etiketi döndür. """
    df = prepare_dataframe(json_rec)

    # Stage-1 öznitelik hizalaması
    df1 = df.reindex(columns=STAGE1_COLS, fill_value=0)
    prob = stage1.predict_proba(df1)[0, 1]
    is_attack = prob > THRESHOLD

    if not is_attack:
        return prob, "Normal", None

    # Stage-2 öznitelik hizalaması
    df2 = df.reindex(columns=STAGE2_COLS, fill_value=0)
    attack_type = stage2.predict(df2)[0]
    return prob, "Attack", attack_type

# ────────────────────────────────────────────────────────────────
# 5) Test ve çıktı
# ────────────────────────────────────────────────────────────────

def main():
    for tag, rec, truth in records:
        prob, cls, cat = predict_flow(rec)
        print(f"\n--- {tag} ---")
        print(f"Stage-1 olasılık = {prob:.3f}  →  {cls}")
        if cls == "Attack":
            print(f"Stage-2 tahmin   = {cat}")
        if truth is not None:
            # Doğruluk göstergesi
            ok = (cls == "Normal" and truth == "Normal") or (cat == truth)
            print(f"Gerçek etiket    = {truth}  {'✅' if ok else '❌'}")

if __name__ == "__main__":
    main()
