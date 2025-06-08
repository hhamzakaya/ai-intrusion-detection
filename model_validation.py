import joblib
import pandas as pd
import json

# 1) Binary modeli yükleyin (yolu kendi dosyanıza göre güncelleyin)
binary_pipe = joblib.load("stage1_binary_pipe.joblib")

# 2) Modelin beklediği özellik adlarını alın
features = list(binary_pipe.feature_names_in_)

# 3) JSON’dan DataFrame’e dönüştürüp eksikleri 0 ile dolduran yardımcı
def prepare_df(sample: dict) -> pd.DataFrame:
    # Tüm feature’ları 0 ile başlat
    base = {f: 0 for f in features}
    # sample’daki değerlerle güncelle
    base.update(sample)
    df = pd.DataFrame([base])
    return df

# 4) Saldırı/normal tahmini yapan fonksiyon
def predict_attack(sample: dict) -> str:
    df = prepare_df(sample)
    pred = binary_pipe.predict(df)[0]
    return "Saldırı" if pred == 1 else "Normal"

# 5) Gerçek etiket ile karşılaştıran fonksiyon
def predict_and_check(sample: dict):
    # JSON kaydı yazdır
    print("\nJSON Girdisi:", json.dumps(sample, indent=2, ensure_ascii=False))
    # Model tahmini
    pred = predict_attack(sample)
    # Gerçek etiket (sample içinde label yoksa 0=Normal kabul ederiz)
    true = "Saldırı" if sample.get("label", 0) == 1 else "Normal"
    # Sonuç
    print(f"Gerçek: {true}, Model: {pred} →", "✅ Doğru" if pred == true else "❌ Yanlış")

# 6) Örnek senaryolar
normal_sample = {
    "id": 1,
    "dur": 0.05,
    "spkts": 10, "dpkts": 8,
    "sbytes": 200, "dbytes": 180,
    "rate": 4000,
    "sttl": 64, "dttl": 64,
    "swin": 8192, "dwin": 65535,
    "proto": "tcp", "service": "http", "state": "FIN",
    # diğer sayısal ct_…, jit, loss vb. otomatik 0 olacak
    "label": 0
}

attack_sample = {
    "id": 2,
    "dur": 5.0,
    "spkts": 20000, "dpkts": 20000,
    "sbytes": 500000, "dbytes": 500000,
    "rate": 200000,
    "sttl": 255, "dttl": 255,
    "swin": 65535, "dwin": 65535,
    "proto": "tcp", "service": "http", "state": "REJ",
    # diğer sayısal ct_…, jit, loss vb. otomatik 0 olacak
    "label": 1
}

# 7) Testleri çalıştır
if __name__ == "__main__":
    predict_and_check(normal_sample)
    predict_and_check(attack_sample)
