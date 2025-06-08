# data_utils.py

import io
import ipaddress
import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder, StandardScaler



def load_csv_from_gcs(bucket_name, file_name):
    """
    GCS'den CSV dosyası indirip DataFrame döndürür.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    return df

def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0

def preprocess_data(train_df, test_df):
    """
    Yalnızca 37 özellik kullanılarak veri ön işleme yapılır:
      - Eksik sütunlar sıfırla doldurulur
      - IP'ler sayıya çevrilir
      - 'attack_cat' sadece eğitim verisinde anlamlı, test setinde sıfır olarak yer alır
      - Kategorik sütunlara LabelEncoder uygulanır
      - Sayısal sütunlar StandardScaler ile ölçeklenir
    """

    selected_features = [
        "srcip", "dstip", "proto", "service", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
        "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
        "res_bdy_len", "ct_state_ttl", "ct_flw_http_mthd",
        "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
        "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "attack_cat",  # yeniden dahil edildi
        "label"
    ]
    
        # kural-bazlı sütunları da modele sokmak için listeye ekleyelim
    rule_features = [
        "is_dos_like","is_backdoor_like","is_recon_like",
        "is_fuzzers_like","is_exploit_like","is_analysis_like"
    ]
    selected_features += rule_features
    # Eksik sütunları sıfırla doldur
    for df in [train_df, test_df]:
        for col in selected_features:
            if col not in df.columns:
                df[col] = 0
                    # Domain-bazlı (kural tabanlı) özellik mühendisliği
    for df in [train_df, test_df]:
        df["is_dos_like"] = (df["sbytes"] > 40000) & (df["dur"] < 2.0) & (df["sinpkt"] < 0.01)
        df["is_backdoor_like"] = (df["ct_src_dport_ltm"] > 0) & (df["is_ftp_login"] == 1) & (df["ct_ftp_cmd"] > 1)


        df["is_recon_like"] = (df["ct_dst_ltm"] > 7) & (df["ct_dst_sport_ltm"] > 4) & (df["dur"] < 0.2)
        df["is_fuzzers_like"] = (df["sjit"] > 0.7) & (df["djit"] > 0.7) & (df["sbytes"] > 20000)
        df["is_exploit_like"] = (df["trans_depth"] > 4) & (df["res_bdy_len"] > 1000) & (df["dbytes"] > 30000)
        df["is_analysis_like"] = (df["ct_flw_http_mthd"] > 0) & (df["trans_depth"] > 2) & (df["res_bdy_len"] > 5000)


    # Sadece bu sütunları kullan
    train_df = train_df[selected_features].copy()
    test_df = test_df[selected_features].copy()

    # IP adreslerini sayıya çevir
    for col in ["srcip", "dstip"]:
        train_df[col] = train_df[col].apply(ip_to_int)
        test_df[col] = test_df[col].apply(ip_to_int)

    # Kategorik sütunları tespit et
    cat_cols = [col for col in ["proto", "service", "state", "attack_cat"]
                if col in train_df.columns and train_df[col].dtype == "object"]

   
    # Encode işlemi için birleştir
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    train_size = train_df.shape[0]

    for col in cat_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))

    # Yeniden ayır
    train_encoded = combined_df.iloc[:train_size].copy()
    test_encoded = combined_df.iloc[train_size:].copy()

        # Etiket ve özellik ayrımı (attack_cat kullanılıyor)
    X_train = train_encoded.drop(columns=["attack_cat"], errors="ignore")
    y_train = train_encoded["attack_cat"] if "attack_cat" in train_encoded.columns else None
    X_test  = test_encoded .drop(columns=["attack_cat"], errors="ignore")
    y_test  = test_encoded ["attack_cat"] if "attack_cat" in test_encoded.columns else None


    # Sayısal sütunları ölçekle
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, y_train, X_test, y_test
