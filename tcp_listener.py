
import socket
import json
import csv
from datetime import datetime
from collections import deque
import pandas as pd
import ipaddress
import joblib

# -------------------------------------------------------------------
# Ayarlar
# -------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 5050

# Sliding‐window anomaly detection
WINDOW_SIZE  = 500
THRESHOLD    = 500.0
window       = deque(maxlen=WINDOW_SIZE)
total_sbytes = 0

# Her IP için flow sayacı
ip_flow_counts = {}

# Stage-1 ve Stage-2 modellerini yükle
stage1_pipe = joblib.load("stage1_binary_pipe.joblib")
stage2_pipe = joblib.load("models/stage2_pipe.joblib")


# Kategorik mapping’ler
PROTO_MAP   = {"TCP":0,"UDP":1,"ICMP":2}
SERVICE_MAP = {"-":0}
STATE_MAP   = {"SF":0,"S0":1,"REJ":2,"RSTR":3,"SH":4}

# -------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# -------------------------------------------------------------------
def ip_to_int(ip_str: str) -> int:
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0

def get_cidr(ip_str: str, prefix: int = 24) -> str:
    from ipaddress import ip_network
    net = ip_network(f"{ip_str}/{prefix}", strict=False)
    return f"{net.network_address}/{prefix}"

def check_anomaly(rec: dict):
    global total_sbytes
    # Sliding‐window average
    if len(window) == WINDOW_SIZE:
        total_sbytes -= window[0]["sbytes"]
    window.append(rec)
    total_sbytes += rec["sbytes"]
    avg = total_sbytes / len(window)
    if avg > THRESHOLD:
        print(f" Anomali tespit: Ortalama sbytes = {avg:.2f}")

# -------------------------------------------------------------------
# Bağlantı İşleyicisi
# -------------------------------------------------------------------
def handle_connection(conn: socket.socket, addr):
    print(f" Bağlantı: {addr}")
    buffer = ""

    while True:
        chunk = conn.recv(8192).decode("utf-8", errors="ignore")
        if not chunk:
            break
        buffer += chunk
        lines = buffer.split("\n")

        for line in lines[:-1]:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(" JSON parse hatası:", e)
                continue

            # Zorunlu alanlar
            obj.setdefault("service", "-")
            obj.setdefault("state", "-")

            # Anomali kontrolü
            try:
                check_anomaly(obj)
            except Exception as e:
                print(" Anomali kontrol hatası:", e)

            # Model için DataFrame
            df_raw = pd.DataFrame([obj])
         # sütun adlarını normalize et (strip+lower)
            df_raw.columns = df_raw.columns.str.strip().str.lower()

            # --- Stage-1: Normal vs Attack ---
            try:
                prob_attack = stage1_pipe.predict_proba(df_raw)[0,1]
                is_attack  = prob_attack > 0.6   # Eşik: 0.6
            except Exception as e:
                print(" Stage-1 atlandı:", e)
                continue

            if not is_attack:
                print(f" Normal: srcip={obj.get('srcip')}")
                print("                                                           ")
                continue

            # --- Stage-2: Saldırı Türü Tahmini ---
            try:
                # Model.predict doğrudan string etiket döndürür
                att_label = stage2_pipe.predict(df_raw)[0]
                if att_label in ["-", "", None]:
                 att_label = "Normal"

            except Exception as e:
                print(" Stage-2 atlandı:", e)
                continue


            # Log ve bildirim
            
            print(f" Saldırı Türü: {att_label} | srcip={obj.get('srcip')}")
            
            with open(f"alerts.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if f.tell()==0:
                    w.writerow(["timestamp","srcip","state","attack_cat","prob_attack"])
                w.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    obj.get("srcip","-"),
                    obj.get("state","-"),
                    att_label,
                    f"{prob_attack:.3f}"
                ])

        buffer = lines[-1]

    conn.close()
    print(f" Bağlantı kapandı: {addr}")

# -------------------------------------------------------------------
# Sunucu Döngüsü
# -------------------------------------------------------------------
def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f" TCP dinleyici {HOST}:{PORT} – beklemede… (Ctrl-C ile çık)")

        while True:
            try:
                conn, addr = server.accept()
                handle_connection(conn, addr)
            except KeyboardInterrupt:
                print(" Dinleyici kapatılıyor…")
                break
            except Exception as e:
                print(" Sunucu hatası:", e)

if __name__ == "__main__":
    start_server()
