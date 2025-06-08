# preprocessing.py
import ipaddress, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["sport", "dport", "dur", "sbytes", "dbytes"]
CAT_COLS = ["proto", "service", "state", "src_cidr", "dst_cidr"]

def _to_cidr24(ip):
    try:
        net = ipaddress.ip_network(f"{ip}/24", strict=False)
        return f"{net.network_address}/24"
    except Exception:
        return "0.0.0.0/24"

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    if "src_cidr" not in df.columns:
        df["src_cidr"] = df["srcip"].map(_to_cidr24)
    if "dst_cidr" not in df.columns:
        df["dst_cidr"] = df["dstip"].map(_to_cidr24)
    return df

def build_preproc():
    """ColumnTransformer (sayısal+categorical)"""
    num_pipe = Pipeline([("sc", StandardScaler())])
    cat_pipe = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", num_pipe, [c for c in NUM_COLS]),
        ("cat", cat_pipe, CAT_COLS)
    ])
