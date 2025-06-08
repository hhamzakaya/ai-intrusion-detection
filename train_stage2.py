
import joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------------
# 1) Veri yükle & temel temizlik
# ----------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # sütun adlarını trim + lower
    df.columns = df.columns.str.strip().str.lower()
    return df

train_df = load_dataset("UNSW_NB15_training-set.csv")
test_df  = load_dataset("UNSW_NB15_testing-set.csv")

# ----------------------------------------------------------------------------
# 2) Yalnız saldırı kayıtları + 5 ana tür + Other
# ----------------------------------------------------------------------------
train_att = train_df[train_df["label"] != 0].reset_index(drop=True)
test_att  = test_df [test_df ["label"] != 0].reset_index(drop=True)


top5 = train_att["attack_cat"].value_counts().nlargest(5).index.tolist()
print(" 5 ana tür:", top5)

#  collapse adımı – nadir kategorileri “Other” altında topla
train_att["attack_collapsed"] = train_att["attack_cat"].where(
    train_att["attack_cat"].isin(top5), other="Other")
test_att ["attack_collapsed"] = test_att ["attack_cat"].where(
    test_att ["attack_cat"].isin(top5), other="Other")

#  Modelin hedefi olarak bu yeni sütunu kullan
train_att["y"] = train_att["attack_collapsed"]
test_att ["y"] = test_att ["attack_collapsed"]
# ----------------------------------------------------------------------------
# 3) Özellik / hedef ayır
# ----------------------------------------------------------------------------
X_train_raw = train_att.drop(columns=["label","attack_cat","attack_collapsed","y"])
y_train     = train_att["y"]
X_test_raw  = test_att.drop(columns=["label","attack_cat","attack_collapsed","y"])
y_test      = test_att["y"]

# otomatik sayısal / kategorik
num_cols = X_train_raw.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_train_raw.select_dtypes(include=["object","bool"]).columns.tolist()
print(f" {len(num_cols)} sayısal, {len(cat_cols)} kategorik sütun kullanılacak")

preproc = ColumnTransformer([
    ("num",  StandardScaler(), num_cols),
    ("cat",  OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([
    ("pre", preproc),
    ("clf", RandomForestClassifier(random_state=42,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1))
])

param_dist = {
    "clf__n_estimators":      [100, 200, 300],
    "clf__max_depth":         [None, 15, 25],
    "clf__min_samples_split": [2, 5, 10],
}

rs = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=3,
                        scoring="f1_macro", random_state=42,
                        verbose=2, n_jobs=-1, error_score="raise")
print(" Hiperparametre araması…")
rs.fit(X_train_raw, y_train)
print("✓ En iyi:", rs.best_params_)
model = rs.best_estimator_

# ----------------------------------------------------------------------------
# 4) Değerlendirme
# ----------------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(X_train_raw, y_train,
                                            test_size=0.2, stratify=y_train,
                                            random_state=42)
model.fit(X_tr, y_tr)
print("\n Validation:")
print(classification_report(y_val, model.predict(X_val)))
print("\n Test:")
print(classification_report(y_test, model.predict(X_test_raw)))

joblib.dump(model, "models/stage2_pipe.joblib")
print("\n Kaydedildi: models/stage2_pipe.joblib")
