import pandas as pd
from data_utils import preprocess_data
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# 1) CSV yolları
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE  = "UNSW_NB15_testing-set.csv"

# 2) Veri yükleme
df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

# 3) Ön işleme
X_train, y_train, X_test, y_test = preprocess_data(df_train, df_test)

# 4) RandomForest modeli
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)
auc_rf     = roc_auc_score(y_test, y_proba_rf, multi_class='ovr')
f1_rf      = f1_score(y_test, y_pred_rf, average='macro')

# 5) LightGBM modeli
lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')
lgbm.fit(X_train, y_train)
y_pred_lgbm  = lgbm.predict(X_test)
y_proba_lgbm = lgbm.predict_proba(X_test)
auc_lgbm     = roc_auc_score(y_test, y_proba_lgbm, multi_class='ovr')
f1_lgbm      = f1_score(y_test, y_pred_lgbm, average='macro')

# 6) Sonuçları yazdır
print(f"RandomForest — ROC-AUC: {auc_rf:.4f}, F1-score: {f1_rf:.4f}")
print(f"LightGBM     — ROC-AUC: {auc_lgbm:.4f}, F1-score: {f1_lgbm:.4f}")
