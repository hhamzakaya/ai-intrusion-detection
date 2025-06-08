import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from data_utils import preprocess_data
from model import evaluate_model

#  Lokal CSV yolları
TRAIN_FILE = r"C:\Users\lenovo\Desktop\netSec_pr_component\UNSW_NB15_training-set.csv"
TEST_FILE  = r"C:\Users\lenovo\Desktop\netSec_pr_component\UNSW_NB15_testing-set.csv"

# 1️ CSV'leri yükle
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

# 2️ Ön işleme
X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

# 3️ Pipeline: önce oversampling, sonra RF
pipe = Pipeline([
    ("ros", RandomOverSampler(random_state=42)),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 4️ Hiperparametre ızgarası
param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=2
)

print("GridSearchCV başlıyor...")
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
best_model = grid.best_estimator_

# 5️ Test seti üzerinde değerlendir
print("Best model test seti üzerinde:")
evaluate_model(best_model, X_test, y_test)

# 6️ En iyi modeli kaydet
joblib.dump(best_model, "model_pipeline.joblib")
print("En iyi model kaydedildi: model_pipeline.joblib")
