
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import lightgbm as lgb
from data_utils import preprocess_data

def main():
    train_df = pd.read_csv("UNSW_NB15_training-set.csv")
    test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")

    
    X_full, y_full, X_test, y_test = preprocess_data(train_df, test_df)

   
    le = LabelEncoder()
    y_full_enc = le.fit_transform(y_full)
    y_test_enc = le.transform(y_test)

   
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full_enc,
        stratify=y_full_enc,
        test_size=0.2,
        random_state=42
    )

    clf = lgb.LGBMClassifier(
        objective='multiclass',               
        num_class=len(le.classes_),
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=200,
        min_child_samples=20,
        feature_fraction=0.8,
        class_weight='balanced',          
        random_state=42
    )

    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0)
        ]
    )

  
    y_proba = clf.predict_proba(X_test)        
    roc     = roc_auc_score(y_test_enc, y_proba, multi_class='ovr')
    y_pred  = clf.predict(X_test)
    f1      = f1_score(y_test_enc, y_pred, average='macro')
    y_pred_labels = le.inverse_transform(y_pred)

    print(f"\n🏁 Test ROC-AUC:      {roc:.4f}")
    print(f"🏁 Test F1-score (macro): {f1:.4f}\n")
    print(classification_report(
        y_test,
        y_pred_labels,
        digits=4,
        zero_division=0  
    ))

if __name__ == "__main__":
    main()
