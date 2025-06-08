# model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(X_train, y_train, **kwargs):
    """
    RandomForest modelini eğitir.
    class_weight='balanced' varsayılan olarak kullanılır.
    kwargs ile hiperparametre özelleştirmeleri yapılabilir.
    """
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        **kwargs
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test, print_result=True):
    """
    Modeli test verisi üzerinde değerlendirir.
    Sonuçları dictionary olarak döner.
    """
    if y_test is None:
        if print_result:
            print("Test setinde 'label' sütunu yok. Değerlendirme yapılamıyor.")
        return None

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    if print_result:
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

    return {
        "accuracy": acc,
        "report": report
    }
