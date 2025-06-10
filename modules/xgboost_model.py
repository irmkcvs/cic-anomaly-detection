from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.evaluation import plot_confusion

def train_xgboost(X_train, X_test, y_train, y_test):
    print("[XGBoost] Model eğitiliyor...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("[XGBoost] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_confusion(y_test, y_pred, "XGBoost - Confusion Matrix", "conf_matrix_xgboost.png")
