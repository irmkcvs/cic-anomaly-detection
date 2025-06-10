from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.evaluation import plot_confusion

def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("[Random Forest] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    plot_confusion(y_test, y_pred, "Random Forest - Confusion Matrix", "conf_matrix_random_forest.png")
