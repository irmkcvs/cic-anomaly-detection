from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from utils.evaluation import plot_confusion

def train_svm(X_train, X_test, y_train, y_test):
    print("[SVM] Model eğitiliyor... (Bu işlem biraz zaman alabilir ⏳)")
    model = SVC(kernel='rbf', random_state=42)  # RBF kernel = daha güçlü, doğrusal olmayan ilişkiler için iyi
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("[SVM] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    plot_confusion(y_test, y_pred, "SVM - Confusion Matrix", "conf_matrix_svm.png")
