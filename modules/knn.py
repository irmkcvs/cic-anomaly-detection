from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.evaluation import plot_confusion

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("[KNN] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    plot_confusion(y_test, y_pred, "KNN - Confusion Matrix", "conf_matrix_knn.png")
