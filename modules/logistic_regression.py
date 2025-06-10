from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils.evaluation import plot_confusion

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("[Logistic Regression] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    plot_confusion(y_test, y_pred, "Logistic Regression - Confusion Matrix", "conf_matrix_logistic_regression.png")
#max_iter=1000 yazdım çünkü Logistic Regression büyük veride bazen convergence (yakınsama) hatası veriyor. Bunu önlemek için iterasyon sayısını artırıyoruz.
#n_jobs=-1 yazınca da tüm CPU çekirdeklerini kullanıyor = daha hızlı çalışıyor.