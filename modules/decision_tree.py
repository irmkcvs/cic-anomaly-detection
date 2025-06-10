from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils.evaluation import plot_confusion

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("[Decision Tree] Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_confusion(y_test, y_pred, "Decision Tree - Confusion Matrix", "conf_matrix_decision_tree.png")