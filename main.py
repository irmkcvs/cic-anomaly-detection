from utils.data_loader import load_cic_ids_data
from utils.preprocessing import preprocess_data
from utils.evaluation import plot_confusion

#from modules.decision_tree import train_decision_tree
#from modules.random_forest import train_random_forest
#from modules.logistic_regression import train_logistic_regression
#from modules.knn import train_knn
#from modules.svm import train_svm
from modules.xgboost_model import train_xgboost

def main():
    df = load_cic_ids_data()
    print("[DEBUG] Dataset satır sayısı:", df.shape)
    print("[DEBUG] Dataset sütunları:", df.columns.tolist())

    X_train, X_test, y_train, y_test = preprocess_data(df)

    #train_decision_tree(X_train, X_test, y_train, y_test)
    #train_random_forest(X_train, X_test, y_train, y_test)
    #train_logistic_regression(X_train, X_test, y_train, y_test)
    #train_knn(X_train, X_test, y_train, y_test)
    #train_svm(X_train, X_test, y_train, y_test)
    train_xgboost(X_train, X_test, y_train, y_test)
if __name__ == "__main__":

    main()

