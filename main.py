##############################
### SETTING ENVIRONMENT
##############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from helpers import check_df, grab_col_names, correlation_matrix, plot_importance, crop_recommendation_data_prep, base_models, test_base_model, plot_importance
from config import lr_params, knn_params, cart_params, rf_params, lightgbm_params, classifiers, hyperparameter_optimization
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


################################################
# Pipeline Main Function
################################################


def main():
    df = pd.read_csv("Crop_recommendation.csv")
    X_train, X_test, y_train, y_test = crop_recommendation_data_prep(df)
    base_models(X_train, y_train)
    best_models = hyperparameter_optimization(X_train, y_train)
    voting_clf = voting_classifier(best_models, X_train, y_train)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

if __name__ == "__main__":
    print("THE PROCESS HAS STARTED")
    main()
