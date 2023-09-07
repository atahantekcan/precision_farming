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


#EDA

def check_df(dataframe, head=5):   #Veri setinin genel görünümüne bakış atmak için kullanılabilecek fonksiyon.
    """
    Veri setine genel bakış sağlayan fonksiyon.

    :param dataframe: pd.dataframe
    incelenecek dataframe'in adı girilir.
    :param head: int.
    İlk kaç gözlemin incelenmek istediğine dair sayı girilir.
    :return: None
    """
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Duplicated #####################")
    print(dataframe.duplicated().sum())
    print("##################### Describe #####################")
    print(dataframe.describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


#PLOTS

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

################
#PRE-PROCESSING
################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def crop_recommendation_data_prep(dataframe):

    #1: RAINFALL*PH

    dataframe["NEW_RAINFALL_PH"] = dataframe["rainfall"]*dataframe["ph"]

    ##2: NUTRIENT BALANCE INDEX (NBI)

    dataframe["N/K"]= dataframe["N"] / dataframe["K"]
    dataframe["N/P"] = dataframe["N"] / dataframe["P"]
    dataframe["P/K"] = dataframe["P"] / dataframe["K"]

    ratios = ["N/K","N/P","P/K"]
    dataframe[ratios] = MinMaxScaler().fit_transform(dataframe[ratios])

    dataframe["NEW_NBI"] = np.cbrt(dataframe["N/K"] * dataframe["N/P"] * dataframe["P/K"]) ###kök 3 al.
    dataframe.drop(ratios, axis=1, inplace=True)


    ##3: CATEGORIZE BY PH LEVELS

    dataframe.loc[dataframe["ph"] < 7, "NEW_PH_CAT"] = "acidic"
    dataframe.loc[dataframe["ph"] > 7, "NEW_PH_CAT"] = "alkaline"
    dataframe.loc[dataframe["ph"] == 7, "NEW_PH_CAT"] = "neutral"
    dataframe.loc[(dataframe["ph"] >= 5.5) & (dataframe["ph"] <= 6.5), "NEW_PH_CAT"] = "optimal"

    ##4: CATEGORIZE BY RAINFALL

    #sns.catplot(data=df, x='label', y="rainfall", kind='box', height=10, aspect=22/8)
    #plt.title(f"{col}", size=12)
    #plt.xticks(rotation='vertical')
    #plt.show()

    #df["rainfall"].describe([0.10,0.20,0.30,0.33,0.40,0.50,0.60,0.66,0.70,0.75,0.80,0.90,0.99])

    dataframe["NEW_RAINFALL_CAT"] = pd.cut(x=dataframe["rainfall"], bins=[0, 44, 71, 111, 188, 300], labels=["extreme_low", "low", "medium", "high", "extreme_high"])

    ##5. ONE HOT ENCODER
    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dff = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
        return dff

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dff = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(dff)

    X_scaled = StandardScaler().fit_transform(dff[num_cols])
    dff[num_cols] = pd.DataFrame(X_scaled, columns=dff[num_cols].columns)

    dff = dff.sample(frac = 1)

    y = dff["label"]
    X = dff.drop(["label"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    return X_train, X_test, y_train, y_test
################
#BASE MODELS
################

def base_models(X, y, scoring="accuracy", cv=3, Test= False):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression(max_iter=1000)),
                       ('KNN', KNeighborsClassifier()),
                       ("SVC", SVC()),
                       ("CART", DecisionTreeClassifier()),
                       ("RF", RandomForestClassifier()),
                       ('GBM', GradientBoostingClassifier()),
                       ('LightGBM', LGBMClassifier())]
    
    for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

    if Test:
        test_base_model(X_test, y_test, classifiers, scoring=scoring)
def test_base_model(X, y, classifiers, scoring="accuracy", cv=3):
                for name, classifier in classifiers:
                    cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
                    print(f"Base_Model_Test_{scoring}_Result: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


################
#MODEL EVALUATION
################
def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


################
#STACKING & ENSEMBLE LEARNING
################

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LR', best_models["LR"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1_macro", "roc_auc_ovr"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1_macro'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc_ovr'].mean()}")
    return voting_clf