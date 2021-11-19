import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from autoML import autoML

pd.set_option('display.max_columns', None)  # 모든 열을 출력하도록
target_feature = 'AGE_GROUP'

# set hyper-parameter
max_depth = {"max_depth" : [2, 3, 4, 5, 6, 7, 8]}
gamma = {"gamma" : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
n_neighbors = {"n_neighbors" : list(range(1, 6))}


# fill or drop na value
def checkNA(data):
    # drop unused data
    data = data.drop(
        ['HCHK_YEAR', 'SIDO', 'TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE', 'HCHK_OE_INSPEC_YN', 'CRS_YN',
         'TTR_YN', 'DATA_STD__DT'], axis=1)
    data = data.dropna(subset=['AGE_GROUP', 'HEIGHT', 'WEIGHT', 'WAIST'], axis=0)
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    return data

# read data from csv file
def read_data():
    data = pd.read_csv("health_checkup.csv")
    data = checkNA(data)
    new_age_group = {
        5.0: 2030,
        6.0: 2030,
        7.0: 2030,
        8.0: 2030,
        9.0: 4050,
        10.0: 4050,
        11.0: 4050,
        12.0: 4050,
        13.0: 6070,
        14.0: 6070,
        15.0: 6070,
        16.0: 6070,
        17.0: 80,
        18.0: 80
    }
    data['AGE_GROUP'] = data['AGE_GROUP'].apply(lambda x: new_age_group[x])
    return data

# for one-hot-encoding
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


def Preprocessing(feature, encode_list, scale_list):
    # feature : dataframe of feature

    # scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]

    # encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    # one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
    encoders_name = ["ordinal", "onehot"]

    # result box
    result_dictionary = {}
    i = 0

    if encode_list == []:
        for scaler in scalers:
            if i == 0:  # not scaling
                result_dictionary[scalers_name[i]] = feature.copy()

            else:
                # ===== scalers
                result_dictionary[scalers_name[i]] = feature.copy()
                result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(feature[scale_list])  # scaling
            i = i + 1
        return result_dictionary

    for scaler in scalers:
        if i == 0:  # not scaling
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
                feature[encode_list])
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
                                                                        encode_list)

        else:
            # ===== scalers + ordinal encoding
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(
                feature[scale_list])  # scaling
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
                feature[encode_list])  # encoding

            # ===== scalers + OneHot encoding
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(
                feature[scale_list])  # scaling
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"],
                                                                        encode_list)  # encoding

        i = i + 1

    return result_dictionary

def makeplot(title, value, x_list):
    plt.plot(x_list, value)
    plt.title(title)
    plt.show()


def classification(data, target):
    # decision tree (entropy)
    print("decision tree (entropy)")
    classifier = DecisionTreeClassifier(criterion='entropy')
    dt_ent_result, plist = autoML(0, data, target, classifier, max_depth, 1)
    makeplot("decision tree (entropy)_autoML", dt_ent_result, plist)


    # decision tree (gini)
    print("decision tree (gini)")
    classifier = DecisionTreeClassifier(criterion='gini')
    dt_gini_result, plist = autoML(0, data, target, classifier, max_depth, 1)
    makeplot("decision tree (gini)_autoML", dt_gini_result, plist)


    # svm
    print("support vector machine")
    classifier = SVC(kernel='rbf')
    svm_result, plist = autoML(0, data, target, classifier, gamma, 1)
    makeplot("SVM_autoML", svm_result, plist)

    # K-Nearest Neighbors
    print("K-Nearest Neighbors")
    classifier = KNeighborsClassifier()
    knn_result, plist = autoML(0, data, target, classifier, n_neighbors, 1)
    makeplot("knn_autoML", knn_result, plist)


def main():
    data = read_data()
        
    age = pd.DataFrame(data["AGE_GROUP"])
    feature = data.drop(columns=["AGE_GROUP"])

    data = Preprocessing(feature, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])

    feature_cases_ordinal = [
        data['original_ordinal'].columns.values,
        ['SEX', 'AGE_GROUP', 'BLDS', 'HMG', 'OLIG_PROTE_CD', 'CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP',
         'SMK_STAT_TYPE_CD', 'DRK_YN'],
        ['SEX', 'AGE_GROUP', 'HEIGHT', 'WEIGHT', 'WAIST', 'SIGHT_LEFT', 'SIGHT_RIGHT', 'HEAR_LEFT', 'HEAR_RIGHT',
         'BP_HIGH', 'BP_LWST', 'OLIG_PROTE_CD', 'SMK_STAT_TYPE_CD', 'DRK_YN']
    ]


    # for feature_list in feature_cases_ordinal:  # for all feature test set:
    #     using_dataset = data[feature_list]
    # data_x = using_dataset.drop(columns=[target_feature])

    classification(data, age)  # classification start



if __name__ == "__main__":
    main()