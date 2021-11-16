import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # 모든 열을 출력하도록

target_feature = 'AGE_GROUP'


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


def plotCurrentResult(score, title):
    plt.title(title)
    x_values = range(1, len(score) + 1)
    plt.xlabel('Parameter set')
    if "Decision Tree" in score['current_model'][0]:
        tempList = [_['max_depth'] for _ in score['params']]
        plt.xticks(x_values, tempList)
    elif "Support Vector Machine" in score['current_model'][0]:
        tempList = [[_['C'], _['gamma']] for _ in score['params']]
        plt.xticks(x_values, tempList)
    else:
        tempList = [_['n_neighbors'] for _ in score['params']]
        plt.xticks(x_values, tempList)

    plt.ylabel('mean score')
    y_values = score['mean_test_score'].tolist()
    plt.plot(x_values, y_values)
    plt.show()


def plotCurrentResult2(score, title):
    plt.title(title)
    x_values = range(1, len(score) + 1)
    xList = []
    yList = []
    plt.xlabel('Model states')
    plt.ylabel('best score')

    for _ in score:
        xList.append([_['best-model'], _['best-param']])
        yList.append(_['best-score'])
    plt.xticks(x_values, xList)

    plt.plot(x_values, yList)
    plt.show()


def classification(data, target):
    # temp best record variable
    best_answer = {
        'best-model': "",
        'best-param': "",
        'best-score': -1.0,
    }
    whole_score = pd.DataFrame()

    # split train / test dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # case 1: decision tree (entropy)
    classifier = DecisionTreeClassifier(criterion="entropy")
    params = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8]
    }
    grid_tree = GridSearchCV(classifier, param_grid=params, cv=3, refit=True)
    grid_tree.fit(X_train, y_train)
    bestModel = grid_tree.best_estimator_
    bestParam = grid_tree.best_params_
    bestScore = bestModel.score(X_test, y_test)
    if bestScore > best_answer['best-score']:
        best_answer = {
            'best-model': "Decision Tree (entropy)",
            'best-param': bestParam,
            'best-score': bestScore
        }
    scores = pd.DataFrame(grid_tree.cv_results_)[
        ['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]
    scores['current_model'] = "Decision Tree (entropy)"
    whole_score = pd.concat([whole_score, scores], ignore_index=True)
    plotCurrentResult(scores, "Decision Tree (entropy)")

    # case 2: decision tree (gini)
    classifier = DecisionTreeClassifier(criterion="gini")
    params = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8]
    }
    grid_tree = GridSearchCV(classifier, param_grid=params, cv=3, refit=True)
    grid_tree.fit(X_train, y_train)
    bestModel = grid_tree.best_estimator_
    bestParam = grid_tree.best_params_
    bestScore = bestModel.score(X_test, y_test)
    if bestScore > best_answer['best-score']:
        best_answer = {
            'best-model': "Decision Tree (gini)",
            'best-param': bestParam,
            'best-score': bestScore
        }
    scores = pd.DataFrame(grid_tree.cv_results_)[
        ['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]
    scores['current_model'] = "Decision Tree (gini)"
    whole_score = pd.concat([whole_score, scores], ignore_index=True)
    plotCurrentResult(scores, "Decision Tree (gini)")

    # case 3: svm
    classifier = SVC(kernel='rbf')
    params = {
        'C': [1, 5, 10],
        'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    }
    grid_tree = GridSearchCV(classifier, param_grid=params, cv=3, refit=True)
    grid_tree.fit(X_train, y_train)
    bestModel = grid_tree.best_estimator_
    bestParam = grid_tree.best_params_
    bestScore = bestModel.score(X_test, y_test)
    if bestScore > best_answer['best-score']:
        best_answer = {
            'best-model': "Support Vector Machine",
            'best-param': bestParam,
            'best-score': bestScore
        }
    scores = pd.DataFrame(grid_tree.cv_results_)[
        ['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score',
         'split2_test_score']]
    scores['current_model'] = "Support Vector Machine"
    whole_score = pd.concat([whole_score, scores], ignore_index=True)
    plotCurrentResult(scores, "Support Vector Machine")

    # case 4: K-Nearest Neighbors
    classifier = KNeighborsClassifier()
    params = {
        'n_neighbors': list(range(1, 6))
    }
    grid_tree = GridSearchCV(classifier, param_grid=params, cv=3, refit=True)
    grid_tree.fit(X_train, y_train)
    bestModel = grid_tree.best_estimator_
    bestParam = grid_tree.best_params_
    bestScore = bestModel.score(X_test, y_test)
    if bestScore > best_answer['best-score']:
        best_answer = {
            'best-model': "KNN",
            'best-param': bestParam,
            'best-score': bestScore
        }
    scores = pd.DataFrame(grid_tree.cv_results_)[
        ['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]
    scores['current_model'] = "KNN"
    whole_score = pd.concat([whole_score, scores], ignore_index=True)
    plotCurrentResult(scores, "KNN")

    print(whole_score)

    return best_answer


def main():
    data = read_data()
    data = Preprocessing(data, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])
    feature_cases_ordinal = [
        data['original_ordinal'].columns.values,
        ['SEX', 'AGE_GROUP', 'BLDS', 'HMG', 'OLIG_PROTE_CD', 'CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP',
         'SMK_STAT_TYPE_CD', 'DRK_YN'],
        ['SEX', 'AGE_GROUP', 'HEIGHT', 'WEIGHT', 'WAIST', 'SIGHT_LEFT', 'SIGHT_RIGHT', 'HEAR_LEFT', 'HEAR_RIGHT',
         'BP_HIGH', 'BP_LWST', 'OLIG_PROTE_CD', 'SMK_STAT_TYPE_CD', 'DRK_YN']
    ]
    feature_cases_onehot = [
        data['original_onehot'].columns.values,
        ['SEX_1.0', 'SEX_2.0', 'AGE_GROUP', 'BLDS', 'HMG', 'OLIG_PROTE_CD', 'CREATININE', 'SGOT_AST', 'SGPT_ALT',
         'GAMMA_GTP', 'SMK_STAT_TYPE_CD', 'DRK_YN'],
        ['SEX_1.0', 'SEX_2.0', 'AGE_GROUP', 'HEIGHT', 'WEIGHT', 'WAIST', 'SIGHT_LEFT', 'SIGHT_RIGHT', 'HEAR_LEFT',
         'HEAR_RIGHT', 'BP_HIGH', 'BP_LWST', 'OLIG_PROTE_CD', 'SMK_STAT_TYPE_CD', 'DRK_YN']
    ]

    # best answer template
    best_answer = {
        'best-encoding+scaling': "",
        'best-feature-list': "",
        'best-model': "",
        'best-param': "",
        'best-score': -1.0,
    }

    for key, value in data.items():  # for all encoding + scaling dataset:
        temp_answer = {
            'best-encoding+scaling': "",
            'best-feature-list': "",
            'best-model': "",
            'best-param': "",
            'best-score': -1.0,
        }
        tempList = []

        # divide case ordinal / onehot
        if 'ordinal' in key:
            for feature_list in feature_cases_ordinal:  # for all feature test set:
                using_dataset = value[feature_list]
                data_y = using_dataset[target_feature]
                data_x = using_dataset.drop(columns=[target_feature])

                temp = classification(data_x, data_y)  # classification start
                tempList.append(temp)
                if temp_answer['best-score'] < temp['best-score']:
                    temp_answer['best-encoding+scaling'] = key
                    temp_answer['best-feature-list'] = feature_list
                    temp_answer['best-model'] = temp['best-model']
                    temp_answer['best-param'] = temp['best-param']
                    temp_answer['best-score'] = temp['best-score']
        else:
            for feature_list in feature_cases_onehot:  # for all feature test set:
                using_dataset = value[feature_list]
                data_y = using_dataset[target_feature]
                data_x = using_dataset.drop(columns=[target_feature])

                temp = classification(data_x, data_y)  # classification start
                tempList.append(temp)
                if temp_answer['best-score'] < temp['best-score']:
                    temp_answer['best-encoding+scaling'] = key
                    temp_answer['best-feature-list'] = feature_list
                    temp_answer['best-model'] = temp['best-model']
                    temp_answer['best-param'] = temp['best-param']
                    temp_answer['best-score'] = temp['best-score']

        # plot current best result with this data ( encoding + scaling )
        plotCurrentResult2(tempList, key)

    print(best_answer)
    return


if __name__ == "__main__":
    main()