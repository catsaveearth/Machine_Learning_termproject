import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pyclustering.utils import timedcall;
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plts
from autoML import autoML


# set hyper-parameter
n_cluster = [2, 4, 8, 10, 13, 15]
DBSCAN_eps = [0.1, 0.2, 0.5, 0.7, 1]

# fill or drop na value
def checkNA(data):
    # drop unused data
    data = data.drop(['HCHK_YEAR', 'SIDO', 'TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE', 'HCHK_OE_INSPEC_YN', 'CRS_YN', 'TTR_YN', 'DATA_STD__DT'], axis=1)
    data = data.dropna(subset=['AGE_GROUP', 'HEIGHT', 'WEIGHT', 'WAIST'], axis=0)
    data = data.fillna(method='ffill')
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
        data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis=1)
        data = data.drop(column, axis=1)
    return data

def Preprocessing(feature, encode_list, scale_list):
    # feature : dataframe of feature

    #scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]    
    scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]

    # encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    #one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
    encoders_name = ["ordinal", "onehot"]

    # result box
    result_dictionary = {}
    i = 0

    if encode_list == []:
        for scaler in scalers:
            if i == 0: #not scaling
                result_dictionary[scalers_name[i]] = feature.copy()

            else:
                #===== scalers
                result_dictionary[scalers_name[i]] = feature.copy()
                result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(feature[scale_list]) #scaling
            i = i + 1
        return result_dictionary


    for scaler in scalers:
        if i == 0: #not scaling
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(feature[encode_list])
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"], encode_list)

        else:
            #===== scalers + ordinal encoding
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(feature[scale_list]) #scaling
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(feature[encode_list]) #encoding

            #===== scalers + OneHot encoding
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(feature[scale_list]) #scaling
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"], encode_list) #encoding

        i = i + 1

    return result_dictionary

def makeplot(title, value, x_list):
    plt.plot(x_list, value)
    plt.title(title)
    plt.show()


def clustering(data, target):
    # # k-mean
    # print("Kmeans")
    # kmeans_result, plist = autoML(1, data, target, KMeans, n_cluster)
    # makeplot("kmeans_autoML", kmeans_result, plist)

    # # EM
    # print("EM (GMM)")
    # EM_result, plist = autoML(1, data, target, GaussianMixture, n_cluster)
    # makeplot("EM_autoML", EM_result, plist)

    # DBSCAN
    print("dbscan")
    dbscan_result, plist = autoML(1, data, target, DBSCAN, DBSCAN_eps)
    makeplot("DBSCAN_autoML", dbscan_result, plist)


    # Spectral
    print("Spectral")
    spectral_result, plist = autoML(1, data, target, SpectralClustering, n_cluster)
    makeplot("spectral_autoML", spectral_result, plist)


def main():
    # data load and fill&drop
    data = read_data()
    
    # split feature and GT(age)
    age = pd.DataFrame(data["AGE_GROUP"])
    feature = data.drop(columns=["AGE_GROUP"])

    # preprocessing - scaling and encoding
    feature = Preprocessing(feature, ["SEX", "HEAR_LEFT", "HEAR_RIGHT", "OLIG_PROTE_CD", "SMK_STAT_TYPE_CD", "DRK_YN"], 
                            ["HEIGHT", "WEIGHT", "WAIST", "SIGHT_LEFT", "SIGHT_RIGHT", "BP_HIGH", "BP_LWST", "BLDS", "HMG", "CREATININE", "SGOT_AST", "SGPT_ALT", "GAMMA_GTP"])

    # clustering
    clustering(feature, age)

        

if __name__ == "__main__":
	main()