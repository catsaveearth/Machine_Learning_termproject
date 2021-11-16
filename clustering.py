import pandas as pd
import numpy as np
import pprint
import matplotlib.pylab as plt
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pyclustering.utils import timedcall;
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#sum of distance for elbow method
kmeans_sumofDistance = {}
kmeans_sumofDistance_original = {}

#silhouette
kmeans_silhouette = {}
gmm_silhouette = {}
dbscan_silhouette = {}
Spectral_silhouette = {}

#purity
kmeans_purity = {}
gmm_purity = {}
dbscan_purity = {}
Spectral_purity = {}

def main():
    # data load and fill&drop
    data = read_data()
    
    # split feature and GT(age)
    age = pd.DataFrame(data["AGE_GROUP"])
    feature = data.drop(columns=["AGE_GROUP"])

    # preprocessing - scaling and encoding
    feature = Preprocessing(feature, ["SEX", "HEAR_LEFT", "HEAR_RIGHT", "OLIG_PROTE_CD", "SMK_STAT_TYPE_CD", "DRK_YN"], 
                            ["HEIGHT", "WEIGHT", "WAIST", "SIGHT_LEFT", "SIGHT_RIGHT", "BP_HIGH", "BP_LWST", "BLDS", "HMG", "CREATININE", "SGOT_AST", "SGPT_ALT", "GAMMA_GTP"])

    # set hyper-parameter
    n_cluster = [2, 4, 8, 10, 13, 15]
    DBSCAN_eps = [0.1, 0.2, 0.5, 0.7, 1]

    for key, value in feature.items():
        start_time = time.time()

        # if "minmax_ordinal" not in key:
        #     continue
        # n_cluster = [4]
        # DBSCAN_eps = [0.2]

        pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
        printcipalComponents = pca.fit_transform(value)
        principalDf = pd.DataFrame(data=printcipalComponents, columns = ['PCA1', 'PCA2'])

        plt.title(key)
        plt.scatter(principalDf["PCA1"], principalDf["PCA2"])
        plt.show()

        # # k-mean
        # print("Kmeans")
        # kmean_sum_of_squared_distances = []
        # kmean_silhouette_sub = []
        # kmeans_purity_sub = []
        # for k in n_cluster:
        #     kmeans = KMeans(n_clusters=k).fit(principalDf)
        #     plt.title("K-menas_15_standard_ordinal")
        #     plt.scatter(principalDf["PCA1"], principalDf["PCA2"], c=kmeans.labels_)
        #     plt.show()

            # # sum of distance for elbow methods
            # kmean_sum_of_squared_distances.append(kmeans.inertia_)

            # # silhouette (range -1~1)
            # kmean_silhouette_sub.append(silhouette_score(principalDf, kmeans.labels_, metric='euclidean'))

            # # purity
            # kmeans_purity_sub.append(purity_score(age, kmeans.labels_))
            # print(time.time() - start_time)

        # if "original" in key.split("_"):
        #     kmeans_sumofDistance_original[key] = kmean_sum_of_squared_distances
        # else:
        #     kmeans_sumofDistance[key] = kmean_sum_of_squared_distances

        # kmeans_silhouette[key] = kmean_silhouette_sub
        # kmeans_purity[key] = kmeans_purity_sub

        # # EM
        # print("EM (GMM)")
        # gmm_silhouette_sub = []
        # gmm_purity_sub = []

        # for k in n_cluster:
        #     gmm = GaussianMixture(n_components=k)
        #     labels = gmm.fit_predict(principalDf)
            
        #     plt.title("EM_4_normalize_onehot")
        #     plt.scatter(principalDf["PCA1"], principalDf["PCA2"], c=labels)
        #     plt.show()

        #     # silhouette (range -1~1)
        #     gmm_silhouette_sub.append(silhouette_score(principalDf, labels, metric='euclidean'))

        #     # purity
        #     gmm_purity_sub.append(purity_score(age, labels))
        #     print(time.time() - start_time)

        # gmm_silhouette[key] = gmm_silhouette_sub
        # gmm_purity[key] = gmm_purity_sub


        # # DBSCAN
        # print("dbscan")
        # dbscan_silhouette_sub = []
        # dbscan_purity_sub = []

        # for eps in DBSCAN_eps:
        #     dbscan = DBSCAN(eps=eps, min_samples=10)
        #     labels = dbscan.fit_predict(principalDf)
            
        #     plt.title("DBSCAN_0.2_minmax_ordinal")
        #     plt.scatter(principalDf["PCA1"], principalDf["PCA2"], c=labels)
        #     plt.show()

        #     # silhouette (range -1~1)
        #     try:
        #         current_silhouette = silhouette_score(principalDf, labels, metric='euclidean')
        #     except:
        #         current_silhouette = -5

        #     dbscan_silhouette_sub.append(current_silhouette)
        #     dbscan_purity_sub.append(purity_score(age, labels))
        #     print(time.time() - start_time)

        # dbscan_silhouette[key] = dbscan_silhouette_sub
        # dbscan_purity[key] = dbscan_purity_sub


        # Spectral
        # print("Spectral")
        # Spectral_silhouette_sub = []
        # Spectral_purity_sub = []

        # for k in n_cluster:
        #     Spectral_Clustering = SpectralClustering(n_clusters=k)
        #     labels = Spectral_Clustering.fit_predict(principalDf)

        #     Spectral_silhouette_sub.append(silhouette_score(principalDf, labels, metric='euclidean'))
        #     Spectral_purity_sub.append(purity_score(age, labels))
        #     print(time.time() - start_time)

        # Spectral_silhouette[key] = Spectral_silhouette_sub
        # Spectral_purity[key] = Spectral_purity_sub



    ## k-mean result
    # makeplot("KMeans_distance", kmeans_sumofDistance, n_cluster)
    # makeplot("KMeans_distance_original", kmeans_sumofDistance_original, n_cluster)
    # makeplot("KMeans_silhouette", kmeans_silhouette, n_cluster)
    # key, value = fineMaxValueKey(kmeans_silhouette)
    # print("k-means best silhouette : ", value, key)
    # makeplot("KMeans_purity", kmeans_purity, n_cluster)
    # key, value = fineMaxValueKey(kmeans_purity)
    # print("k-means best purity : ", value, key)


    # # GMM result
    # makeplot("EM_silhouette", gmm_silhouette, n_cluster)
    # makeplot("EM_purity", gmm_purity, n_cluster)
    # key, value = fineMaxValueKey(gmm_silhouette)
    # print("EM best silhouette : ", value, key)
    # key, value = fineMaxValueKey(gmm_purity)
    # print("EM best purity : ", value, key)

    # # DBSCAN result
    # makeplot("DBSCAN_silhouette", dbscan_silhouette, DBSCAN_eps)
    # makeplot("DBSCAN_purity", dbscan_purity, DBSCAN_eps)
    # key, value = fineMaxValueKey(dbscan_silhouette)
    # print("DBSCAN best silhouette : ", value, key)
    # key, value = fineMaxValueKey(dbscan_purity)
    # print("DBSCAN best purity : ", value, key)

    # # Spectral result
    # makeplot("Spectral_silhouette", Spectral_silhouette, n_cluster)
    # makeplot("Spectral_purity", Spectral_purity, n_cluster)
    # key, value = fineMaxValueKey(Spectral_silhouette)
    # print("Spectral best silhouette : ", value, key)
    # key, value = fineMaxValueKey(Spectral_purity)
    # print("Spectral best purity : ", value, key)


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

#Test purity
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def makeplot(title, dict, x_list):
    for key, value in dict.items():
        plt.plot(x_list, value, label=key)
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()

def fineMaxValueKey(dict):
    key = None
    largest = 0
    for keys, item in dict.items():
        if max(item) > largest:
            largest = max(item)
            key = keys

    return key, largest



if __name__ == "__main__":
	main()