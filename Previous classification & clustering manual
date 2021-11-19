# Machine_Learning_termproject
Machine learning team 1 Term project, Department of Software, Gachon Univ, South Korea. (2021 autumn semester)
<br> <br>

## Introduction
It has scaling+encoded datasets in 12 ways, and trains through four classification and clustering models, respectively.

## scikit-learn style manual

## Classification

> function Classification(data, target)

Exhaustive search over specified classification model and parameter values.

4 classification model will be used to find the best score using input dataset.

- Decision Tree ( entropy )
- Decision Tree ( gini )
- Support Vector Machine ( SVC )
- K Neighbor Nearest ( KNN )
 
For all models, we will use [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to find the best hyperparameter among the basic hyperparameter set. The basic hyperparameter set is below.

- Decision Tree
  - max_depth : [ 2, 3, 4, 5, 6, 7, 8 ]
- SVC
  - C : [ 1, 5, 10 ]
  - gamma : [ 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1 ]
- KNN
  - n_neighbors : [ 1, 2, 3, 4, 5 ]

Classification() will find the best model and hyperparameter, and plot all the testing scores and return the best model and its information.



### Parameters:

#### data : dataset values

This parameter will get pandas Dataframe object. Note that this data should not include the target attribute.

#### target : target values

This parameter will get pandas Dataframe object. Since there is 1 target attribute, this object's type will be like series type.



### Examples

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("health_checkup.csv")
data = Preprocessing(data, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])

best_answer = { 'best-score' : 0 };
for key, value in data.items():
    using_dataset = value[feature_list]
    data_y = using_dataset[target_feature]
    data_x = using_dataset.drop(columns=[target_feature])
    
    temp_answer = classification(data_x, data_y)
    if best_answer['best-score'] < temp_answer['best-score']:
        best_answer = temp_answer

print(best_answer)
```



---



## Classification

> function Classification(data, target, scaler=[original, standard, minmax, robust, maxabs, normalize], encoder=[ordinal, onehot], model=[DecisionTree(entropy), DecisionTree(gini), SVC, KNN], params)

Exhaustive search over specified classification model and parameter values.

For all models, we will use [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to find the best hyperparameter among the hyperparameter set.

Classification() will find the best model and hyperparameter, and plot all the testing scores and return the best model and its information.



### Parameters:

#### data : dataset values

This parameter will get pandas Dataframe object. Note that this data should not include the target attribute.

#### target : target values

This parameter will get pandas Dataframe object. Since there is 1 target attribute, this object's type will be like series type.

#### scaler: preprocessing scaler list

This parameter will get the list of scalers' name. you can use your own scaler list, but default value is below

- original
- standard
- minmax
- robust
- maxabs
- normalize

#### encoder: preprocessing encoder list

This parameter will get the list of encoders' name. you can use your own encoder list, but default value is below

- ordinal
- onehot

#### model: classification model list

This parameter will get the list of classification models' name. you can use your own classification model list, but default value is below

- DecisionTreeClassifier(criterion="entropy")
- DecisionTreeClassifier(criterion="gini")
- SVC(kernel='rbf')
- KNeighborsClassifier()

#### params: hyperparameter for all models

This parameter will get the list of all of the hyperparameters. there is no default value. Let me show the example of this.

- Decision Tree
  - max_depth : [ 2, 3, 4, 5, 6, 7, 8 ]
- SVC
  - C : [ 1, 5, 10 ]
  - gamma : [ 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1 ]
- KNN
  - n_neighbors : [ 1, 2, 3, 4, 5 ]



### Examples

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("health_checkup.csv")
data = Preprocessing(data, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])

best_answer = { 'best-score' : 0 };
for key, value in data.items():
    using_dataset = value[feature_list]
    data_y = using_dataset[target_feature]
    data_x = using_dataset.drop(columns=[target_feature])
    
    temp_answer = classification(data_x, data_y)
    if best_answer['best-score'] < temp_answer['best-score']:
        best_answer = temp_answer

print(best_answer)
```

---


<br> <br>

## Clustering

> function Clustering(data, target)

Exhaustive search over specified clustering model and parameter values.

4 clustering model will be used to find the best score using input dataset.

- k-means
- EM ( GMM )
- DBSCAN
- Spectral clustering
 

- k-means, EM, Spectral clusteirng
  - n_cluster : [ 2, 4, 8, 10, 13, 15 ]
- DBSCAN
  - DBSCAN_eps : [ 0.1, 0.2, 0.5, 0.7, 1 ]

Clustering() trains a model by parameter, and returns all test scores of the model in a list.



### Parameters:

#### data : dataset values

This parameter will get pandas Dataframe object. Note that this data should not include the target attribute.

#### target : target values

This parameter will get pandas Dataframe object. Since there is 1 target attribute, this object's type will be like series type.



### Examples

```python
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv("health_checkup.csv")
data = Preprocessing(data, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])

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


for key, value in data.items():
    target = using_dataset[target_feature]
    using_dataset = using_dataset.drop(columns=[target_feature])
    
    kmean_sum_of_squared_distances, kmean_silhouette_sub, kmeans_purity_sub, gmm_silhouette_sub, gmm_purity_sub, dbscan_silhouette_sub, dbscan_purity_sub, Spectral_silhouette_sub, Spectral_purity_sub = clustering(using_dataset, target)

    if "original" in key.split("_"):
        kmeans_sumofDistance_original[key] = kmean_sum_of_squared_distances
    else:
        kmeans_sumofDistance[key] = kmean_sum_of_squared_distances

    kmeans_silhouette[key] = kmean_silhouette_sub
    kmeans_purity[key] = kmeans_purity_sub


# k-mean result
makeplot("KMeans_distance", kmeans_sumofDistance, n_cluster)
makeplot("KMeans_distance_original", kmeans_sumofDistance_original, n_cluster)
makeplot("KMeans_silhouette", kmeans_silhouette, n_cluster)
key, value = fineMaxValueKey(kmeans_silhouette)
print("k-means best silhouette : ", value, key)
makeplot("KMeans_purity", kmeans_purity, n_cluster)
key, value = fineMaxValueKey(kmeans_purity)
print("k-means best purity : ", value, key)

```

---



## Clustering

> function Clustering(data, target, scaler=[original, standard, minmax, robust, maxabs, normalize], encoder=[ordinal, onehot], model=[k-means, EM (GMM), DBSCAN, Spectral_clustering], params)

Exhaustive search over specified clustering model and parameter values.

Clustering() trains a model by parameter, and returns all test scores of the model in a list.



### Parameters:

#### data : dataset values

This parameter will get pandas Dataframe object. Note that this data should not include the target attribute.

#### target : target values

This parameter will get pandas Dataframe object. Since there is 1 target attribute, this object's type will be like series type.

#### scaler: preprocessing scaler list

This parameter will get the list of scalers' name. you can use your own scaler list, but default value is below

- original
- standard
- minmax
- robust
- maxabs
- normalize

#### encoder: preprocessing encoder list

This parameter will get the list of encoders' name. you can use your own encoder list, but default value is below

- ordinal
- onehot

#### model: clustering model list

This parameter will get the list of clustering models' name. you can use your own clustering model list, but default value is below

- KMeans()
- GaussianMixture()
- DBSCAN(min_samples=10)
- SpectralClustering()

#### params: hyperparameter for all models

This parameter will get the list of all of the hyperparameters. there is no default value. Let me show the example of this.

- k-means, EM, Spectral clusteirng
  - n_cluster : [ 2, 4, 8, 10, 13, 15 ]
- DBSCAN
  - DBSCAN_eps : [ 0.1, 0.2, 0.5, 0.7, 1 ]



### Examples

```python
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv("health_checkup.csv")
data = Preprocessing(data, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])

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


for key, value in data.items():
    target = using_dataset[target_feature]
    using_dataset = using_dataset.drop(columns=[target_feature])
    
    kmean_sum_of_squared_distances, kmean_silhouette_sub, kmeans_purity_sub, gmm_silhouette_sub, gmm_purity_sub, dbscan_silhouette_sub, dbscan_purity_sub, Spectral_silhouette_sub, Spectral_purity_sub = clustering(using_dataset, target)

    if "original" in key.split("_"):
        kmeans_sumofDistance_original[key] = kmean_sum_of_squared_distances
    else:
        kmeans_sumofDistance[key] = kmean_sum_of_squared_distances

    kmeans_silhouette[key] = kmean_silhouette_sub
    kmeans_purity[key] = kmeans_purity_sub


# k-mean result
makeplot("KMeans_distance", kmeans_sumofDistance, n_cluster)
makeplot("KMeans_distance_original", kmeans_sumofDistance_original, n_cluster)
makeplot("KMeans_silhouette", kmeans_silhouette, n_cluster)
key, value = fineMaxValueKey(kmeans_silhouette)
print("k-means best silhouette : ", value, key)
makeplot("KMeans_purity", kmeans_purity, n_cluster)
key, value = fineMaxValueKey(kmeans_purity)
print("k-means best purity : ", value, key)
```

---



<br> <br>

## Third-Party License
 ### scikit-learn (https://github.com/scikit-learn/scikit-learn)
 ```
  BSD 3-Clause License

  Copyright (c) 2007-2021 The scikit-learn developers.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ```
