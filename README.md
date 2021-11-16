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









