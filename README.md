# Machine_Learning_termproject
Machine learning team 1 Term project, Department of Software, Gachon Univ, South Korea. (2021 autumn semester)
<br> <br>

## Introduction
It has scaling+encoded datasets in 12 ways, and trains through four classification and clustering models, respectively.

## scikit-learn style manual

## AutoML
> autoML(type, feature, target, model, parameter, model_op = None):

The best score is obtained for the input dataset types and parameter sets.
Even if the parameter that gives the best score does not exist in the input parameter set, if the overall score is on the rise, linearly create a parameter and get a score.
"autoML" prints the parameter and data type that recorded the best score, and returns a table containing the best score for each parameter.


### Parameters:
#### type : int, task types
If 0, this task is “classification”, but if 1, this task is “clustering”. In the case of classification, it is necessary to distinguish it from clustering because GridSearchCV is used.

#### feature : dataset values
This parameter will get pandas Dataframe object. Note that this data should not include the target attribute.

#### target : target values
This parameter will get pandas Dataframe object. Since there is 1 target attribute, this object's type will be like series type.

#### model : estimator object
This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.

#### parameter : dict or list of dictionaries
Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionarie. It should be a list for Clusteinrg and a dictionary or list for Classification.

#### model_op : int, default=None
This option is used when using GridSearchCV in Classification. 


### Examples
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from autoML import autoML 
import matplotlib.pyplot as plt

data = pd.read_csv("health_checkup.csv")
data = Preprocessing(data, ['SEX'], ['HEIGHT', 'WEIGHT', 'WAIST', 'BP_HIGH', 'BP_LWST', 'BLDS'])

classifier = DecisionTreeClassifier(criterion='entropy')
dt_ent_result, plist = autoML(0, data, target, classifier, max_depth, 1)
makeplot("decision tree (entropy)_autoML", dt_ent_result, plist) 

```



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
