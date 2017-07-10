# StackGen

## Synopsis

StackGen is a python package that performs stacked generalization. Check out the seminal paper 'Stacked Generalization' by David Wolpert (Ref: Wolpert, David H. "Stacked generalization." Neural networks 5.2 (1992): 241-259) and the accompanying report.pdf in the repo for more information.

## Motivation

<p align="justify"> 
Stacked generalization or 'stacking' has become a staple technique in Kaggle competitions (and other data science competitions as well) among top competitors. This technique is used to  combine multiple machine learning models using a meta-learner and can greatly improve generalization error. However, the implementation of stacked generalization is rather tedious and confusing for beginners and pros alike. The StackGen package aims to hide the complexity behind this technique and provide an easy, familiar api for users to perform stacking. Please refer to report.pdf in this repo for more information. 
</p>

## Requirements
Python 3.3+ (support for namespace packages)  
NumPy  
scikit-learn  
unittest (for tests_stackgen.py)  


## Code Example

*Regression Example*
```python
from Stacked_Generalization.stackgen import StackGen
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets

boston = datasets.load_boston()
X, Y = boston.data, boston.target
X_TR, X_TE, y_TR, y_TE = train_test_split(X, Y, test_size=0.3, random_state=9)
    
stacked_regressor = StackGen(base_models = [Ridge(), Lasso(),RandomForestRegressor(random_state = 9)], 
                            stacker = RandomForestRegressor(random_state = 9), 
                            classification = False, 
                            n_folds = 3, 
                            kf_random_state = 9, 
                            stack_with_orig = False, 
                            save_results = 0)
final_result = stacked_regressor.fit_predict(X_TR, y_TR, X_TE, y_TE)
```

*Classification Example*
```python
from Stacked_Generalization.stackgen import StackGen
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
import numpy as np

data = np.genfromtxt('CTG_dataset.csv', delimiter=',')
X = data[1:,0:21]
Y = (data[1:, 21:22]).flatten()
X_TR, X_TE, y_TR, y_TE = train_test_split(X, Y, test_size=0.3, stratify = Y, random_state=9)

stacked_classifier = StackGen(base_models = [KNeighborsClassifier(n_neighbors=10), GaussianNB()], 
                              stacker = RandomForestClassifier(random_state= 9),
                              classification = True, 
                              n_folds = 5, 
                              stratified = True, 
                              kf_random_state = 9, 
                              stack_with_orig = True,
                              save_results = 0)
final_result = stacked_classifier.fit_predict(X_TR, y_TR, X_TE, y_TE)
```

*Trained models and predictions can be saved to disk by setting save_results flag to 1, 2 or 3* 

## Installation

Clone the project to your local machine and import the StackGen package in your scripts. Installation via pip will be available soon!

## Tests

Run tests_stackgen.py to test all methods of StackGen class for both classification and Regression cases.

## License

MIT License.
