# Title : Stacked Generalization (StackGen)
# Author : Mahesh Raja <maheshr1990@yahoo.com>    


import numpy as np
import time
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


class StackGen(object):
    """Stacked Generalization
    
    This class implements stacked generalization or 'stacking'. 
    
    It fits several base models in a cross-validated fashion. It then combines 
    their training and out of sample(oos) predictions to generate new train and
    oos datasets which are also called meta features. A stacker model is then fit 
    using the new train dataset and makes predictions for the new out of sample dataset. 
    
    Why use this technique? This technique usually results in much lower 
    generalization error than what can be achieved by any of the individual models. 
    Many Kaggle competition winners consistently use stacking to lower their
    generalization error and boost their scores.
    
    Please refer to the user guide / documentation for more information.
    
    
    Parameters
    ----------  
    base_models : list, optional, default: [RandomForestRegressor()]
        List of base models for the stacking process. Models can be scikit-learn Classifier 
        or Regressor objectsaccording to the predictions task. Model objects must have fit and 
        predict(for regression tasks)/predict_proba(for classification tasks) methods.
        
    stacker : scikit-learn Classifier or Regressor object, optional, default: None
        Stacker object with fit and predict/predict_proba methods. 
        If stacker is None, base models' oos predictions are averaged.
    
    classification : boolean, optional, default: False
        Indicates whether the prediction task at hand is classification or regression task.
        
    n_folds : int, optional, default: 3
        Number of cross validation folds.
        
    stratified: boolean, optional, default: False
        This parameter is ignored when 'classification' is set to False.
        If True and classification is also True, stratified k-fold technique is used for cross-validation.
        If False, normal k-fold technique is used for cross-validation.
        
    kf_random_state: int, optional, default: None
        Seed for k-fold or stratified k-fold splits. Set to positive value for reproducible results.
        
    save_results: int, optional, default: 0
        This parameter can take values 0,1,2 and 3. It indicates different options for savings results to disk.
        Results are saved as a list of dictionaries(y) using joblib dump.
        0: Results not saved to disk
        1: Save stacker's out of sample predictions to disk 
        2: Save blended train and out of sample predictions of
           base models, and stacker's out of sample predictions to disk
        3: Option 2 + save fitted base and stacker models from every fold of cross validation

        ***Results are saved to disk under the following naming convention***
        StackGenResults_[list of base models]_stk-[stacking model]_[yyyy-mm-dd_hh-mm]_[savetype1,2,3].pkl
        
    stack_with_orig: Boolean, optional, default: False
        If True, the train and oos predictions of base models are merged (horizontally stacked) 
        with the original train and oos datasets respectively to generate new blended train 
        and blended oos datasets. 
        This is also a widely used technique that reduces generalization error.
    
    
    Attributes
    ----------
    results_ : list
        Results from the stacking process are stored as list of dictionaries. Contents of the results_ list 
        depend on the value of 'save_results' parameter.
        ---Value of save_results: Contents of results_ list---
        0: Stacker's out of sample predictions
        1: Stacker's out of sample predictions 
        2: Stacker's out of sample predictions, blended train and out of sample predictions of 
           base models
        3: Stacker's out of sample predictions, blended train and out of sample predictions of 
           base models, fitted base and stacker models across all CV folds
        
        
    Examples
    --------
    ### Regression example ###
    >>> from Stacked_Generalization.stackgen import StackGen
    >>> from sklearn.linear_model import Ridge, Lasso
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import datasets

    >>> boston = datasets.load_boston()
    >>> X, Y = boston.data, boston.target
    >>> X_TR, X_TE, y_TR, y_TE = train_test_split(X, Y, test_size=0.3, random_state=9)
    
    >>> stacked_regressor = StackGen([Ridge(), Lasso(),RandomForestRegressor(random_state = 9)], 
                            stacker = RandomForestRegressor(random_state = 9), classification = False, 
                            n_folds = 3, kf_random_state = 9, stack_with_orig = False, save_results = 0)
    >>> foo = stacked_regressor.fit_predict(X_TR, y_TR, X_TE, y_TE)



    ### Classification example ###
    >>> from Stacked_Generalization.stackgen import StackGen
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.naive_bayes import GaussianNB 
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np

    >>> data = np.genfromtxt('CTG_dataset.csv', delimiter=',')
    >>> X = data[1:,0:21]
    >>> Y = (data[1:, 21:22]).flatten()
    >>> X_TR, X_TE, y_TR, y_TE = train_test_split(X, Y, test_size=0.3, stratify = Y, random_state=9)
    
    >>> stacked_classifier = StackGen(base_models = [KNeighborsClassifier(n_neighbors=10), GaussianNB()], 
                              stacker = RandomForestClassifier(n_estimators = 300, random_state= 9),
                              classification = True, n_folds = 5, stratified = True, kf_random_state = 9, 
                              save_results = 0 , stack_with_orig = True)
    >>> foo = stacked_classifier.fit_predict(X_TR, y_TR, X_TE, y_TE)


    Notes
    -----
    Refer to the supplied jupyter notebooks for more examples.

    """
        
    #Dictionary of possible values for save_results. Ensures easier maintenance
    save_results_dict = {"not_saved":0 , "save_oos":1, "save_oos_blends":2, "save_all":3}


    def __init__(self, base_models = [RandomForestRegressor()], stacker = None, classification = False, n_folds = 3, 
                 stratified = False, kf_random_state = None , save_results = 0, stack_with_orig = False):
        """
        Initialize parameters and perform quick sanity checks. 
        """
        self.base_models = base_models
        self.stacker = stacker
        self.classification = classification
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = kf_random_state
        self.save_results = save_results
        self.stack_with_orig = stack_with_orig
        self.results_ = []
        
        #This parameter indicates the number of unique labels or classes. It will remain 1 for regression.
        self.n_classes = 1

        #Value/type checks for supplied arguments
        if self.save_results not in StackGen.save_results_dict.values():
            raise ValueError("Invalid argument. Please set save_results to 0, 1, 2 or 3.")

        if not isinstance(self.classification, bool):
            raise TypeError("Invalid argument. Please set classification flag to either True or False.")

        if not isinstance(self.stratified, bool):
            raise TypeError("Invalid argument. Please set stratified flag to either True or False.")

        if not isinstance(self.stack_with_orig, bool):
            raise TypeError("Invalid argument. Please set stack_with_orig flag to either True or False.")

        if not isinstance(self.n_folds, int):
            raise TypeError("Invalid argument. Please set n_folds to a positive integer.")

        if self.random_state is not None:
            if not isinstance(self.random_state, int):
                raise TypeError("Invalid argument. Please set kf_random_state to any integer.")
 

        #Quick check on model types and required methods
        if self.classification:
            for model in self.base_models:
                if not (hasattr(model, 'predict_proba') and hasattr(model, 'fit')):
                    raise TypeError("Cannot use model %s " % model.__repr__())
            if self.stacker is not None:
                if not (hasattr(self.stacker, 'predict_proba') and hasattr(self.stacker, 'fit')):
                    raise TypeError("Cannot use model %s " % self.stacker.__repr__())
        else:
            for model in self.base_models:
                if not (hasattr(model, 'predict') and hasattr(model, 'fit')):
                    raise TypeError("Cannot use model %s " % model.__repr__())
            if self.stacker is not None:
                if not (hasattr(self.stacker, 'predict') and hasattr(self.stacker, 'fit')):
                    raise TypeError("Cannot use model %s " % self.stacker.__repr__())
          

        #Set kfolds for cross-validation
        if self.n_folds < 2:
            raise ValueError("Please set number of cv folds to an integer greater than 1. Suggested value is between 3 an 10")
        if self.stratified and self.classification:
            self.kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        else:
            self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            
    def get_predictions(self, model, data):
        """Calculates and returns fitted model's predictions on supplied data.
        
        Parameters
        ----------
        model: (fitted) scikit-learn Classifier or Regressor object
            Fitted scikit-learn classifier or regressor model object used to make predictions on data.
        
        data: numpy array
            Data which the fitted model makes predictions on.

        Returns
        -------
        The method returns continuous valued predictions in case of regression and probabilities in case of classification.
        """
        
        if self.classification:
            return model.predict_proba(data)
        else:
            return model.predict(data)
    
    
    
    def get_error(self, y_actual, y_pred):
        """Calculates and returns the appropriate error metric based on supplied actual target values and predicted values.
        
        Parameters
        ----------
        y_actual: numpy array, shape: (number_of_samples, )
            Actual or true target values.
            
        y_pred: numpy array, shape: (number_of_samples, ) or (number_of_samples, number_of_unique_classes)
            Predicted target values. Can be continuous valued in case of regression and 
            probabilities in case of classification.

        Returns
        -------
        The method returns logistic loss (log loss or cross entropy) for classification tasks and
        returns mean squared error for regression tasks.
        """    
        if self.classification:
            return log_loss(y_actual, y_pred)
        else:
            return mean_squared_error(y_actual, y_pred)
        
        
    #def set_kfolds(self):
    #    """
    #    """
    #    if self.stratified and self.classification:
    #        self.kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
    #    else:
    #        self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                
                
        
    def fit_predict(self, data, labels, oos_data, oos_labels = None):
        """Pivotal method used by the end user to interact with the class and get final predictions.
        
        User passes data to the class through this method. The method in turn 
        calls other helper methods, fits models and makes predictions, saves results to disk (if required) 
        and returns the stacker's final oos predictions.
        
        Parameters
        ----------
        data: numpy array, shape: (number_of_samples, number_of_features)
            Training data. Corresponds to the traditional X.
            
        labels: numpy array, shape: (number_of_samples, )
            True target values or labels. Corresponds to the traditional y.

        oos_data: numpy array, shape: (oos_number_samples, number_of_features)
            This is the unseen/new/out of sample(oos) data for which predictions are to be made.  
    
        oos_labels: numpy array, optional, shape:(oos_number_samples, ), default: None
            Optional parameter. True target values for oos samples.
        
        Returns
        -------
        stacker_oos_predictions: numpy array, shape: (oos_number_samples, ) or (oos_number_samples, number_of_unique_classes)
            The stacker's oos predictions are returned. The predictions are either continuous valued in case 
            of regressions tasks or probabilities in case of classification tasks.
        """        
        
        #Check if the supplied arrays are numpy arrays
        if not (isinstance(data, np.ndarray) and isinstance(labels, np.ndarray) and isinstance(oos_data, np.ndarray)):
            raise TypeError("Only numpy arrays are accepted as inputs. Please check your inputs and try again.")
        if oos_labels is not None:
            if not isinstance(oos_labels, np.ndarray):
                raise TypeError("Only numpy arrays are accepted as inputs. Please check your inputs and try again.")

        #For classification tasks, set number of target classes
        if self.classification:
            self.n_classes = np.unique(labels).shape[0]
        
        #Fit base models
        dataset_blend_train, dataset_blend_oos = self.fit_base_models(data, labels, oos_data, oos_labels)
        
        #If user wants to blend the train and oos predictions with original features
        if (self.stack_with_orig) and (self.stacker is not None):
            dataset_blend_train.append(data)
            dataset_blend_oos.append(oos_data)


        #Fit stacker model with blended train and oos datasets
        stacker_train_predictions, stacker_oos_predictions = self.fit_stacker(dataset_blend_train, 
                                                            labels, dataset_blend_oos, oos_labels)
        
        #Append items to results_ list based on 'save_results' and/or save to disk
        self.results_.append({(type(self.stacker).__name__ + "_stacker_oos_predictions"):stacker_oos_predictions})
        base_models = ""
        for base_model in self.base_models:
            base_models = base_models + type(base_model).__name__ + "_"
            
        if self.save_results in [StackGen.save_results_dict["save_oos_blends"], StackGen.save_results_dict["save_all"]]:
            self.results_.append({(base_models + "dataset_blend_train"):np.hstack(dataset_blend_train)})
            self.results_.append({(base_models + "dataset_blend_oos"):np.hstack(dataset_blend_oos)})
            
        if self.save_results != StackGen.save_results_dict["not_saved"]:
            _ = joblib.dump(self.results_, ("StackGenResults_" + base_models + "stk-" + type(self.stacker).__name__ 
                                            + "_" + time.strftime("%Y-%m-%d_%H-%M") + "_savetype" + str(self.save_results)  
                                            + ".pkl"), compress = 3)
            print ("Results have been saved to disk!")

        return stacker_oos_predictions
    

            
    def fit_base_models(self, data, labels, oos_data, oos_labels=None):
        """Helper method called by fit_predict method that fits the training data to the base models and makes 
        predictions on train and oos datasets. The method then blends/combines the predictions and 
        returns the newly generated train and oos datasets.
        
        Parameters
        ----------
        data: numpy array, shape: (number_of_samples, number_of_features)
            Training data. Corresponds to the traditional X.
            
        labels: numpy array, shape: (number_of_samples, )
            True target values or labels. Corresponds to the traditional y.

        oos_data: numpy array, shape: (oos_number_samples, number_of_features)
            This is the unseen/new/out of sample(oos) data for which predictions are to be made.  
    
        oos_labels: numpy array, optional, shape:(oos_number_samples, ), default: None
            Optional parameter. True target values for oos samples.
        
        Returns
        -------
        dataset_blend_train: numpy array, shape: (number_of_samples, number_of_base_models) for Regression 
        or (number_of_samples, number_of_base_models * number_of_unique_classes) for Classification
            Newly created train dataset generated by horizontally stacking base models' predictions on the 
            training data in a cross-validated (out-of-fold predictions) fashion. 
    
        dataset_blend_oos: numpy array, shape: (oos_number_samples, number_of_base_models) for Regression 
        or (oos_number_samples, number_of_base_models * number_of_unique_classes) for Classification
            Newly created oos dataset generated by horizontally stacking base models' predictions on the oos data in a
            cross-validated (oos predictions of all k models across k-fold cv are averaged) fashion. 
        """
        dataset_blend_train = []
        dataset_blend_oos = []
        
        for i, model in enumerate(self.base_models):
            train_predictions, oos_predictions = self.cv_fit_model(model, data, labels, oos_data, oos_labels)
            dataset_blend_train.append(train_predictions)
            dataset_blend_oos.append(oos_predictions)

        return (dataset_blend_train, dataset_blend_oos)
              
        
        
    def fit_stacker(self, dataset_blend_train, labels, dataset_blend_oos, oos_labels = None):
        """ Helper method called by fit_predict method that fits the newly blended training dataset to the stacker model
        in a cross-validated fashion and returns the model's predictions on the newly blended train and oos datasets. 
        
        Parameters
        ----------
        dataset_blend_train: numpy array, shape: (number_of_samples, number_of_base_models) for Regression 
        or (number_of_samples, number_of_base_models * number_of_unique_classes) for Classification
            Training data for the stacker model. Size may increase if user decides to stack with original features.
            
        labels: numpy array, shape: (number_of_samples, )
            True target values or labels.

        dataset_blend_oos: numpy array, shape: (oos_number_samples, number_of_base_models) for Regression 
        or (oos_number_samples, number_of_base_models * number_of_unique_classes) for Classification
            Newly generated oos dataset. Size may increase if user decides to stack with original features.
    
        oos_labels: numpy array, optional, shape:(oos_number_samples, ), default: None
            Optional parameter. True target values for oos samples.
        
        Returns
        -------
        stacker_train_predictions: numpy array, shape: (number_of_samples, ) for Regression 
        or (number_of_samples, number_of_unique_classes) for Classification
            Predictions of stacker model on blended train data. 
    
        stacker_oos_predictions: numpy array, shape: (oos_number_samples, ) for Regression 
        or (oos_number_samples, number_of_unique_classes) for Classification
            Predictions of the stacker model on blended oos data.
        """
        
        #If stacker model is None, return average of base models' oos predictions 
        if self.stacker is None:
            print ("Stacking model not provided. Averaging predictions of base models...")
            stacker_oos_predictions = np.mean(dataset_blend_oos, axis=0)
            if oos_labels is not None:
                print ("OOS error ---> ", self.get_error(oos_labels, stacker_oos_predictions))
            return (None, stacker_oos_predictions)
        
        dataset_blend_train = np.hstack(dataset_blend_train)
        dataset_blend_oos = np.hstack(dataset_blend_oos)
        
        print ("Stacking base models using %s  ----> " % self.stacker.__repr__()) 
        stacker_train_predictions, stacker_oos_predictions = self.cv_fit_model(self.stacker, dataset_blend_train, 
                                                                               labels, dataset_blend_oos, oos_labels)
        return (stacker_train_predictions, stacker_oos_predictions)
            
        
        
    def cv_fit_model(self, model, data, labels, oos_data, oos_labels= None):
        """This helper method is the workhorse that actually fits models to the supplied training data, makes 
        predictions on the train and oos data in a cross-validated fashion and returns the predictions to the callers. 
        This method is called by both fit_base_models and fit_stacker methods to fit/get predictions from base models and 
        stacker model respectively.
        
        Parameters
        ----------
        data: numpy array, shape: (number_of_samples, number_of_features)
            Training data. Shape of the training data may vary according to the calling method.
            
        labels: numpy array, shape: (number_of_samples, )
            True target values or labels. 

        oos_data: numpy array, shape: (oos_number_samples, number_of_features)
            This is the unseen/new/out of sample(oos) data for which predictions are to be made.
            Shape of the training data may vary according to the calling method.
    
        oos_labels: numpy array, optional, shape:(oos_number_samples, ), default: None
            Optional parameter. True target values for oos samples.
        
        Returns
        -------
        train_predictions: numpy array, shape: (number_of_samples, ) for Regression 
        or (number_of_samples, number_of_unique_classes) for Classification
            Cross-validated train predictions.
    
        oos_predictions: numpy array, shape: (oos_number_samples, ) for Regression 
        or (oos_number_samples, number_of_unique_classes) for Classification
            Cross-validated oos predictions. 
        """
        
        print ("### Fitting model %s ###" % model.__repr__())
        
        scores = []
        train_predictions = np.zeros((data.shape[0], self.n_classes))
        oos_predictions = np.zeros((oos_data.shape[0], self.n_classes))

        if self.save_results == StackGen.save_results_dict["save_all"]:
            model_cv_dict = {}
        
        for i, (train_index, test_index) in enumerate(self.kf.split(data, labels)):           
            print ("Fold # of CV ->", i+1)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            model.fit(X_train, y_train)
            y_pred = self.get_predictions(model, X_test)     
            score = self.get_error(y_test, y_pred)
            print ("Error for fold", i+1, "is", score)
            scores.append(score)

            #Get out-of-fold training set predictions
            train_predictions[test_index, ] = y_pred.reshape((len(test_index), self.n_classes))
            
            #Get model predictions for oos data for every CV fold. Average it later.
            oos_predictions += (self.get_predictions(model, oos_data)).reshape((len(oos_data), self.n_classes))
            
            #Save fitted model to dictionary
            if self.save_results == StackGen.save_results_dict["save_all"]:
                model_cv_dict[type(model).__name__ + "_CV" + str(i+1)+ "_" + str(score)] = model
        
        
        #Calculate average of oos predictions
        oos_predictions/=self.n_folds
        
        print ("Average CV error is ", np.mean(scores))
        if oos_labels is not None:
            print ("OOS error ---> ", self.get_error(oos_labels, oos_predictions))
        print ("")
        
        #Append model dictionary to results_ list
        if self.save_results == StackGen.save_results_dict["save_all"]:
            self.results_.append(model_cv_dict)
            
        return (train_predictions, oos_predictions)        