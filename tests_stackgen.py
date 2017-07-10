
import unittest
from Stacked_Generalization.stackgen import StackGen
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
import numpy as np


class StackGenTest(unittest.TestCase):

	def setUp(self):
		#Load data, initiate regression/classification estimators and StackGen object for tests

		name = self.shortDescription()

		if name == "Regression":
			boston = datasets.load_boston()
			X, Y = boston.data, boston.target
			self.X_TR, self.X_TE, self.y_TR, self.y_TE = train_test_split(X, Y, test_size=0.3, random_state=9)
			self.base_models = [Ridge(), Lasso(),RandomForestRegressor(random_state = 9)]
			self.stacker = RandomForestRegressor(random_state= 9)
			self.classification = False
			self.n_folds = 3 
			self.kf_random_state = 9
			self.stack_with_orig = True
			self.stratified = False
			self.save_results = 0
			self.n_classes = 1
		
		if name == "Classification":
			data = np.genfromtxt('CTG_dataset.csv', delimiter=',')
			X = data[1:,0:21]
			Y = (data[1:, 21:22]).flatten()
			self.X_TR, self.X_TE, self.y_TR, self.y_TE = train_test_split(X, Y, test_size=0.3, stratify = Y, random_state=9)

			self.base_models = [KNeighborsClassifier(n_neighbors=10), GaussianNB()]
			self.stacker = RandomForestClassifier(random_state= 9) #n_estimators = 300
			self.classification = True
			self.n_folds = 5
			self.kf_random_state = 9
			self.stack_with_orig = True
			self.stratified = True
			self.save_results = 0
			self.n_classes = np.unique(self.y_TR).shape[0]

		self.StackGenEstimator = StackGen(base_models = self.base_models, 
                              stacker = self.stacker,
                              classification = self.classification, n_folds = self.n_folds, stratified = self.stratified, 
                              kf_random_state = self.kf_random_state, save_results = self.save_results , stack_with_orig = self.stack_with_orig)
		self.StackGenEstimator.n_classes = self.n_classes


	#def tearDown(self):
	#	print ('End of test!', self.shortDescription())


	def test_regression_fit_base_models(self):
		"""Regression"""

		print ("Testing fit_base_models() for regression case.")

		dataset_blend_train, dataset_blend_oos = self.StackGenEstimator.fit_base_models(self.X_TR, self.y_TR, self.X_TE, self.y_TE)

        #Check the shape of blended train and oos datasets
		self.assertTupleEqual(np.array(dataset_blend_train).shape, (len(self.base_models), np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		self.assertTupleEqual(np.array(dataset_blend_oos).shape, (len(self.base_models), np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))
		print ("Test passed! \n")

	def test_regression_fit_stacker(self):
		"""Regression"""

		print ("Testing fit_stacker() for regression case.")

		blend_train_dummy = np.ones((len(self.base_models), np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		blend_oos_dummy = np.ones((len(self.base_models), np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		stacker_train_predictions, stacker_oos_predictions = self.StackGenEstimator.fit_stacker(blend_train_dummy,self.y_TR, blend_oos_dummy, self.y_TE)

		#Check the shape of stacker train and oos predictions
		self.assertTupleEqual(np.array(stacker_train_predictions).shape, (np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		self.assertTupleEqual(np.array(stacker_oos_predictions).shape, (np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))
		print ("Test passed! \n")


	def test_regression_fit_predict(self):
		"""Regression"""

		print ("Testing fit_predict() for regression case.")

		preds = self.StackGenEstimator.fit_predict(self.X_TR, self.y_TR, self.X_TE, self.y_TE)

		#Check the shape of final oos predictions
		self.assertTupleEqual(np.array(preds).shape, (np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		print ("Test passed! \n")


	def test_regression_cv_fit_model(self):
		"""Regression"""

		print ("Testing cv_fit_model() for regression case.")

		model_train_preds, model_oos_preds = self.StackGenEstimator.cv_fit_model(Ridge(),self.X_TR, self.y_TR, self.X_TE, self.y_TE)

		#Check the shape of train and oos predictions
		self.assertTupleEqual(np.array(model_train_preds).shape, (np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		self.assertTupleEqual(np.array(model_oos_preds).shape, (np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		print ("Test passed! \n")


	def test_classification_fit_base_models(self):
		"""Classification"""

		print ("Testing fit_base_models() for classification case.")

		dataset_blend_train, dataset_blend_oos = self.StackGenEstimator.fit_base_models(self.X_TR, self.y_TR, self.X_TE, self.y_TE)

        #Check the shape of blended train and oos datasets
		self.assertTupleEqual(np.array(dataset_blend_train).shape, (len(self.base_models), np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		self.assertTupleEqual(np.array(dataset_blend_oos).shape, (len(self.base_models), np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		self.assertTrue(np.max(dataset_blend_train)<=1)
		self.assertTrue(np.min(dataset_blend_train)>=0)
		self.assertTrue(np.max(dataset_blend_oos)<=1)
		self.assertTrue(np.min(dataset_blend_oos)>=0)

		print ("Test passed! \n")



	def test_classification_fit_stacker(self):
		"""Classification"""

		print ("Testing fit_stacker() for classification case.")

		blend_train_dummy = np.ones((len(self.base_models), np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		blend_oos_dummy = np.ones((len(self.base_models), np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		stacker_train_predictions, stacker_oos_predictions = self.StackGenEstimator.fit_stacker(blend_train_dummy,self.y_TR, blend_oos_dummy, self.y_TE)

		#Check the shape of stacker train and oos predictions
		self.assertTupleEqual(np.array(stacker_train_predictions).shape, (np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		self.assertTupleEqual(np.array(stacker_oos_predictions).shape, (np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		self.assertTrue(np.max(stacker_train_predictions)<=1)
		self.assertTrue(np.min(stacker_train_predictions)>=0)
		self.assertTrue(np.max(stacker_oos_predictions)<=1)
		self.assertTrue(np.min(stacker_oos_predictions)>=0)

		print ("Test passed! \n")


	def test_classification_fit_predict(self):
		"""Classification"""

		print ("Testing fit_predict() for classification case.")

		preds = self.StackGenEstimator.fit_predict(self.X_TR, self.y_TR, self.X_TE, self.y_TE)

		#Check the shape of final oos predictions
		self.assertTupleEqual(np.array(preds).shape, (np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		self.assertTrue(np.max(preds)<=1)
		self.assertTrue(np.min(preds)>=0)

		print ("Test passed! \n")


	def test_classification_cv_fit_model(self):
		"""Classification"""

		print ("Testing cv_fit_model() for classification case.")

		model_train_preds, model_oos_preds = self.StackGenEstimator.cv_fit_model(GaussianNB(), self.X_TR, self.y_TR, self.X_TE, self.y_TE)

		#Check the shape of train and oos predictions
		self.assertTupleEqual(np.array(model_train_preds).shape, (np.array(self.y_TR).shape[0], self.StackGenEstimator.n_classes))
		self.assertTupleEqual(np.array(model_oos_preds).shape, (np.array(self.y_TE).shape[0], self.StackGenEstimator.n_classes))

		self.assertTrue(np.max(model_train_preds)<=1)
		self.assertTrue(np.min(model_train_preds)>=0)
		self.assertTrue(np.max(model_oos_preds)<=1)
		self.assertTrue(np.min(model_oos_preds)>=0)

		print ("Test passed! \n")


if __name__ == '__main__':
	unittest.main()

