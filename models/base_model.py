import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class Model:
	def __init__(self, parameters):
		# dictionary of parameters
		self.parameters = parameters

	def fit(self, data, targets):
		data, targets = check_X_y(data,targets)

		# Do your training here

		self.classes_ = unique_labels(targets)


		self.data_ = data
		self.targets_ = targets
		


		self.is_fitted_ = True
		return self

	def predict(self, data):

		data = check_array(data, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        closest = np.argmin(euclidean_distances(data, self.data_), axis=1)
		return self.targets_[closest]
