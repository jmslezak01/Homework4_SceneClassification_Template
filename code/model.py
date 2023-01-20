import numpy as np
from sklearn import svm


class SVM(object):

    def __init__(self):
        self.model = svm.LinearSVC()
    
    def train(self, inputs, labels):
        """
        This function trains the binary SVM classifier based on the inputs and corresponding labels

        Inputs:
        inputs: a n x d numpy matrix where n is the number of samples and d is the number of features
        labels: a 1 x n numpy array where n is the number of samples

        Outputs:
        A 1 x d numpy array representing the weights from the model where d is the number is the number of features,
        and a number representing the bias from the model
        """
        try:
            if (np.size(np.unique(labels)) > 2):
                raise Exception
            self.model.fit(inputs, labels)
            weights = self.model.coef_
            bias = self.model.intercept_
            return weights, bias
        except Exception:
            print("More than 2 labels have been inputted, function only accepts 2 labels")
            raise